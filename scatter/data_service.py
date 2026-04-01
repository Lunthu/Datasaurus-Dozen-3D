"""
Data service layer.

Shapes are stored as plain (x, y) ShapePoint rows — no generation logic here.
Pipeline:
  1. SELECT all ShapePoint rows for the shape.
  2. Apply the chosen axis metric transforms.
  3. Force all axes to identical summary statistics (the Datasaurus trick).
"""

import numpy as np
from .models import Shape

# ── Datasaurus target statistics ─────────────────────────────────────────────
T_MEAN = 50.0
T_STD  = 22.0

# ── Metric registry ───────────────────────────────────────────────────────────

def _rolling_std(arr, w=12):
    out = np.empty(len(arr))
    for i in range(len(arr)):
        out[i] = arr[max(0, i - w):i + 1].std()
    return out


def _local_density(x, y, k=15):
    from scipy.spatial import KDTree
    tree = KDTree(np.column_stack([x, y]))
    dists, _ = tree.query(np.column_stack([x, y]), k=k + 1)
    return 1.0 / (dists[:, 1:].mean(axis=1) + 1e-6)


def _curvature(x, y):
    dx, dy   = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = np.where((dx**2 + dy**2)**1.5 < 1e-8, 1e-8, (dx**2 + dy**2)**1.5)
    return np.abs(dx * ddy - dy * ddx) / denom


METRIC_FNS = {
    'x_coord':    lambda x, y: x.copy(),
    'y_coord':    lambda x, y: y.copy(),
    'distance':   lambda x, y: np.sqrt((x - x.mean())**2 + (y - y.mean())**2),
    'angle':      lambda x, y: np.degrees(np.arctan2(y - y.mean(), x - x.mean())) % 360,
    'density':    lambda x, y: _local_density(x, y),
    'curvature':  lambda x, y: _curvature(x, y),
    'momentum_x': lambda x, y: np.abs(np.gradient(x)),
    'momentum_y': lambda x, y: np.abs(np.gradient(y)),
    'rolling_std':lambda x, y: _rolling_std(x),
    'norm_rank':  lambda x, y: (np.argsort(np.argsort(x)) / max(len(x) - 1, 1)) * 100,
}

METRIC_LABELS = {
    'x_coord':    'X coordinate',
    'y_coord':    'Y coordinate',
    'distance':   'Distance from centroid',
    'angle':      'Angle from centroid (°)',
    'density':    'Local point density',
    'curvature':  'Path curvature',
    'momentum_x': 'X momentum',
    'momentum_y': 'Y momentum',
    'rolling_std':'Rolling std dev',
    'norm_rank':  'Normalised rank',
}


# ── Stat enforcement ──────────────────────────────────────────────────────────

def _force_stats(arr, mean=T_MEAN, std=T_STD):
    """Affine-rescale arr so it has exactly (mean, std), then soft-clip."""
    s = arr.std()
    if s < 1e-10:
        return np.full_like(arr, mean)
    rescaled = (arr - arr.mean()) / s * std + mean
    return np.clip(rescaled, mean - 3 * std, mean + 3 * std)


# ── Point cache ───────────────────────────────────────────────────────────────
# Keyed by shape slug. Cleared when seed_shapes runs or points are edited.
_POINT_CACHE: dict = {}


def invalidate_cache(slug: str | None = None):
    if slug:
        _POINT_CACHE.pop(slug, None)
    else:
        _POINT_CACHE.clear()


def _load_points(shape: Shape):
    """Read (x, y) arrays directly from the Shape.points JSON field."""
    if shape.slug not in _POINT_CACHE:
        pts = shape.points
        if not pts:
            raise ValueError(f"Shape '{shape.slug}' has no points.")
        arr = np.array(pts, dtype=float)
        x, y = arr[:, 0], arr[:, 1]
        # Normalise to zero-mean/unit-std so all shapes sit in the same
        # coordinate space before metric transforms and stat enforcement.
        x = (x - x.mean()) / max(x.std(), 1e-10)
        y = (y - y.mean()) / max(y.std(), 1e-10)
        _POINT_CACHE[shape.slug] = (x, y)
    return _POINT_CACHE[shape.slug]


# ── Public API ────────────────────────────────────────────────────────────────

def get_shape_data(shape_id: str, x_metric: str, y_metric: str, z_metric: str, user=None) -> dict:
    try:
        shape = Shape.objects.get(slug=shape_id, is_active=True)
    except Shape.DoesNotExist:
        raise ValueError(f"Unknown or inactive shape: '{shape_id}'")
    if not shape.is_visible_to(user):
        raise ValueError(f"Unknown or inactive shape: '{shape_id}'")

    for m in (x_metric, y_metric, z_metric):
        if m not in METRIC_FNS:
            raise ValueError(f"Unknown metric: '{m}'")

    rx, ry = _load_points(shape)

    ax = _force_stats(METRIC_FNS[x_metric](rx, ry))
    ay = _force_stats(METRIC_FNS[y_metric](rx, ry))
    az = _force_stats(METRIC_FNS[z_metric](rx, ry))

    return {
        'shape_id': shape.slug,
        'label':    shape.label,
        'emoji':    shape.emoji,
        'color':    shape.color,
        'x': [round(v, 3) for v in ax.tolist()],
        'y': [round(v, 3) for v in ay.tolist()],
        'z': [round(v, 3) for v in az.tolist()],
        'stats': {
            'n':       len(ax),
            'mean_x':  round(float(ax.mean()),               3),
            'mean_y':  round(float(ay.mean()),               3),
            'mean_z':  round(float(az.mean()),               3),
            'std_x':   round(float(ax.std()),                3),
            'std_y':   round(float(ay.std()),                3),
            'std_z':   round(float(az.std()),                3),
            'corr_xy': round(float(np.corrcoef(ax, ay)[0, 1]), 3),
            'corr_xz': round(float(np.corrcoef(ax, az)[0, 1]), 3),
            'corr_yz': round(float(np.corrcoef(ay, az)[0, 1]), 3),
        },
    }


def _visible_shapes_qs(user=None):
    """Return a queryset of shapes visible to the given user."""
    from django.db.models import Q
    qs = Shape.objects.filter(is_active=True)
    if user and getattr(user, 'is_authenticated', False):
        if getattr(user, 'is_staff', False):
            return qs  # admin sees all
        # authenticated non-staff: public shapes + own shapes
        return qs.filter(Q(owner__isnull=True) | Q(owner=user))
    # anonymous: public shapes only
    return qs.filter(owner__isnull=True)


def list_shapes(user=None):
    return [
        {'id': s.slug, 'label': s.label, 'emoji': s.emoji, 'color': s.color}
        for s in _visible_shapes_qs(user).order_by('sort_order', 'label')
    ]


def list_metrics():
    return [{'id': mid, 'label': METRIC_LABELS[mid]} for mid in METRIC_FNS]
