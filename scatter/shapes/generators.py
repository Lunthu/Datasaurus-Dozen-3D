"""
Shape silhouette point generators.
Each generator returns (x, y) arrays of raw silhouette points,
then get_shape_data() rescales them so all shapes share the same
summary statistics (mean, std, correlation) — the Datasaurus spirit in 3D.
"""

import numpy as np
from scipy.interpolate import splprep, splev

RNG = np.random.default_rng(42)

TARGET_MEAN_X = 54.26
TARGET_MEAN_Y = 47.83
TARGET_STD_X  = 16.76
TARGET_STD_Y  = 26.93
TARGET_CORR   = -0.064


def _normalise(x, y):
    """Rescale x,y to match target summary statistics exactly."""
    x = (x - x.mean()) / x.std() * TARGET_STD_X + TARGET_MEAN_X
    y = (y - y.mean()) / y.std() * TARGET_STD_Y + TARGET_MEAN_Y
    return x, y


def _spline_noise(pts, n=300, noise=0.6):
    """Fit a spline through control points and sample n noisy points."""
    pts = np.array(pts)
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
    t = np.linspace(0, 1, n)
    xi, yi = splev(t, tck)
    xi += RNG.normal(0, noise, n)
    yi += RNG.normal(0, noise, n)
    return xi, yi


# ─── Individual generators ────────────────────────────────────────────────────

def gen_dinosaur(n=800):
    """T-Rex silhouette approximated with spline control points."""
    body = [
        (10, 20), (18, 15), (30, 12), (45, 10), (60, 12),
        (72, 18), (78, 28), (75, 40), (65, 48), (55, 52),
        (45, 54), (35, 52), (28, 46), (22, 38), (14, 30), (10, 20),
    ]
    head = [
        (72, 18), (80, 10), (90, 8), (98, 12), (100, 20),
        (95, 28), (86, 32), (78, 28), (72, 18),
    ]
    # tail
    tail = [
        (10, 20), (0, 25), (-8, 32), (-10, 42), (-5, 50),
        (5, 48), (10, 40), (10, 20),
    ]
    # front small arm
    arm = [
        (60, 28), (68, 22), (72, 16), (68, 14),
        (64, 18), (60, 24), (60, 28),
    ]
    # legs
    leg1 = [
        (40, 54), (38, 65), (35, 78), (38, 82),
        (44, 82), (46, 72), (48, 60), (44, 54),
    ]
    leg2 = [
        (55, 52), (56, 65), (55, 78), (58, 82),
        (64, 80), (63, 68), (60, 56), (55, 52),
    ]

    parts = [body, head, tail, arm, leg1, leg2]
    counts = [300, 120, 80, 60, 120, 120]
    xs, ys = [], []
    for pts, cnt in zip(parts, counts):
        xi, yi = _spline_noise(pts, cnt, noise=0.8)
        xs.append(xi); ys.append(yi)
    x = np.concatenate(xs)[:n]
    y = np.concatenate(ys)[:n]
    return _normalise(x, -y)  # flip y so dino stands upright


def gen_dog(n=800):
    """Sitting dog silhouette."""
    body = [
        (20, 40), (30, 32), (50, 28), (70, 30), (82, 38),
        (85, 52), (80, 65), (70, 72), (55, 75), (40, 73),
        (28, 65), (20, 54), (20, 40),
    ]
    head = [
        (70, 30), (78, 20), (85, 12), (92, 10), (98, 16),
        (100, 26), (96, 36), (88, 38), (80, 36), (70, 30),
    ]
    ear_l = [
        (78, 20), (82, 8), (88, 2), (94, 4), (96, 14), (88, 20), (78, 20),
    ]
    ear_r = [
        (85, 12), (88, 4), (80, 0), (74, 6), (74, 14), (80, 16), (85, 12),
    ]
    tail = [
        (20, 40), (8, 32), (2, 20), (6, 10), (14, 12),
        (18, 22), (20, 34), (20, 40),
    ]
    front_leg = [
        (40, 73), (38, 82), (36, 92), (40, 96),
        (46, 96), (48, 86), (48, 75),
    ]
    back_leg = [
        (65, 74), (64, 84), (62, 94), (66, 96),
        (72, 94), (72, 84), (68, 74),
    ]
    parts = [body, head, ear_l, ear_r, tail, front_leg, back_leg]
    counts = [280, 150, 60, 60, 80, 90, 80]
    xs, ys = [], []
    for pts, cnt in zip(parts, counts):
        xi, yi = _spline_noise(pts, cnt, noise=0.7)
        xs.append(xi); ys.append(yi)
    x = np.concatenate(xs)[:n]
    y = np.concatenate(ys)[:n]
    return _normalise(x, -y)


def gen_cat(n=800):
    """Sitting cat silhouette."""
    body = [
        (25, 55), (30, 42), (45, 35), (62, 36), (75, 44),
        (78, 58), (74, 70), (62, 78), (48, 80), (34, 76),
        (26, 66), (25, 55),
    ]
    head = [
        (45, 35), (48, 22), (55, 14), (62, 16), (68, 26),
        (68, 36), (62, 36), (55, 34), (48, 34), (45, 35),
    ]
    ear_l = [(48, 22), (44, 10), (52, 8), (56, 18), (52, 22)]
    ear_r = [(62, 16), (66, 6), (72, 10), (70, 20), (64, 22)]
    tail = [
        (26, 66), (14, 72), (6, 82), (8, 90),
        (16, 90), (22, 82), (26, 72),
    ]
    front_legs = [
        (40, 80), (38, 92), (42, 98), (48, 98), (50, 88), (50, 80),
        (56, 80), (56, 92), (60, 98), (66, 98), (66, 88), (62, 80),
    ]
    parts = [body, head, ear_l, ear_r, tail, front_legs]
    counts = [280, 160, 50, 50, 80, 180]
    xs, ys = [], []
    for pts, cnt in zip(parts, counts):
        xi, yi = _spline_noise(pts, cnt, noise=0.6)
        xs.append(xi); ys.append(yi)
    x = np.concatenate(xs)[:n]
    y = np.concatenate(ys)[:n]
    return _normalise(x, -y)


def gen_skyline(n=800):
    """City skyline as a scatter of building outline points."""
    # Define buildings as (x_left, width, height) in a row
    buildings = [
        (0, 8, 40), (9, 10, 60), (20, 7, 45), (28, 12, 90),
        (41, 6, 55), (48, 14, 100), (63, 8, 70), (72, 5, 50),
        (78, 10, 80), (89, 6, 55), (96, 8, 40),
    ]
    xs, ys = [], []
    for (bx, bw, bh) in buildings:
        # outline: left wall, roof, right wall
        per = max(20, int(n * (bw + bh * 2) / (sum(b[1] + b[2] * 2 for b in buildings))))
        # left wall
        ly = np.linspace(0, bh, per // 3)
        xs.append(np.full(per // 3, bx) + RNG.normal(0, 0.3, per // 3))
        ys.append(ly)
        # roof
        rx = np.linspace(bx, bx + bw, per // 3)
        xs.append(rx)
        ys.append(np.full(per // 3, bh) + RNG.normal(0, 0.3, per // 3))
        # right wall
        xs.append(np.full(per // 3, bx + bw) + RNG.normal(0, 0.3, per // 3))
        ys.append(np.linspace(0, bh, per // 3))
        # windows (random dots inside building)
        nw = per // 4
        wx = RNG.uniform(bx + 1, bx + bw - 1, nw)
        wy = RNG.uniform(2, bh - 2, nw)
        xs.append(wx); ys.append(wy)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    idx = RNG.choice(len(x), min(n, len(x)), replace=False)
    return _normalise(x[idx], y[idx])


def gen_star(n=800):
    """5-pointed star silhouette."""
    k = 5
    angles_outer = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, k, endpoint=False)
    angles_inner = angles_outer + np.pi / k
    R_out, R_in = 1.0, 0.4

    pts = []
    for i in range(k):
        pts.append((np.cos(angles_outer[i]) * R_out, np.sin(angles_outer[i]) * R_out))
        pts.append((np.cos(angles_inner[i]) * R_in, np.sin(angles_inner[i]) * R_in))
    pts.append(pts[0])

    pts = np.array(pts)
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=False, k=1)
    t = np.linspace(0, 1, n)
    xi, yi = splev(t, tck)
    xi += RNG.normal(0, 0.02, n)
    yi += RNG.normal(0, 0.02, n)
    return _normalise(xi, yi)


def gen_heart(n=800):
    """Heart curve."""
    t = np.linspace(0, 2 * np.pi, n)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    x += RNG.normal(0, 0.3, n)
    y += RNG.normal(0, 0.3, n)
    return _normalise(x, y)


def gen_circle(n=800):
    """Bullseye — three concentric circles."""
    xs, ys = [], []
    for r, cnt in [(1.0, 340), (0.65, 240), (0.3, 120), (0.05, 100)]:
        t = np.linspace(0, 2 * np.pi, cnt)
        xs.append(np.cos(t) * r + RNG.normal(0, 0.02, cnt))
        ys.append(np.sin(t) * r + RNG.normal(0, 0.02, cnt))
    x = np.concatenate(xs)[:n]
    y = np.concatenate(ys)[:n]
    return _normalise(x, y)


def gen_crab(n=800):
    """Stylised crab silhouette."""
    body = [
        (30, 50), (38, 42), (52, 38), (68, 40), (76, 50),
        (74, 62), (62, 68), (48, 68), (36, 62), (30, 50),
    ]
    # claws
    claw_l = [
        (30, 50), (18, 44), (8, 38), (4, 28), (10, 22),
        (18, 26), (22, 36), (26, 44),
        # claw tips
        (8, 38), (2, 32), (4, 22),
    ]
    claw_r = [
        (76, 50), (88, 44), (98, 38), (102, 28), (96, 22),
        (88, 26), (84, 36), (80, 44),
        (98, 38), (104, 32), (102, 22),
    ]
    # legs (4 per side)
    legs_l = [(30, 50), (22, 56), (14, 62),
              (32, 54), (20, 62), (12, 70),
              (36, 60), (26, 68), (18, 76),
              (40, 64), (32, 72), (26, 80)]
    legs_r = [(76, 50), (84, 56), (92, 62),
              (74, 54), (86, 62), (94, 70),
              (70, 60), (80, 68), (88, 76),
              (66, 64), (74, 72), (80, 80)]
    # eyestalks
    eyes = [(48, 38), (46, 28), (44, 24),
            (58, 38), (60, 28), (62, 24)]

    parts_pts = [body, claw_l, claw_r, legs_l, legs_r, eyes]
    parts_raw = []
    for p in parts_pts:
        arr = np.array(p)
        parts_raw.append(arr)

    # spline body and claws, scatter legs
    xs, ys = [], []
    for pts, cnt in [(body, 200), (claw_l, 150), (claw_r, 150)]:
        xi, yi = _spline_noise(pts, cnt, noise=0.7)
        xs.append(xi); ys.append(yi)

    for leg_pts in [legs_l, legs_r]:
        arr = np.array(leg_pts)
        xs.append(arr[:, 0] + RNG.normal(0, 0.8, len(arr)))
        ys.append(arr[:, 1] + RNG.normal(0, 0.8, len(arr)))

    eye_arr = np.array(eyes)
    xs.append(eye_arr[:, 0] + RNG.normal(0, 0.3, len(eye_arr)))
    ys.append(eye_arr[:, 1] + RNG.normal(0, 0.3, len(eye_arr)))

    # fill with extra scattered body points
    body_arr = np.array(body)
    extra = 200
    t_extra = RNG.uniform(0, 1, extra)
    tck, _ = splprep([body_arr[:, 0], body_arr[:, 1]], s=0, per=True)
    bxi, byi = splev(t_extra, tck)
    xs.append(bxi + RNG.normal(0, 0.5, extra))
    ys.append(byi + RNG.normal(0, 0.5, extra))

    x = np.concatenate(xs)[:n]
    y = np.concatenate(ys)[:n]
    return _normalise(x, -y)


# ─── Registry ─────────────────────────────────────────────────────────────────

SHAPE_GENERATORS = {
    'dinosaur': gen_dinosaur,
    'dog':      gen_dog,
    'cat':      gen_cat,
    'skyline':  gen_skyline,
    'star':     gen_star,
    'heart':    gen_heart,
    'circle':   gen_circle,
    'crab':     gen_crab,
}

SHAPE_META = {
    'dinosaur': {'label': 'T-Rex',        'emoji': '🦕', 'color': '#2D6A4F'},
    'dog':      {'label': 'Dog',          'emoji': '🐕', 'color': '#8B5E3C'},
    'cat':      {'label': 'Cat',          'emoji': '🐈', 'color': '#6B4FA0'},
    'skyline':  {'label': 'City Skyline', 'emoji': '🏙', 'color': '#1A5276'},
    'star':     {'label': 'Star',         'emoji': '⭐', 'color': '#B7770D'},
    'heart':    {'label': 'Heart',        'emoji': '♥',  'color': '#C0392B'},
    'circle':   {'label': 'Bullseye',     'emoji': '◎',  'color': '#1A6B8A'},
    'crab':     {'label': 'Crab',         'emoji': '🦀', 'color': '#C0392B'},
}
