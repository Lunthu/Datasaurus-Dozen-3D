import csv
import io
import json
import numpy as np
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Max
from django.http import JsonResponse, HttpResponse, Http404
from django.shortcuts import render, redirect
from django.utils.text import slugify
from django.views.decorators.http import require_POST

from .data_service import get_shape_data, list_shapes, list_metrics, invalidate_cache
from .models import Shape


# ── Main chart view ────────────────────────────────────────────────────────────

def index(request):
    user    = request.user
    shapes  = list_shapes(user)
    metrics = list_metrics()
    return render(request, 'scatter/index.html', {
        'shapes':        shapes,
        'metrics':       metrics,
        'shapes_json':   json.dumps(shapes),
        'metrics_json':  json.dumps(metrics),
        'default_shape': shapes[0]['id'] if shapes else '',
        'default_x':     'x_coord',
        'default_y':     'y_coord',
        'default_z':     'distance',
    })


# ── REST API ───────────────────────────────────────────────────────────────────

def api_shapes(request):
    return JsonResponse({'shapes': list_shapes(request.user)})


def api_metrics(request):
    return JsonResponse({'metrics': list_metrics()})


def api_data(request, shape_id):
    x_metric = request.GET.get('x', 'x_coord')
    y_metric = request.GET.get('y', 'y_coord')
    z_metric = request.GET.get('z', 'distance')
    try:
        data = get_shape_data(shape_id, x_metric, y_metric, z_metric, user=request.user)
        return JsonResponse(data)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=404)


def export_csv(request, shape_id):
    x_metric = request.GET.get('x', 'x_coord')
    y_metric = request.GET.get('y', 'y_coord')
    z_metric = request.GET.get('z', 'distance')
    try:
        data = get_shape_data(shape_id, x_metric, y_metric, z_metric, user=request.user)
    except ValueError as e:
        raise Http404(str(e))

    filename = f"datasaurus_{shape_id}_{x_metric}_{y_metric}_{z_metric}.csv"
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    writer = csv.writer(response)
    writer.writerow(['shape', 'label', x_metric, y_metric, z_metric])
    for x, y, z in zip(data['x'], data['y'], data['z']):
        writer.writerow([shape_id, data['label'], x, y, z])
    return response


# ── Auth views ─────────────────────────────────────────────────────────────────

def login_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect(request.GET.get('next', 'index'))
        error = 'Invalid username or password.'
    return render(request, 'scatter/auth.html', {
        'mode': 'login',
        'error': error,
    })


def register_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        password2 = request.POST.get('password2', '')
        if not username:
            error = 'Username is required.'
        elif len(username) < 3:
            error = 'Username must be at least 3 characters.'
        elif User.objects.filter(username=username).exists():
            error = 'That username is already taken.'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters.'
        elif password != password2:
            error = 'Passwords do not match.'
        else:
            user = User.objects.create_user(username=username, password=password)
            login(request, user)
            return redirect('index')
    return render(request, 'scatter/auth.html', {
        'mode': 'register',
        'error': error,
    })


def logout_view(request):
    logout(request)
    return redirect('index')


# ── CSV Import (login required) ────────────────────────────────────────────────

@login_required(login_url='/login/')
def import_page(request):
    return render(request, 'scatter/import.html')


@login_required(login_url='/login/')
@require_POST
def import_csv_view(request):
    """
    POST: parse an uploaded CSV, rescale to shared stats, save as user's private Shape.
    Returns JSON {ok, label, slug, n_points, mean_x, std_x, mean_y, std_y}
    """
    from .data_service import T_MEAN, T_STD

    upload = request.FILES.get('csv_file')
    if not upload:
        return JsonResponse({'ok': False, 'error': 'No file uploaded.'}, status=400)

    try:
        raw_text = upload.read().decode('utf-8-sig')
    except Exception as e:
        return JsonResponse({'ok': False, 'error': f'Could not read file: {e}'}, status=400)

    # Auto-detect delimiter
    sample = raw_text[:2000]
    delim = ','
    for d in (',', '\t', ';'):
        if d in sample:
            delim = d
            break

    def is_numeric(val):
        try:
            float(val); return True
        except (ValueError, TypeError):
            return False

    import csv as csv_module
    reader   = csv_module.reader(io.StringIO(raw_text), delimiter=delim)
    all_rows = [r for r in reader if any(c.strip() for c in r)]
    if not all_rows:
        return JsonResponse({'ok': False, 'error': 'CSV file is empty.'}, status=400)

    start     = 0 if is_numeric(all_rows[0][0]) else 1
    data_rows = all_rows[start:]
    if len(data_rows) < 2:
        return JsonResponse({'ok': False, 'error': 'Need at least 2 data rows.'}, status=400)

    n_cols = max(len(r) for r in data_rows)
    numeric_cols = []
    for c in range(n_cols):
        try:
            vals = [float(r[c]) for r in data_rows if c < len(r) and r[c].strip()]
            if len(vals) == len(data_rows):
                numeric_cols.append(vals)
                if len(numeric_cols) == 2:
                    break
        except ValueError:
            pass

    if len(numeric_cols) < 2:
        return JsonResponse({
            'ok': False,
            'error': f'Need 2 numeric columns, found {len(numeric_cols)}. Check delimiter and data.',
        }, status=400)

    x_raw = np.array(numeric_cols[0])
    y_raw = np.array(numeric_cols[1])

    def force_stats(arr, mean=T_MEAN, std=T_STD):
        s = arr.std()
        if s < 1e-10:
            return np.full_like(arr, mean)
        return np.clip((arr - arr.mean()) / s * std + mean, mean - 3*std, mean + 3*std)

    rescale = request.POST.get('rescale', '1') == '1'
    x_out   = force_stats(x_raw) if rescale else x_raw
    y_out   = force_stats(y_raw) if rescale else y_raw
    pts     = [[round(float(x), 4), round(float(y), 4)] for x, y in zip(x_out, y_out)]

    # Metadata
    stem  = upload.name.rsplit('.', 1)[0]
    label = (request.POST.get('label') or '').strip() or stem.replace('_', ' ').replace('-', ' ').title()
    emoji = (request.POST.get('emoji') or '◉').strip() or '◉'
    color = (request.POST.get('color') or '#6366f1').strip()
    if not color.startswith('#') or len(color) != 7:
        color = '#6366f1'

    # Build a slug that can't collide with public shapes:
    # user-supplied slug (or derived) + "_u{user_id}" suffix
    base_slug  = slugify((request.POST.get('slug') or '').strip() or label) or 'shape'
    final_slug = f"{base_slug}_u{request.user.pk}"

    max_order = Shape.objects.aggregate(m=Max('sort_order'))['m'] or 0

    shape, _ = Shape.objects.update_or_create(
        slug=final_slug,
        defaults=dict(
            label=label, emoji=emoji, color=color,
            points=pts, owner=request.user,
            sort_order=max_order + 1, is_active=True,
        ),
    )
    invalidate_cache(final_slug)

    x_arr = np.array([p[0] for p in pts])
    y_arr = np.array([p[1] for p in pts])
    return JsonResponse({
        'ok':       True,
        'label':    shape.label,
        'slug':     shape.slug,
        'n_points': len(pts),
        'mean_x':   round(float(x_arr.mean()), 3),
        'std_x':    round(float(x_arr.std()),  3),
        'mean_y':   round(float(y_arr.mean()), 3),
        'std_y':    round(float(y_arr.std()),  3),
    })
