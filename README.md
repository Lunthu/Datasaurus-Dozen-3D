# Datasaurus 3D

A Django web application inspired by the [Datasaurus Dozen](https://en.wikipedia.org/wiki/Datasaurus_dozen).
Interactive 3D scatter plots where dinosaurs, dogs, cats, crabs, and skylines all
share identical summary statistics — yet look completely different.

## Features

- **8 shape silhouettes**: T-Rex, Dog, Cat, City Skyline, Star, Heart, Bullseye, Crab
- **10 axis metrics**: X/Y coordinates, distance from centroid, angle, local density,
  path curvature, momentum, rolling std dev, normalised rank
- **Interactive 3D scatter** via Plotly.js — drag, rotate, zoom
- **Compare all mode** — see all shapes side-by-side with identical axes
- **Live statistics panel** — watch mean/std/correlation barely change as you switch shapes
- Zero database required — all shapes generated from mathematical silhouette algorithms

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install django djangorestframework numpy scipy

# 3. Run the development server
python manage.py runserver

# 4. Open your browser
open http://127.0.0.1:8000
```

## Project structure

```
datasaurus/          ← Django project config
scatter/
  shapes/
    generators.py    ← Silhouette point generators (one per shape)
  data_service.py    ← Metric registry + data pipeline
  views.py           ← Django views + REST endpoints
  urls.py
  templates/
    scatter/
      index.html     ← Full UI (sidebar + Plotly chart)
manage.py
README.md
```

## REST API

| Endpoint | Description |
|---|---|
| `GET /api/shapes/` | List all available shapes |
| `GET /api/metrics/` | List all available axis metrics |
| `GET /api/data/{shape}/?x=…&y=…&z=…` | Point cloud data for Plotly |

### Example

```
GET /api/data/dinosaur/?x=x_coord&y=y_coord&z=distance
```

```json
{
  "shape_id": "dinosaur",
  "label": "T-Rex",
  "color": "#2D6A4F",
  "x": [...],
  "y": [...],
  "z": [...],
  "stats": {
    "n": 800,
    "mean_x": 54.22,
    "mean_y": 47.83,
    "std_x": 16.76,
    "corr_xy": -0.064
  }
}
```

## Adding a new shape

1. Open `scatter/shapes/generators.py`
2. Write a generator function `gen_myshape(n=800) -> (x, y)` that returns two numpy arrays
3. Add it to `SHAPE_GENERATORS` and `SHAPE_META` dictionaries
4. That's it — it appears automatically in the UI

## Adding a new metric

1. Open `scatter/data_service.py`
2. Add a lambda `(x, y) -> array` to `METRIC_FNS`
3. Add a human-readable label to `METRIC_LABELS`
