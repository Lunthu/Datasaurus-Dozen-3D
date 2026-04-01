"""
Management command: python manage.py import_shape <csv_file> [options]

Imports a CSV file as a new Shape, then rescales the X and Y coordinates
so they share the same summary statistics as every other shape:
    mean_x = 50.0,  std_x = 22.0
    mean_y = 50.0,  std_y = 22.0

CSV format (flexible):
  - Must contain exactly two numeric columns interpretable as X and Y.
  - First row may optionally be a header (auto-detected).
  - Delimiter: comma (default), tab, or semicolon (auto-detected).
  - Extra columns are ignored.

Example:
    python manage.py import_shape my_shape.csv --slug rocket --label "Rocket" --emoji 🚀 --color #8B0000
"""

import csv
import sys
import pathlib
import numpy as np
from django.core.management.base import BaseCommand, CommandError
from scatter.models import Shape

T_MEAN = 50.0
T_STD  = 22.0


def _force_stats(arr, mean=T_MEAN, std=T_STD):
    """Affine-rescale arr to have exactly (mean, std), then soft-clip."""
    s = arr.std()
    if s < 1e-10:
        return np.full_like(arr, mean)
    rescaled = (arr - arr.mean()) / s * std + mean
    return np.clip(rescaled, mean - 3 * std, mean + 3 * std)


def _detect_delimiter(sample: str) -> str:
    for delim in (',', '\t', ';'):
        if delim in sample:
            return delim
    return ','


def _parse_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a CSV file and return (x, y) numpy arrays.
    Auto-detects delimiter and header row.
    """
    text = pathlib.Path(path).read_text(encoding='utf-8-sig')
    delim = _detect_delimiter(text[:1000])

    reader = csv.reader(text.splitlines(), delimiter=delim)
    rows   = [row for row in reader if any(c.strip() for c in row)]

    if not rows:
        raise CommandError('CSV file is empty.')

    # Try to parse first row as floats; if it fails, it's a header
    def is_numeric_row(row):
        try:
            [float(c) for c in row if c.strip()]
            return True
        except ValueError:
            return False

    start = 0 if is_numeric_row(rows[0]) else 1

    # Find two numeric columns (first two that parse as float across all rows)
    data_rows = rows[start:]
    if not data_rows:
        raise CommandError('No data rows found after header.')

    n_cols = max(len(r) for r in data_rows)
    col_valid = []
    for c in range(n_cols):
        try:
            vals = [float(r[c]) for r in data_rows if c < len(r) and r[c].strip()]
            if len(vals) > 0:
                col_valid.append((c, vals))
        except ValueError:
            pass

    if len(col_valid) < 2:
        raise CommandError(
            f'Need at least 2 numeric columns, found {len(col_valid)}. '
            'Check your delimiter and data.'
        )

    _, x_vals = col_valid[0]
    _, y_vals = col_valid[1]

    # Align lengths
    n = min(len(x_vals), len(y_vals))
    return np.array(x_vals[:n]), np.array(y_vals[:n])


class Command(BaseCommand):
    help = 'Import a CSV as a new Shape, rescaling points to match shared statistics.'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str,
            help='Path to the CSV file to import.')
        parser.add_argument('--slug', type=str, default=None,
            help='URL-safe slug for the new shape (auto-derived from label if omitted).')
        parser.add_argument('--label', type=str, default=None,
            help='Display name (defaults to the CSV filename without extension).')
        parser.add_argument('--emoji', type=str, default='◉',
            help='Emoji for the shape (default: ◉).')
        parser.add_argument('--color', type=str, default='#6366f1',
            help='Hex accent colour (default: #6366f1).')
        parser.add_argument('--overwrite', action='store_true',
            help='Replace an existing shape with the same slug.')
        parser.add_argument('--no-rescale', action='store_true',
            help='Store raw coordinates without rescaling to shared statistics.')
        parser.add_argument('--preview', action='store_true',
            help='Print stats and first 5 points without saving to the database.')

    def handle(self, *args, **options):
        from scatter.data_service import invalidate_cache
        from django.utils.text import slugify

        csv_path = options['csv_file']
        if not pathlib.Path(csv_path).exists():
            raise CommandError(f'File not found: {csv_path}')

        # ── Parse ────────────────────────────────────────────────────────────
        self.stdout.write(f'Parsing {csv_path} …')
        x_raw, y_raw = _parse_csv(csv_path)
        self.stdout.write(f'  Read {len(x_raw)} points.')
        self.stdout.write(
            f'  Raw   x: mean={x_raw.mean():.3f}  std={x_raw.std():.3f}  '
            f'range=[{x_raw.min():.2f}, {x_raw.max():.2f}]'
        )
        self.stdout.write(
            f'  Raw   y: mean={y_raw.mean():.3f}  std={y_raw.std():.3f}  '
            f'range=[{y_raw.min():.2f}, {y_raw.max():.2f}]'
        )

        # ── Rescale ──────────────────────────────────────────────────────────
        if options['no_rescale']:
            x_out, y_out = x_raw, y_raw
            self.stdout.write('  Rescaling: skipped (--no-rescale).')
        else:
            x_out = _force_stats(x_raw)
            y_out = _force_stats(y_raw)
            self.stdout.write(
                f'  Rescaled x: mean={x_out.mean():.3f}  std={x_out.std():.3f}'
            )
            self.stdout.write(
                f'  Rescaled y: mean={y_out.mean():.3f}  std={y_out.std():.3f}'
            )

        pts = [[round(float(x), 4), round(float(y), 4)]
               for x, y in zip(x_out, y_out)]

        # ── Preview mode ─────────────────────────────────────────────────────
        if options['preview']:
            self.stdout.write('\nFirst 5 points after rescaling:')
            for p in pts[:5]:
                self.stdout.write(f'  {p}')
            self.stdout.write('\n(preview only — nothing saved)')
            return

        # ── Resolve label / slug ─────────────────────────────────────────────
        stem  = pathlib.Path(csv_path).stem
        label = options['label'] or stem.replace('_', ' ').replace('-', ' ').title()
        slug  = options['slug']  or slugify(label)

        if not slug:
            raise CommandError('Could not derive a valid slug. Use --slug.')

        # ── Sort order ───────────────────────────────────────────────────────
        max_order = Shape.objects.aggregate(m=__import__('django.db.models', fromlist=['Max']).Max('sort_order'))['m']
        next_order = (max_order or 0) + 1

        # ── Save ─────────────────────────────────────────────────────────────
        existing = Shape.objects.filter(slug=slug).first()
        if existing and not options['overwrite']:
            raise CommandError(
                f"A shape with slug '{slug}' already exists (id={existing.pk}). "
                "Use --overwrite to replace it."
            )

        shape, created = Shape.objects.update_or_create(
            slug=slug,
            defaults=dict(
                label=label,
                emoji=options['emoji'],
                color=options['color'],
                points=pts,
                sort_order=next_order,
                is_active=True,
            ),
        )
        invalidate_cache(slug)

        action = 'Created' if created else 'Updated'
        self.stdout.write(self.style.SUCCESS(
            f'\n{action} shape: {shape}  ({len(pts)} points)'
        ))
        self.stdout.write(
            f'  Visible at: /  (slug="{slug}", sort_order={shape.sort_order})'
        )
