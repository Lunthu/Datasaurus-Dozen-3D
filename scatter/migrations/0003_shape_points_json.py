"""
Migration: collapse ShapePoint rows into Shape.points JSON field.

Uses raw SQL for the data copy to avoid ORM descriptor conflicts
that arise when both the old FK reverse relation (named 'points')
and the new JSONField (also named 'points') exist simultaneously
in the migration state.
"""

import json
from django.db import migrations, models


def copy_points_to_json(apps, schema_editor):
    db = schema_editor.connection
    with db.cursor() as cur:
        # Fetch all shape IDs
        cur.execute("SELECT id FROM scatter_shape")
        shape_ids = [row[0] for row in cur.fetchall()]

        for shape_id in shape_ids:
            cur.execute(
                "SELECT x, y FROM scatter_shapepoint WHERE shape_id = %s ORDER BY id",
                [shape_id],
            )
            pts = [[round(x, 4), round(y, 4)] for x, y in cur.fetchall()]
            cur.execute(
                "UPDATE scatter_shape SET points = %s WHERE id = %s",
                [json.dumps(pts), shape_id],
            )


def reverse_json_to_points(apps, schema_editor):
    db = schema_editor.connection
    with db.cursor() as cur:
        cur.execute("SELECT id, points FROM scatter_shape")
        for shape_id, pts_json in cur.fetchall():
            pts = json.loads(pts_json) if isinstance(pts_json, str) else pts_json
            for xy in (pts or []):
                cur.execute(
                    "INSERT INTO scatter_shapepoint (shape_id, x, y) VALUES (%s, %s, %s)",
                    [shape_id, xy[0], xy[1]],
                )


class Migration(migrations.Migration):

    dependencies = [
        ('scatter', '0002_replace_shapepart_with_shapepoint'),
    ]

    operations = [
        migrations.AddField(
            model_name='shape',
            name='points',
            field=models.JSONField(
                default=list,
                help_text='List of [x, y] coordinate pairs.',
            ),
        ),
        migrations.RunPython(copy_points_to_json, reverse_json_to_points),
        migrations.DeleteModel(name='ShapePoint'),
    ]
