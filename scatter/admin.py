import csv
from django.contrib import admin
from django.http import HttpResponse
from django.utils.html import format_html
from .models import Shape


@admin.action(description='Export selected shapes as CSV (raw points)')
def export_shapes_csv(modeladmin, request, queryset):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="shapes_export.csv"'
    writer = csv.writer(response)
    writer.writerow(['shape_slug', 'shape_label', 'owner', 'x', 'y'])
    for shape in queryset.order_by('sort_order'):
        owner = shape.owner.username if shape.owner_id else 'public'
        for xy in (shape.points or []):
            writer.writerow([shape.slug, shape.label, owner, xy[0], xy[1]])
    return response


@admin.register(Shape)
class ShapeAdmin(admin.ModelAdmin):
    list_display   = ('emoji', 'label', 'slug', 'owner_display', 'color_swatch',
                      'n_points', 'sort_order', 'is_active')
    list_editable  = ('sort_order', 'is_active')
    list_filter    = ('is_active', 'owner')
    search_fields  = ('label', 'slug', 'owner__username')
    prepopulated_fields = {'slug': ('label',)}
    readonly_fields    = ('created_at', 'updated_at', 'n_points')
    actions = [export_shapes_csv]

    fieldsets = (
        (None, {
            'fields': ('slug', 'label', 'emoji', 'color', 'owner', 'sort_order', 'is_active'),
        }),
        ('Point data', {
            'fields': ('points',),
            'description': (
                'Raw dot coordinates as [[x, y], ...]. '
                'Run <code>python manage.py seed_shapes --slug &lt;slug&gt;</code> to regenerate.'
            ),
        }),
        ('Metadata', {
            'fields': ('n_points', 'created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )

    def get_queryset(self, request):
        # Staff see everything
        return super().get_queryset(request)

    @admin.display(description='Owner')
    def owner_display(self, obj):
        if obj.owner_id is None:
            return format_html('<span style="color:#6b7280">public</span>')
        return format_html('<code>{}</code>', obj.owner.username)

    @admin.display(description='Color')
    def color_swatch(self, obj):
        return format_html(
            '<span style="display:inline-block;width:16px;height:16px;border-radius:3px;'
            'background:{};vertical-align:middle;border:1px solid #ccc;margin-right:4px"></span>{}',
            obj.color, obj.color,
        )

    @admin.display(description='Points')
    def n_points(self, obj):
        return format_html('<code>{}</code>', obj.point_count)
