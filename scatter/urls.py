from django.urls import path
from . import views

urlpatterns = [
    # Main
    path('',                                views.index,           name='index'),
    # Auth
    path('login/',                          views.login_view,      name='login'),
    path('register/',                       views.register_view,   name='register'),
    path('logout/',                         views.logout_view,     name='logout'),
    # Import (login required)
    path('import/',                         views.import_page,     name='import_page'),
    path('import/upload/',                  views.import_csv_view, name='import_csv'),
    # API
    path('api/shapes/',                     views.api_shapes,      name='api_shapes'),
    path('api/metrics/',                    views.api_metrics,     name='api_metrics'),
    path('api/data/<str:shape_id>/',        views.api_data,        name='api_data'),
    path('api/data/<str:shape_id>/export/', views.export_csv,      name='export_csv'),
]
