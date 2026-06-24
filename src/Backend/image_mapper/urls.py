from django.urls import path
from . import views

urlpatterns = [
    path('transform/', views.transform_image_view, name='transform_image'),
    path('status/<int:task_id>/', views.task_status_view, name='task_status'), # Tracking route
]