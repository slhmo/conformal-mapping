import math
from django.db import models


class ConformalTask(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    # file storage(images are stored somewhere)
    source_image = models.ImageField(upload_to='sources/')
    transformed_image = models.ImageField(upload_to='results/', blank=True, null=True)

    # state
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='PENDING')
    error_message = models.TextField(blank=True, null=True)

    custom_expression = models.TextField(
        blank=True,
        null=True,
        help_text="Custom Python/NumPy math expression using the variable 'z'"
    )

    # pipeline functions
    transform_type = models.CharField(
        max_length=255,
        default='THREE BLUE DROSTE',
        help_text="Comma-separated chain sequence of transformation steps (e.g., MOBIUS,EXP)"
    )

    # Center offsets
    center_x = models.IntegerField(blank=True, null=True, help_text="Custom X focus center")
    center_y = models.IntegerField(blank=True, null=True, help_text="Custom Y focus center")

    math_scale = models.FloatField(default=math.pi)
    img_size_scale = models.IntegerField(default=1)
    source_zoom = models.FloatField(blank=True, null=True)

    # custom domain intervals
    x_bound_min = models.FloatField(blank=True, null=True)
    x_bound_max = models.FloatField(blank=True, null=True)
    y_bound_min = models.FloatField(blank=True, null=True)
    y_bound_max = models.FloatField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Task {self.id} [{self.transform_type}] - {self.status}"