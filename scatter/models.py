from django.db import models
from django.contrib.auth.models import User
from django.core.validators import RegexValidator


class Shape(models.Model):
    """
    A named silhouette rendered as a 3D scatter plot.

    Visibility rules:
      owner = None   → predefined / public (visible to everyone, including anonymous)
      owner = <user> → private (visible only to that user and admin staff)

    All dot coordinates are stored inline as a JSON array of [x, y] pairs.
    """
    slug = models.SlugField(
        unique=True,
        help_text="URL-safe identifier used in API paths, e.g. 'dinosaur'.",
    )
    label = models.CharField(max_length=64, help_text="Display name, e.g. 'T-Rex'")
    emoji = models.CharField(max_length=8, default='◉')
    color = models.CharField(
        max_length=7,
        default='#2563eb',
        validators=[RegexValidator(r'^#[0-9A-Fa-f]{6}$', 'Enter a valid hex colour.')],
        help_text="Accent hex colour for scatter plot markers.",
    )
    points = models.JSONField(
        default=list,
        help_text="List of [x, y] coordinate pairs.",
    )
    owner = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='shapes',
        help_text="Null = public predefined shape. Set to a user = private to that user.",
    )
    sort_order = models.PositiveSmallIntegerField(
        default=0,
        help_text="Lower numbers appear first in the UI.",
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Inactive shapes are hidden from the UI and API.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['sort_order', 'label']
        verbose_name = 'Shape'
        verbose_name_plural = 'Shapes'

    def __str__(self):
        owner_tag = f' [{self.owner.username}]' if self.owner_id else ' [public]'
        return f"{self.emoji} {self.label} ({self.slug}){owner_tag}"

    @property
    def point_count(self):
        return len(self.points) if self.points else 0

    def is_visible_to(self, user) -> bool:
        """Return True if this shape should be visible to the given user."""
        if self.owner_id is None:
            return True                     # public predefined
        if user and user.is_authenticated:
            if user.is_staff:
                return True                 # admin sees everything
            return self.owner_id == user.pk # own shapes only
        return False
