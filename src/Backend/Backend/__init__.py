from .celery import app as celery_app

__all__ = ('celery_app',)   # to ensure Celery loads whenever Django starts up