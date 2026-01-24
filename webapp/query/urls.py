from django.urls import path

from . import views

app_name = "query"
urlpatterns = [
    # ex: /query/
    path("", views.index, name="index"),
    # ex: /query/status/
    path("status/", views.status, name="status"),
    # ex: /query/process/
    path("process/", views.process, name="process"),
    # ex: /query/settings/
    path("settings/", views.settings, name="settings"),
]