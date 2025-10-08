from django.urls import path

from . import views

app_name = "generator"
urlpatterns = [
    # ex: /generator/
    path("", views.index, name="index"),
    # ex: /generator/status/
    path("status/", views.status, name="status"),
    # ex: /generator/results/
    path("results/", views.results, name="results"),
    # ex: /generator/process/
    path("process/", views.process, name="process"),
]