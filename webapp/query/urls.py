from django.urls import path

from . import views

app_name = "query"
urlpatterns = [
    # ex: /query/
    path("", views.index, name="index"),
    # ex: /query/status/
    path("status/", views.status, name="status"),
    # ex: /query/results/
    path("results/", views.results, name="results"),
    # ex: /query/process/
    path("process/", views.process, name="process"),
]