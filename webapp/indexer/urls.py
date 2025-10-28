from django.urls import path

from . import views

app_name = "indexer"
urlpatterns = [
    # ex: /indexer/
    path("", views.index, name="index"),
    # ex: /generator/status/
    path("status/", views.status, name="status"),
    # ex: /indexer/process/
    path("process/", views.process, name="process"),
]