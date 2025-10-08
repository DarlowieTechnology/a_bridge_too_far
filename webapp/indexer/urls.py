from django.urls import path

from . import views

app_name = "indexer"
urlpatterns = [
    # ex: /indexer/
    path("", views.index, name="index"),
    # ex: /indexer/process/
    path("process/", views.process, name="process"),
]