from django.urls import path

from . import views

app_name = "llmcosts"
urlpatterns = [
    # ex: /llmcosts/
    path("", views.index, name="index"),
]