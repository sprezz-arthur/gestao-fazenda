from django.urls import path

from . import views

urlpatterns = [
    path(
        "fotoordenha/<int:object_pk>/custom_model_action/",
        views.detect_ordenhas,
        name="detect-ordenhas",
    ),
    path(
        "fotoordenha/<int:object_pk>/custom_model_action/",
        views.got_to_ordenhas,
        name="go-to-ordenhas",
    ),
]
