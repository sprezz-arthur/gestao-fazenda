from django.urls import path

from . import views

urlpatterns = [
    path("fotoordenha/<int:object_pk>/custom_model_action/",
        views.detect_ordenhas, name="detect-ordenhas")
]
