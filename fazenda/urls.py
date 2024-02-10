from django.urls import path

from . import views

urlpatterns = [
    path(
        "fotoordenha/<int:object_pk>/detect-ordenhas/",
        views.detectar_ordenhas,
        name="detect-ordenhas",
    ),
    path(
        "fotoordenha/<int:object_pk>/see-ordenhas/",
        views.see_ordenhas,
        name="see-ordenhas",
    ),
]
