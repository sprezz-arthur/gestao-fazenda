from django.urls import path

from . import views

app_name = "example_labeller"

urlpatterns = [
    path("tool/", views.tool, name="tool"),
    path("tool/<int:pk>", views.tool, name="tool"),
    path(
        "labelling_tool_api/",
        views.LabellingToolAPI.as_view(),
        name="labelling_tool_api",
    ),
    path("get_api_labels/<int:image_id>/", views.get_api_labels, name="get_api_labels"),
]
