from django.utils.safestring import mark_safe

from django.contrib.admin.widgets import AdminFileWidget


class AdminImageWidget(AdminFileWidget):
    def render(self, name, value, attrs=None, renderer=None):
        output = []
        if value and getattr(value, "url", None):
            image_url = value.url
            file_name = str(value)
            output.append(
                '<a href="{}" target="_blank"><img src="{}" alt="{}" style="max-height: 200px;"/></a>'.format(
                    image_url, image_url, file_name
                )
            )
        output.append(super().render(name, value, attrs))
        return mark_safe("".join(output))
