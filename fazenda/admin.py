from django.contrib import admin
from fazenda import models

from dragndrop_related.views import DragAndDropRelatedImageMixin


@admin.register(models.Fazenda)
class FazendaAdmin(admin.ModelAdmin):
    ...


@admin.register(models.Vaca)
class VacaAdmin(admin.ModelAdmin):
    list_display = ["zero_numero", "prefixo", "nome", "image_tag"]

    @admin.display(description="Número")
    def zero_numero(self, obj):
        return f"{obj.numero:03d}"

    @admin.display(description="Foto")
    def image_tag(self, obj):
        from django.utils.html import mark_safe

        return mark_safe(
            """
            <img src="/midia/%s" width="75" height="75"/>"""
            % (obj.foto)
        )


class ImageInline(admin.StackedInline):
    extra = 0
    model = models.FotoOrdenha


@admin.register(models.FotoOrdenha)
class FotoOrdenhaAdmin(admin.ModelAdmin):
    ...


@admin.register(models.FichaOrdenha)
class FichaOrdenhaAdmin(DragAndDropRelatedImageMixin, admin.ModelAdmin):
    inlines = [ImageInline]
    fields = ["data"]
    list_display = ["data", "image", "linhas", "bbox"]

    @admin.display(description="Foto")
    def image(self, obj):
        from django.utils.html import mark_safe

        try:
            image = obj.images.first().image
        except Exception:
            image = None

        return mark_safe(
            """
            <img src="/midia/%s" height="200" alt="Original"/>"""
            % (image)
        )

    @admin.display(description="Linhas")
    def linhas(self, obj):
        from django.utils.html import mark_safe

        try:
            image = obj.images.first().linhas
        except Exception:
            image = None

        return mark_safe(
            """
            <img src="/midia/%s" height="200" alt="Linhas"/>"""
            % (image)
        )

    @admin.display(description="Bounding Boxes")
    def bbox(self, obj):
        from django.utils.html import mark_safe

        try:
            image = obj.images.first().bbox
        except Exception:
            image = None

        return mark_safe(
            """
            <img src="/midia/%s" height="200" alt="Bounding Boxes"/>"""
            % (image)
        )
