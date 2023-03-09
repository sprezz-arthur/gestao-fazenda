from django.contrib import admin
from fazenda import models


@admin.register(models.Fazenda)
class FazendaAdmin(admin.ModelAdmin):
    pass


@admin.register(models.Vaca)
class VacaAdmin(admin.ModelAdmin):
    list_display = ["zero_numero", "prefixo", "nome", "image_tag"]

    @admin.display(description="Número")
    def zero_numero(self, obj):
        return f"{obj.numero:03d}"


class ImageInline(admin.StackedInline):
    extra = 0
    model = models.FotoOrdenha


@admin.register(models.FotoOrdenha)
class FotoOrdenhaAdmin(admin.ModelAdmin):
    pass


@admin.register(models.FichaOrdenha)
class FichaOrdenhaAdmin(admin.ModelAdmin):
    inlines = [ImageInline]
    readonly_fields = ["image", "linhas", "bbox"]
    list_display = ["data", "image", "linhas", "bbox"]

    @admin.display(description="Foto")
    def image(self, obj):
        try:
            return obj.fotos.first().image.url
        except models.FotoOrdenha.DoesNotExist:
            return ""

    @admin.display(description="Linhas")
    def linhas(self, obj):
        try:
            oxe = obj.fotos.first()
            diro = dir(oxe)
            linhas = obj.fotos.first().linhas
            return obj.fotos.first().linhas.url
        except (models.FotoOrdenha.DoesNotExist, ValueError):
            return ""

    @admin.display(description="Bounding Boxes")
    def bbox(self, obj):
        try:
            return obj.fotos.first().bbox.url
        except (models.FotoOrdenha.DoesNotExist, ValueError):
            return ""
