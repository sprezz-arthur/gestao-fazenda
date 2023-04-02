from django.utils.safestring import mark_safe
from django.contrib import admin
from django.utils.safestring import mark_safe
from django.db.models import ImageField, IntegerField, FloatField
from django.forms import TextInput

from django.forms import TextInput

from . import widgets
from . import models
from . import forms


import csv
from django.http import HttpResponse


def exportar_ordenha_pra_csv(modeladmin, request, queryset):
    # Define the CSV filename

    filename = "{}.csv".format(queryset.model._meta.verbose_name_plural.lower())

    # Set the response headers
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="{}"'.format(filename)

    # Define the CSV writer
    writer = csv.writer(response, delimiter=";")

    # Write the header row
    header_row = "Nome;Nome;Ord. 1;Ord. 2;Ord. 3;Tot.;Data;Responsável;DEL;Dias sec. prev.;Grupo no controle;Observação;".split(
        ";"
    )
    writer.writerow(header_row)

    # Write the data rows
    for obj in queryset:
        row = []
        row.append(f"{obj.numero:03d} {obj.nome}")
        row.append(f"{obj.numero:03d} {obj.nome}")

        row.append(str(obj.peso_manha).replace(".", ","))
        row.append(str(obj.peso_tarde).replace(".", ","))

        for i in range(len(header_row) - 4):
            row.append("")

        writer.writerow(row)

    return response


exportar_ordenha_pra_csv.short_description = "Exportar Ordenha pra CSV"


@admin.register(models.Fazenda)
class FazendaAdmin(admin.ModelAdmin):
    pass


@admin.register(models.Ordenha)
class OrdenhaAdmin(admin.ModelAdmin):
    list_display = ["pk", "numero", "nome", "peso_manha", "peso_tarde"]
    list_editable = ["numero", "nome", "peso_manha", "peso_tarde"]

    change_list_template = "admin/change_list_compact.html"
    formfield_overrides = {
        FloatField: {
            "widget": TextInput(attrs={"size": "4", "style": "width: 100px;"})
        },
        IntegerField: {
            "widget": TextInput(attrs={"size": "4", "style": "width: 100px;"})
        },
    }

    actions = [exportar_ordenha_pra_csv]

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.order_by("pk")

    def changelist_view(self, request, extra_context=None):
        try:
            query_params = request.GET
            foto = models.FotoOrdenha.objects.get(pk=query_params.get("foto_id"))
            extra_context = {"bbox_url": foto.bbox.url}
        except Exception as e:
            pass
        return super().changelist_view(request, extra_context=extra_context)


@admin.register(models.OrdenhaDetectada)
class OrdenhaDetectadaAdmin(admin.ModelAdmin):
    list_display = ["pk", "numero", "nome", "peso_manha", "peso_tarde"]
    readonly_fields = ["numero", "nome", "peso_manha", "peso_tarde"]

    change_list_template = "admin/change_list_compact.html"
    formfield_overrides = {
        FloatField: {
            "widget": TextInput(attrs={"size": "4", "style": "width: 100px;"})
        },
        IntegerField: {
            "widget": TextInput(attrs={"size": "4", "style": "width: 100px;"})
        },
    }

    actions = [exportar_ordenha_pra_csv]

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.order_by("pk")

    def changelist_view(self, request, extra_context=None):
        try:
            query_params = request.GET
            foto = models.FotoOrdenha.objects.get(pk=query_params.get("foto_id"))
            extra_context = {"bbox_url": foto.bbox.url}
        except Exception as e:
            pass
        return super().changelist_view(request, extra_context=extra_context)


@admin.register(models.Vaca)
class VacaAdmin(admin.ModelAdmin):
    list_display = ["zero_numero", "prefixo", "nome", "image_tag"]
    search_fields = ["nome", "numero"]
    ordering = ["numero"]
    list_filter = ["prefixo"]

    @admin.display(description="Número")
    def zero_numero(self, obj):
        return f"{obj.numero:03d}"


class ImageInline(admin.StackedInline):
    extra = 0
    model = models.FotoOrdenha

    exclude = ["labels", "bounds"]

    template = "admin/inline/stacked.html"

    formfield_overrides = {
        ImageField: {"widget": widgets.AdminImageWidget},
    }


class OrdenhaInline(admin.StackedInline):
    extra = 0
    model = models.Ordenha

    template = "admin/inline/stacked.html"


class OrdenhaDetectadaInline(admin.StackedInline):
    extra = 0
    model = models.OrdenhaDetectada

    readonly_fields = ["numero", "nome", "peso_manha", "peso_tarde"]

    template = "admin/inline/stacked.html"


@admin.register(models.FotoOrdenha)
class FotoOrdenhaAdmin(admin.ModelAdmin):
    form = forms.FotoOrdenhaForm

    list_display = [
        "__str__",
        "original_thumbnail",
        "dewarped_thumbnail",
        "bbox_thumbnail",
    ]

    formfield_overrides = {
        ImageField: {"widget": widgets.AdminImageWidget},
    }

    change_form_template = "admin/change_form.html"

    @admin.display(description="Original")
    def original_thumbnail(self, obj):
        try:
            return mark_safe(
                f'<a href="{obj.original.url}"><img src="{obj.original.url}" height="300"/></a>'
            )
        except Exception:
            return ""

    @admin.display(description="Dewarped")
    def dewarped_thumbnail(self, obj):
        try:
            return mark_safe(
                f'<a href="{obj.dewarped.url}"><img src="{obj.dewarped.url}" height="300"/></a>'
            )
        except Exception:
            return ""

    @admin.display(description="Contour")
    def dewarped_contour_thumbnail(self, obj):
        try:
            return mark_safe(
                f'<a href="{obj.dewarped_contour.url}"><img src="{obj.dewarped_contour.url}" height="300"/></a>'
            )
        except Exception:
            return ""

    @admin.display(description="Bounding Boxes")
    def bbox_thumbnail(self, obj):
        try:
            return mark_safe(
                f'<a href="{obj.bbox.url}"><img src="{obj.bbox.url}" height="300"/></a>'
            )
        except Exception:
            return ""

    @admin.display(description="Bounding Boxes Contour")
    def bbox_contour_thumbnail(self, obj):
        try:
            return mark_safe(
                f'<a href="{obj.bbox_contour.url}"><img src="{obj.bbox_contour.url}" height="300"/></a>'
            )
        except Exception:
            return ""


from image_labelling_tool import models as lt_models

admin.site.unregister(lt_models.LabellingTask)
admin.site.unregister(lt_models.LabellingSchema)
admin.site.unregister(lt_models.LabellingColourScheme)
admin.site.unregister(lt_models.LabelClassGroup)
admin.site.unregister(lt_models.LabelClass)
admin.site.unregister(lt_models.LabelClassColour)
admin.site.unregister(lt_models.Labels)


@admin.register(models.Labels)
class LabelsAdmin(admin.ModelAdmin):
    form = forms.LabelsForm
