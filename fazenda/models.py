from django.utils import timezone
from django.utils.safestring import mark_safe
from django.contrib.gis.db import models

from image_labelling_tool import models as lt_models

from .choices import PREFIXO_CHOICES, PERIODO_CHOICES

import json

from utils.dataframe import *


class Labels(lt_models.Labels):
    def save(self, *args, **kwargs):
        poly = json.loads(self.labels_json_str)
        poly = poly[0] if poly else {}

        if poly:
            self.fotoordenha.set_dewarped(poly)
            self.fotoordenha.set_bbox()

        return super().save(*args, **kwargs)

    class Meta:
        proxy = True
        verbose_name = "Label Set"
        verbose_name_plural = "Label Sets"


class Fazenda(models.Model):
    nome = models.CharField(max_length=255, null=False, blank=False)
    cidade = models.CharField(max_length=255, null=False, blank=False)


class Vaca(models.Model):
    numero = models.IntegerField(null=False, blank=False)
    nome = models.CharField(max_length=255, null=False, blank=False)
    prefixo = models.CharField(
        max_length=31, choices=PREFIXO_CHOICES, null=True, blank=True
    )
    nome_ideagri = models.CharField(max_length=255, null=True, blank=True)
    foto = models.ImageField(null=True, blank=True)
    fazenda = models.ForeignKey(
        Fazenda, null=True, blank=True, on_delete=models.SET_NULL
    )

    class Meta:
        unique_together = (
            "numero",
            "nome",
        )

    @property
    def image_tag(self):
        try:
            return mark_safe(f'<img src="{self.foto.url}" width="75" height="75"/>')
        except Exception:
            return ""


class FichaOrdenha(models.Model):
    csv = models.FileField(null=True, blank=True)
    data = models.DateField(default=timezone.now, blank=True, null=True)

    class Meta:
        verbose_name = "Ficha de Ordenhas"
        verbose_name_plural = "Fichas de Ordenhas"

    def get_ordenhas(self, *args, **kwargs):
        if self.fotoordenha.dewarped:
            Ordenha.objects.create(ficha=self, peso=42)


class Ordenha(models.Model):
    numero = models.IntegerField(null=True, blank=True)
    nome = models.CharField(
        max_length=255,
        null=True,
        blank=True,
    )
    peso_manha = models.FloatField(null=True, blank=True)
    peso_tarde = models.FloatField(null=True, blank=True)

    vaca = models.ForeignKey(Vaca, null=True, blank=True, on_delete=models.SET_NULL)

    data = models.DateField(null=True, blank=True)
    ficha = models.ForeignKey(
        FichaOrdenha, null=True, blank=True, on_delete=models.SET_NULL
    )


class OrdenhaDetectada(models.Model):
    ficha = models.ForeignKey(
        FichaOrdenha, null=True, blank=True, on_delete=models.SET_NULL
    )

    ordenha = models.OneToOneField(
        Ordenha, blank=True, null=True, on_delete=models.SET_NULL
    )
    numero = models.CharField(max_length=255, null=True, blank=True)
    nome = models.CharField(
        max_length=255,
        null=True,
        blank=True,
    )
    peso_manha = models.CharField(max_length=255, null=True, blank=True)
    peso_tarde = models.CharField(max_length=255, null=True, blank=True)


class FotoOrdenha(models.Model):
    ficha = models.OneToOneField(
        FichaOrdenha,
        related_name="fotoordenha",
        blank=False,
        null=False,
        on_delete=models.CASCADE,
    )
    labels = models.OneToOneField(
        Labels,
        related_name="fotoordenha",
        blank=True,
        null=False,
        on_delete=models.CASCADE,
    )
    original = models.ImageField()
    dewarped = models.ImageField(null=True, blank=True)

    bbox = models.ImageField(null=True, blank=True)
    bounds = models.TextField(null=True, blank=True)

    class Meta:
        verbose_name = "Foto de Ordenhas"
        verbose_name_plural = "Fotos de Ordenhas"

    def set_dewarped(self, poly=None):
        if self.dewarped and poly is None:
            return
        from utils.geometry import get_dewarped, get_contour

        full_path = self.original.path
        image = get_dewarped(full_path, poly=poly)
        self.dewarped.save(f"dewarped-{self.original.name}", image)
        self.bbox.delete()

    def set_bbox(self):
        from utils.geometry import get_bbox, bounds_to_dict

        if self.bbox or not self.dewarped:
            return
        image, bounds = get_bbox(self.dewarped.path)
        self.bounds = bounds
        self.bbox.save(f"bbox-{self.original.name}", image)

        self.save()

    def save(self, *args, **kwargs):
        try:
            self.ficha
        except Exception:
            self.ficha = FichaOrdenha.objects.create(data=timezone.now())

        try:
            self.labels
        except Exception:
            self.labels = Labels.objects.create(creation_date=timezone.now())

        super().save(*args, **kwargs)

    def get_ordenha_detectada(self):

        table, vacas = get_table(self.dewarped.path)

        for row in table:

            (num, nome, p1, p2) = process_row(row)

            
            print(num, nome, p1, p2)

            ordenha = Ordenha.objects.create(ficha=self.ficha)

            OrdenhaDetectada.objects.create(
                ficha=self.ficha,
                ordenha=ordenha,
                numero=num,
                nome=nome,
                peso_manha=p1,
                peso_tarde=p2,
            )

    def get_ordenha(self):

        table, vacas = get_table(self.dewarped.path)

        for row in table:

            (num, nome, p1, p2) = process_row(row)

            auto_num, auto_nome = closest_vaca((num, nome), vacas)

            ordenha = Ordenha.objects.create(ficha=self.ficha)

            OrdenhaDetectada.objects.create(
                ficha=self.ficha,
                ordenha=ordenha,
                numero=num,
                nome=nome,
                peso_manha=p1,
                peso_tarde=p2,
            )