from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

from django.db import models
from django.utils import timezone
from django.utils.safestring import mark_safe
from image_labelling_tool import models as lt_models

from utils.dataframe import fix_peso, get_closest_vaca_tuple, get_table, process_table

if TYPE_CHECKING:
    from django.db.models import Manager


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
    numero = models.CharField(max_length=255, null=True, blank=True)
    nome = models.CharField(max_length=255, null=False, blank=False)
    prefixo = models.CharField(max_length=31, null=True, blank=True)
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

    def __str__(self) -> str:
        return " ".join(
            filter(lambda s: bool(s), [self.numero, self.prefixo, self.nome])
        )

    @property
    def image_tag(self):
        try:
            return mark_safe(f'<img src="{self.foto.url}" width="75" height="75"/>')
        except Exception:
            return ""


class Ordenha(models.Model):
    foto = models.ForeignKey(
        "FotoOrdenha", null=True, blank=True, on_delete=models.SET_NULL
    )
    numero = models.CharField(max_length=255, null=True, blank=True)
    nome = models.CharField(
        max_length=255,
        null=True,
        blank=True,
    )
    peso_manha = models.DecimalField(
        null=True, blank=True, decimal_places=3, max_digits=12
    )
    peso_tarde = models.DecimalField(
        null=True, blank=True, decimal_places=3, max_digits=12
    )

    vaca = models.ForeignKey(Vaca, null=True, blank=True, on_delete=models.SET_NULL)

    data = models.DateField(null=True, blank=True)

    prefixo = models.CharField(max_length=31, null=True, blank=True)


class OrdenhaDetectada(models.Model):
    foto = models.ForeignKey(
        "FotoOrdenha", null=True, blank=True, on_delete=models.SET_NULL
    )

    ordenha = models.OneToOneField(
        Ordenha,
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
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
    ordenha_set: Manager[Ordenha]
    ordenhadetectada_set: Manager[OrdenhaDetectada]

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
    peso_balde = models.DecimalField(default=0.0, decimal_places=3, max_digits=12)

    def __str__(self) -> str:
        return self.original.name

    class Meta:
        verbose_name = "Foto de Ordenhas"
        verbose_name_plural = "Fotos de Ordenhas"

    def set_dewarped(self, poly=None):
        if self.dewarped and poly is None:
            return
        from utils.geometry import get_dewarped_poly_with_lines

        full_path = self.original.path
        image = get_dewarped_poly_with_lines(full_path, poly=poly)

        self.dewarped.save(f"dewarped-{self.original.name}", image)
        self.bbox.delete()

    def set_bbox(self):
        from utils.geometry import get_bbox

        if self.bbox or not self.dewarped:
            return
        image, bounds = get_bbox(self.dewarped.path)
        self.bounds = bounds
        self.bbox.save(f"bbox-{self.original.name}", image)

        self.save()

    def save(self, *args, **kwargs):
        try:
            self.labels
        except Exception:
            self.labels = Labels.objects.create(creation_date=timezone.now())

        super().save(*args, **kwargs)

    def get_ordenha(self):
        table = get_table(self.dewarped.path)

        self.ordenha_set.all().delete()
        self.ordenhadetectada_set.all().delete()

        vacas = Vaca.objects.values_list("numero", "nome")

        for num, nome, p1, p2 in process_table(table):
            vaca_tuple = get_closest_vaca_tuple((num, nome), vacas)
            auto_num, auto_nome = vaca_tuple
            auto_p1 = fix_peso(p1)
            auto_p2 = fix_peso(p2)

            auto_num = auto_num or None
            auto_nome = auto_nome or None

            ordenhadetecatada, _ = OrdenhaDetectada.objects.get_or_create(
                foto=self,
                numero=num,
                nome=nome,
                peso_manha=p1,
                peso_tarde=p2,
            )

            vaca = Vaca.objects.filter(numero=auto_num, nome=auto_nome).first()

            Ordenha.objects.get_or_create(
                foto=self,
                ordenhadetectada=ordenhadetecatada,
                vaca=vaca,
                prefixo=vaca.prefixo,
                numero=vaca.numero,
                nome=vaca.nome,
                peso_manha=auto_p1,
                peso_tarde=auto_p2,
            )


class Rebanho(models.Model):
    file = models.FileField()

    def __str__(self) -> str:
        return self.file.name

    def save(self, *args, **kwargs):
        import pandas as pd

        from utils.importar_vacas import importar_vacas

        Vaca.objects.all().delete()

        def convert_xlsx_to_csv_in_memory(xlsx_bytes):
            df = pd.read_excel(io.BytesIO(xlsx_bytes))
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            return csv_content

        if self.file.path.endswith(".xlsx"):
            csv_file = convert_xlsx_to_csv_in_memory(self.file.read())
            importar_vacas(csv_file)
        else:  # csv
            importar_vacas(self.file.read())

        super().save(*args, **kwargs)
