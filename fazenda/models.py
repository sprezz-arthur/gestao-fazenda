from django.utils import timezone
from django.utils.safestring import mark_safe
from django.contrib.gis.db import models

from image_labelling_tool import models as lt_models

from .choices import PREFIXO_CHOICES, PERIODO_CHOICES


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


class Ordenha(models.Model):
    periodo = models.CharField(
        max_length=1, choices=PERIODO_CHOICES, null=False, blank=False
    )
    peso = models.FloatField(null=False, blank=False)
    vaca = models.ForeignKey(Vaca, null=False, blank=False, on_delete=models.CASCADE)
    data = models.DateField(null=True, blank=True)
    ficha = models.ForeignKey(
        FichaOrdenha, null=True, blank=True, on_delete=models.SET_NULL
    )


class FotoOrdenha(models.Model):
    ficha = models.OneToOneField(
        FichaOrdenha,
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
    )
    labels = models.ForeignKey(
        lt_models.Labels, models.CASCADE, related_name="original", blank=True, null=True
    )
    original = models.ImageField()
    lines = models.ImageField(null=True, blank=True)
    dewarped = models.ImageField(null=True, blank=True)
    bbox = models.ImageField(null=True, blank=True)

    class Meta:
        verbose_name = "Foto de Ordenhas"
        verbose_name_plural = "Fotos de Ordenhas"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        if not self.dewarped:
            from utils.geometry import get_dewarped

            full_path = self.original.path
            image = get_dewarped(full_path)
            self.dewarped.save(f"dewarped-{self.original.name}", image)

        if not self.bbox:
            from utils.geometry import get_bbox

            full_path = self.dewarped.path
            image = get_bbox(full_path)
            self.bbox.save(f"bbox-{self.original.name}", image)
