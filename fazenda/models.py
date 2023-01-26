from django.db import models

from django.core.files.storage import FileSystemStorage

from django.utils import timezone


class Fazenda(models.Model):
    nome = models.CharField(max_length=255, null=False, blank=False)
    cidade = models.CharField(max_length=255, null=False, blank=False)


class Vaca(models.Model):

    PREFIXO_CHOICES = [
        ("TE", "Transferência Embrionária"),
        (
            "AM",
            "Amojada",
        ),
        (
            "PRI",
            "Primeira Alguma Coisa",
        ),
        (
            "DESC",
            "Descarte",
        ),
        (
            "LI",
            "Lítio",
        ),
    ]

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


class FichaOrdenha(models.Model):
    csv = models.FileField(null=True, blank=True)
    data = models.DateField(default=timezone.now, blank=True, null=True)

    class Meta:
        verbose_name = "Ficha de Ordenhas"
        verbose_name_plural = "Fichas de Ordenhas"


class Ordenha(models.Model):
    MANHA = "M"
    TARDE = "T"

    PERIODO_CHOICES = [(MANHA, "Manhã"), (TARDE, "Tarde")]

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
    ficha = models.ForeignKey(
        FichaOrdenha,
        related_name="images",
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
    )
    ordenha = models.ForeignKey(
        Ordenha, blank=True, null=True, on_delete=models.SET_NULL
    )
    image = models.ImageField()
    linhas = models.ImageField(null=True, blank=True)
    bbox = models.ImageField(null=True, blank=True)

    class Meta:
        verbose_name = "Foto de Ordenhas"
        verbose_name_plural = "Fotos de Ordenhas"
