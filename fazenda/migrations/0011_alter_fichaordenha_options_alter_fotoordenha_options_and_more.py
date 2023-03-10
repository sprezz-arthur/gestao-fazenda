# Generated by Django 4.1.4 on 2023-03-11 22:55

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("fazenda", "0010_rename_original_fotoordenha_image"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="fichaordenha",
            options={
                "verbose_name": "Ficha de Ordenhas",
                "verbose_name_plural": "Fichas de Ordenhas",
            },
        ),
        migrations.AlterModelOptions(
            name="fotoordenha",
            options={
                "verbose_name": "Foto de Ordenhas",
                "verbose_name_plural": "Fotos de Ordenhas",
            },
        ),
        migrations.RemoveField(
            model_name="fotoordenha",
            name="ordenha",
        ),
        migrations.AlterField(
            model_name="fotoordenha",
            name="ficha",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="fotos",
                to="fazenda.fichaordenha",
            ),
        ),
        migrations.AlterField(
            model_name="vaca",
            name="prefixo",
            field=models.CharField(
                blank=True,
                choices=[
                    ("TE", "Transferência Embrionária"),
                    ("AM", "Amojada"),
                    ("PRI", "Primeira Alguma Coisa"),
                    ("DESC", "Descarte"),
                    ("LI", "Lítio"),
                ],
                max_length=31,
                null=True,
            ),
        ),
    ]
