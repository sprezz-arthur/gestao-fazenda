# Generated by Django 4.2.7 on 2023-12-12 12:33

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fazenda", "0002_fotoordenha_peso_balde"),
    ]

    operations = [
        migrations.AlterField(
            model_name="ordenha",
            name="numero",
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name="vaca",
            name="numero",
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
