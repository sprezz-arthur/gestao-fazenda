# Generated by Django 4.1.4 on 2023-01-26 09:11

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("fazenda", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="fichaordenha",
            name="foto_original",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="fazenda.fotoordenha",
            ),
        ),
    ]
