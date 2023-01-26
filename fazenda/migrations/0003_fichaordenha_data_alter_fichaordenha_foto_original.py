# Generated by Django 4.1.4 on 2023-01-26 09:20

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("fazenda", "0002_alter_fichaordenha_foto_original"),
    ]

    operations = [
        migrations.AddField(
            model_name="fichaordenha",
            name="data",
            field=models.DateField(auto_now=True),
        ),
        migrations.AlterField(
            model_name="fichaordenha",
            name="foto_original",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="fazenda.fotoordenha",
            ),
        ),
    ]
