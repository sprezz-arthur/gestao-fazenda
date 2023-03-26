# Generated by Django 4.1.4 on 2023-03-26 12:32

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("fazenda", "0021_ordenhadetectada_remove_ordenha_nome_auto_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="ordenhadetectada",
            name="ordenha",
            field=models.OneToOneField(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="fazenda.ordenha",
            ),
        ),
    ]
