# Generated by Django 4.1.4 on 2023-03-24 07:33

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("fazenda", "0006_labels_alter_fotoordenha_labels"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="fotoordenha",
            name="lines",
        ),
        migrations.AlterField(
            model_name="fotoordenha",
            name="labels",
            field=models.OneToOneField(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="fotoordenha",
                to="fazenda.labels",
            ),
        ),
    ]