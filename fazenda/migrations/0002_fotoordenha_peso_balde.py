# Generated by Django 4.2.7 on 2023-11-26 21:19

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fazenda", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="fotoordenha",
            name="peso_balde",
            field=models.FloatField(default=0),
        ),
    ]
