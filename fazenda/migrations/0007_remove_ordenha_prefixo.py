# Generated by Django 4.2.7 on 2024-08-20 10:08

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("fazenda", "0006_rebanho"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="ordenha",
            name="prefixo",
        ),
    ]
