# Generated by Django 4.2.7 on 2024-08-20 10:33

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fazenda", "0008_ordenha_prefixo"),
    ]

    operations = [
        migrations.AddField(
            model_name="ordenhadetectada",
            name="prefixo",
            field=models.CharField(blank=True, max_length=31, null=True),
        ),
    ]
