# Generated by Django 4.1.4 on 2023-01-26 09:22

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("fazenda", "0003_fichaordenha_data_alter_fichaordenha_foto_original"),
    ]

    operations = [
        migrations.RenameField(
            model_name="fotoordenha",
            old_name="fotos",
            new_name="foto",
        ),
    ]
