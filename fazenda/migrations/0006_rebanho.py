# Generated by Django 4.2.7 on 2024-02-10 20:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fazenda', '0005_ordenha_prefixo'),
    ]

    operations = [
        migrations.CreateModel(
            name='Rebanho',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='')),
            ],
        ),
    ]
