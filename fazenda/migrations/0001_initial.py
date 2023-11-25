# Generated by Django 4.2.7 on 2023-11-25 10:42

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('image_labelling_tool', '0009_alter_labelclass_id_alter_labelclasscolour_id_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Fazenda',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nome', models.CharField(max_length=255)),
                ('cidade', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='FotoOrdenha',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('original', models.ImageField(upload_to='')),
                ('dewarped', models.ImageField(blank=True, null=True, upload_to='')),
                ('bbox', models.ImageField(blank=True, null=True, upload_to='')),
                ('bounds', models.TextField(blank=True, null=True)),
            ],
            options={
                'verbose_name': 'Foto de Ordenhas',
                'verbose_name_plural': 'Fotos de Ordenhas',
            },
        ),
        migrations.CreateModel(
            name='Ordenha',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('numero', models.IntegerField(blank=True, null=True)),
                ('nome', models.CharField(blank=True, max_length=255, null=True)),
                ('peso_manha', models.FloatField(blank=True, null=True)),
                ('peso_tarde', models.FloatField(blank=True, null=True)),
                ('data', models.DateField(blank=True, null=True)),
                ('foto', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='fazenda.fotoordenha')),
            ],
        ),
        migrations.CreateModel(
            name='Labels',
            fields=[
            ],
            options={
                'verbose_name': 'Label Set',
                'verbose_name_plural': 'Label Sets',
                'proxy': True,
                'indexes': [],
                'constraints': [],
            },
            bases=('image_labelling_tool.labels',),
        ),
        migrations.CreateModel(
            name='Vaca',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('numero', models.IntegerField()),
                ('nome', models.CharField(max_length=255)),
                ('prefixo', models.CharField(blank=True, choices=[('TE', 'Transferência Embrionária'), ('AM', 'Amojada'), ('PRI', 'Primeira Alguma Coisa'), ('DESC', 'Descarte'), ('LI', 'Lítio')], max_length=31, null=True)),
                ('nome_ideagri', models.CharField(blank=True, max_length=255, null=True)),
                ('foto', models.ImageField(blank=True, null=True, upload_to='')),
                ('fazenda', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='fazenda.fazenda')),
            ],
            options={
                'unique_together': {('numero', 'nome')},
            },
        ),
        migrations.CreateModel(
            name='OrdenhaDetectada',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('numero', models.CharField(blank=True, max_length=255, null=True)),
                ('nome', models.CharField(blank=True, max_length=255, null=True)),
                ('peso_manha', models.CharField(blank=True, max_length=255, null=True)),
                ('peso_tarde', models.CharField(blank=True, max_length=255, null=True)),
                ('foto', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='fazenda.fotoordenha')),
                ('ordenha', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='fazenda.ordenha')),
            ],
        ),
        migrations.AddField(
            model_name='ordenha',
            name='vaca',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='fazenda.vaca'),
        ),
        migrations.AddField(
            model_name='fotoordenha',
            name='labels',
            field=models.OneToOneField(blank=True, on_delete=django.db.models.deletion.CASCADE, related_name='fotoordenha', to='fazenda.labels'),
        ),
    ]
