# Generated by Django 3.0.4 on 2020-04-07 16:16

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0005_auto_20200407_2139'),
    ]

    operations = [
        migrations.AlterField(
            model_name='register',
            name='images',
            field=models.FileField(blank=True, help_text='Browse the file here It  accepts only csv files only', upload_to='files', validators=[django.core.validators.FileExtensionValidator(['CSV'])]),
        ),
    ]
