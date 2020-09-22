from django.db import models
class Register(models.Model):
    images=models.FileField(upload_to="files",blank=False,help_text="Browse the file here and insert only csv files only")