from django.contrib import admin
from app.models import Register

class RegisterAdmin(admin.ModelAdmin):
    list_display = ['images']
admin.site.register(Register,RegisterAdmin)

# Register your models here.
