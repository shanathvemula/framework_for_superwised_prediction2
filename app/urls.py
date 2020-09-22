from django.urls import path
from app import views

urlpatterns = [
    path('',views.home),
    path('upload/',views.upload),
    path("Loading/",views.Loading),
    path("result/",views.result),
]
