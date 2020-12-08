from django.urls import path
from . import views

urlpatterns=[
    path('', views.Home, name="Home"),
    path('Download', views.Download, name="Download"),
    path('Contact', views.Contact, name="Contact"),
    path('About', views.About, name="About"),
    path('Responce', views.Responce, name="Responce"),
    path('DisplayId', views.DisplayId, name="DisplayId"),

]