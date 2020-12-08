from django.urls import path
from . import views

urlpatterns=[
    path('/', views.nPlateDetection, name="nPlateDetection"),
    
]