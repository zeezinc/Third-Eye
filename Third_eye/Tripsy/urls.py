from django.urls import path
from . import views

urlpatterns=[
    path('/', views.tripsyDetection, name="tripsyDetection"),
    
]