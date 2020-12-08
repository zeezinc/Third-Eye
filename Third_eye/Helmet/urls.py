from django.urls import path
from . import views

urlpatterns=[
    path('/', views.helmetDetection, name="helmetDetection"),
    path('/loc',views.location, name="location")
    
]