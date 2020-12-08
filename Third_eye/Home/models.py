from django.db import models

# Create your models here.

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=50)
    phone = models.IntegerField(10)
    typ = models.CharField(max_length=5)
    img = models.ImageField(blank=True)
    vid =models.FileField(blank=True)
    outimg=models.ImageField(blank=True)
    outvid =models.FileField(blank=True)
    spec= models.CharField(max_length=50, blank=True)