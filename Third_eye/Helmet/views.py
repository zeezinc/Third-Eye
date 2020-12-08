from django.shortcuts import render
from django.http import HttpResponse
import requests

# Create your views here.

def helmetDetection(request):
    return HttpResponse("Helmet Detection In Progess!!")

def location(request):
    # response=requests.get("http://localhost:5000/")
    # geodata = response.json()
    # print(geodata)
    # return HttpResponse(geodata)
    r = requests.get('http://127.0.0.1:5000/helmet/v.mp4', params=request.GET)
    if r.status_code == 200:
        return HttpResponse('Yay, it worked')
    else :
        return HttpResponse('Server is Offline')