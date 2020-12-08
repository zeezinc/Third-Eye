from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib import messages
from .models import User
import time
import requests
import os
# Create your views here.



def Home(request):
    return render(request, 'Home.html')

def Download(request):
    if request.method=='GET':
        return render(request, 'Download.html')

def Contact(request):
    return render(request, 'Contact.html')

def About(request):
    return render(request, 'About.html')

def Responce(request):
    
    if request.method=='POST':

        first_name=request.POST['fname']
        last_name=request.POST['lname']
        email=request.POST['email']
        phone=request.POST['phone']
        typ=request.POST['type']
        # print(type(request.POST['myFile']))
        myFile=request.FILES.get('myFile',False)
        specification=request.POST['specification']
        #print("specification",request.POST['specification'])

        # check if all blank fields
        if (first_name is None or last_name is None or email is None or phone is None or myFile is False) :  
            messages.info(request,'Please, Fill All The Information!')
            return redirect('/')
        else:
            # validation on email
            if User.objects.filter(email=email).exists():
                messages.info(request,'Email is already taken!')
                return redirect('/')
            else:
                
                # create user object and pass the parameters
                if typ=="Image":
                    user = User(name = first_name, email=email, typ= typ, img=myFile,phone=phone, spec=specification)
                    user.save();
                else:
                    user = User(name = first_name, email=email, typ= typ, vid=myFile,phone=phone, spec=specification)
                    user.save();
                print('user created!')
                # print(type(myFile))

                # r=requests.post("http://127.0.0.1:5000/file=",myFile)
                # print(r.text)


                getuser= User.objects.get(email=email)
                print(getuser.id)
                return render(request, 'DisplayId.html',{ 'id':getuser.id, 'user':user, 'File': myFile})
                # return rediect(request,'Display.html',{'user': user})
            
    elif request.method=='GET':
        id=int(request.GET['id'])
        user=User.objects.get(id=id)
        
        path = "G:\MCS part II Project Files\Third Eye Web App\Third_eye\media\Out\Plates"
        img_list = os.listdir(path)
        # print(img_list)
        return render(request, 'Download.html', {'user': user, 'imgs': img_list})

def DisplayId(request):
    return render(request, 'DisplayId.html')
