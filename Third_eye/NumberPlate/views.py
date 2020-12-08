from django.shortcuts import render
from django.http import HttpResponse
import cv2, numpy as np
import time
# Create your views here.

def nPlateDetection(request):
    # return HttpResponse("Number PlateDetection In Progess!!")
    plate_cascade= cv2.CascadeClassifier('model/cascades/data/haarcascade_russian_plate_number.xml')
    print("Number PlateDetection In Progess!!")

    cap = cv2.VideoCapture("model/video7.avi")
    # cap = cv2.VideoCapture(0)
    y_cord_mx=80
    y_cord_mn=20
    count = 0
    if cap.isOpened():
        print("Number PlateDetection In Progess!!")
        while True:
            check, frame = cap.read()
            if check:
                gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('color Frame', frame)
                plates= plate_cascade.detectMultiScale(gray_frame, scaleFactor= 1.5, minNeighbors=5)
            
                
                for (x,y, w, h) in plates:
                    
                    region_of_interest_gray= gray_frame[y: y+h, x: x+w]
                    rect_color=(255, 0, 0)
                    stroke=2
                    cv2.rectangle(frame, (x, y), ( x+w , y+h ), rect_color, stroke )   #param: frame, co-ordinates, Height- width , color of rect. , Stroke
                    cv2.imshow("Tkinter and OpenCV", frame)
                    print(y)
                    if ((y_cord_mn<=y) and (y<=y_cord_mx)):
                        count=count+1
                        print(x, y, w, h)
                        region_image= 'platesOutput/my_image'+str(count)+'.png'
                        cv2.imshow("plates", region_of_interest_gray)
                        cv2.imwrite(region_image, region_of_interest_gray)               
                    
                key = cv2.waitKey(50)

                if key == ord('q'):
                    break
            else:
                print('Frame not available')
                print(cap.isOpened())

    cap.release()
    cv2.destroyAllWindows()
    return HttpResponse("Number PlateDetection In Progess!!")


