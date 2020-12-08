from django.conf import settings
from imageai.Detection import ObjectDetection
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import *
import PIL

class Model:
    def __init__(self):
        self.execution_path = os.getcwd()
        print(self.execution_path)
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath( os.path.join(self.execution_path , "yolo.h5"))
        self.detector.loadModel()
        print("Model Loaded SuccessFully!")
        # detector=settings.gModel



class Detection(Model):                      #inherite the Model
    def __init__(self):
        Model.__init__(self)
        
    def provide(self, inImage, outImage):
        self.inImage=inImage
        self.outImage=outImage
        custom_objects = self.detector.CustomObjects(person=True, motorcycle=True)
        self.returned_image, self.detections, self.extracted_objects = self.detector.detectCustomObjectsFromImage(output_type="array", extract_detected_objects=True,input_image=os.path.join(self.execution_path, self.inImage), output_image_path=os.path.join(self.execution_path , self.outImage), custom_objects=custom_objects, minimum_percentage_probability=65)
        #, output_type="array", extract_detected_objects=True
        print("done, Continue!!")
    
    def displayDetects(self):
        # print the output image first
        plt.imshow(self.returned_image)
        plt.show()
        print("--------------------------------")
        #  second is the classes which are detected
        # print(detections)
        for eachObject in self.detections:
           print(eachObject['name'] ,":", eachObject['percentage_probability'],":", eachObject['box_points'])
           print("--------------------------------")
        # third extracted images from the classes  
        for eachExtract in self.extracted_objects:
           plt.imshow(eachExtract)
           plt.show()
        #    print(eachExtract['box_points'])
           print("--------------------------------")
    
    def detectPeopleOnBike(self):
        self.index=-1
        len_of_detects=len(self.detections)       #length of the detected aarray
        
        for indx in range(len_of_detects):
            if 'motorcycle' in self.detections[indx]['name']:
                self.index=indx
                print("index of the moterbike in detections array is :",self.index)
        # capturing the moterbike in data with index to check with the peoples
        
        self.resultArray=[]                #Peoples on the bike 

        for i in range(len_of_detects):
            if 'motorcycle' != self.detections[self.index]['name']:
                pass
            else:
                if self.detections[i]['box_points'][0] >= self.detections[self.index]['box_points'][0] & self.detections[i]['box_points'][2]<=self.detections[self.index]['box_points'][2]:
                    if (self.detections[i]['box_points'][0] - self.detections[self.index]['box_points'][0]) > 75 or (self.detections[i]['box_points'][3] - self.detections[self.index]['box_points'][3])<10:
                        if (self.detections[self.index]['box_points'][3]-self.detections[i]['box_points'][3])>40:
                            if ((self.detections[self.index]!=self.detections[i])):
        # not considering vehicle
                                self.resultArray.append(self.detections[i]['box_points'])
                                print(self.detections[i])
                                plt.imshow(self.extracted_objects[i])
                                plt.show()
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
        print("Result Array is ",self.resultArray)
        self.coordinates=list()
        if len(self.resultArray)>=2:
            print("Tripsy")
        #     print(resultArray)
            self.resultArray.sort()

        # sorting the array to get the min and max points in the images
            items=[self.resultArray[0][0],self.resultArray[0][1],self.resultArray[len(self.resultArray)-1][2],self.resultArray[len(self.resultArray)-1][3]]
            for item in items:
                self.coordinates.append(item)
        elif len(self.resultArray)==1:
            items=[self.resultArray[0][0],self.resultArray[0][1],self.resultArray[0][2],self.resultArray[0][3]]
            for item in items:
                self.coordinates.append(item)
              #Single
        else:
            items=[0,0,0,0]
            for item in items:
                self.coordinates.append(item)
        print("done")
        print("coordinates of rectangle",self.coordinates)
        # coordinates conntain the coordinates of the rectangle which will show the    *tripsy*    is here

    def displayTripsy(self):
        I = np.asarray(PIL.Image.open(self.inImage, mode='r').convert('RGB'))
        #Open as a array and perform the operation
               
        if len(self.resultArray)>=2:
                print("Tripsy Found")
                # print(I)
                img = cv2.rectangle(I,(self.coordinates[0],self.coordinates[1]),(self.coordinates[2],self.coordinates[3]), (0,0,255),4)
                cv2.imshow("Image",img)
                outImg = Image.fromarray(img)
                outImg.save(self.outImage)

                cv2.waitKey(0)
        else:
                img = cv2.rectangle(I,(self.coordinates[0],self.coordinates[1]),(self.coordinates[2],self.coordinates[3]), (0,255,0),4)
                cv2.imshow("Image",img)
                outImg = Image.fromarray(img)
                outImg.save(self.outImage)
                print("No Tripsy Found")
                cv2.waitKey(0)

detect=Detection()
detect.provide("../../media/trainimg_110.jpg","../../media/outtrainimg_110.jpg")
detect.displayDetects()
detect.detectPeopleOnBike()
detect.displayTripsy()
