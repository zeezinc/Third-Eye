from flask import Flask, flash, request, redirect, url_for
   # import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
from imageai.Detection import ObjectDetection
from imutils.video import VideoStream
import numpy as np
from imutils.video import FPS
import imutils
import time
import os
import cv2
from keras.models import load_model
import numpy as np
import time
from werkzeug.utils import secure_filename
from tripsyModel import imageTripsyDetection

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'])

# Vehicle Count : /vCount
# Helmet : /helmet
# Tripsy: /tripsyDetection
# NumberPlate: /numberPlate





app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "C:/Users/l\Desktop/RetinaNet/Flask Api/uploads/"
app.config['Third_Eye_Media'] = "G:\MCS part II Project Files\Third Eye Web App\Third_eye\media/"
app.config['Third_Eye_Plates_Out'] = "G:\MCS part II Project Files\Third Eye Web App\Third_eye\media/Out/Plates/"
app.config['Third_Eye_Out'] = "G:\MCS part II Project Files\Third Eye Web App\Third_eye\media/Out/"



@app.route("/vCount/<video>")
def vCount(video):
    videopath=os.path.join(app.config['Third_Eye_Media'],'%s'%video)
    cap = cv2.VideoCapture(videopath)
    frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = int(width)
    height = int(height)
    print(frames_count, fps, width, height)

    # creates a pandas data frame with the number of rows the same length as frame count
    df = pd.DataFrame(index=range(int(frames_count)))
    df.index.name = "Frames"

    framenumber = 0  # keeps track of current frame
    carscrossedup = 0  # keeps track of cars that crossed up
    carscrosseddown = 0  # keeps track of cars that crossed down
    carids = []  # blank list to add car ids
    caridscrossed = []  # blank list to add car ids that have crossed
    totalcars = 0  # keeps track of total cars

    fgbg = cv2.createBackgroundSubtractorMOG2()  # create background subtractor

    # information to start saving a video file
    ret, frame = cap.read()  # import image
    ratio = .5  # resize ratio
    image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
    width2, height2, channels = image.shape
    video = cv2.VideoWriter('vehicleCountModel/Out/traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)

    while True:

        ret, frame = cap.read()  # import image

        if ret:  # if there is a frame continue with code

            image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray

            fgmask = fgbg.apply(gray)  # uses the background subtraction

            # applies different thresholds to fgmask to try and isolate cars
            # just have to keep playing around with settings until cars are easily identifiable
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
            closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            dilation = cv2.dilate(opening, kernel)
            retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows

            # creates contours
            contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #im2, was here

            # use convex hull to create polygon around contours
            hull = [cv2.convexHull(c) for c in contours]

            # draw contours
            cv2.drawContours(image, hull, -1, (0, 255, 0), 3)

            # line created to stop counting contours, needed as cars in distance become one big contour
            lineypos = 225
            cv2.line(image, (0, lineypos), (width, lineypos), (255, 0, 0), 5)

            # line y position created to count contours
            lineypos2 = 250
            cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 255, 0), 5)

            # min area for contours in case a bunch of small noise contours are created
            minarea = 300

            # max area for contours, can be quite large for buses
            maxarea = 50000

            # vectors for the x and y locations of contour centroids in current frame
            cxx = np.zeros(len(contours))
            cyy = np.zeros(len(contours))

            for i in range(len(contours)):  # cycles through all contours in current frame

                if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)

                    area = cv2.contourArea(contours[i])  # area of contour

                    if minarea < area < maxarea:  # area threshold for contour

                        # calculating centroids of contours
                        cnt = contours[i]
                        M = cv2.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])

                        if cy > lineypos:  # filters out contours that are above line (y starts at top)

                            # gets bounding points of contour to create rectangle
                            # x,y is top left corner and w,h is width and height
                            x, y, w, h = cv2.boundingRect(cnt)

                            # creates a rectangle around contour
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Prints centroid text in order to double check later on
                            cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        .3, (0, 0, 255), 1)

                            cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                        line_type=cv2.LINE_AA)

                            # adds centroids that passed previous criteria to centroid list
                            cxx[i] = cx
                            cyy[i] = cy

            # eliminates zero entries (centroids that were not added)
            cxx = cxx[cxx != 0]
            cyy = cyy[cyy != 0]

            # empty list to later check which centroid indices were added to dataframe
            minx_index2 = []
            miny_index2 = []

            # maximum allowable radius for current frame centroid to be considered the same centroid from previous frame
            maxrad = 25

            # The section below keeps track of the centroids and assigns them to old carids or new carids

            if len(cxx):  # if there are centroids in the specified area

                if not carids:  # if carids is empty

                    for i in range(len(cxx)):  # loops through all centroids

                        carids.append(i)  # adds a car id to the empty list carids
                        df[str(carids[i])] = ""  # adds a column to the dataframe corresponding to a carid

                        # assigns the centroid values to the current frame (row) and carid (column)
                        df.at[int(framenumber), str(carids[i])] = [cxx[i], cyy[i]]

                        totalcars = carids[i] + 1  # adds one count to total cars

                else:  # if there are already car ids

                    dx = np.zeros((len(cxx), len(carids)))  # new arrays to calculate deltas
                    dy = np.zeros((len(cyy), len(carids)))  # new arrays to calculate deltas

                    for i in range(len(cxx)):  # loops through all centroids

                        for j in range(len(carids)):  # loops through all recorded car ids

                            # acquires centroid from previous frame for specific carid
                            oldcxcy = df.iloc[int(framenumber - 1)][str(carids[j])]

                            # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                            curcxcy = np.array([cxx[i], cyy[i]])

                            if not oldcxcy:  # checks if old centroid is empty in case car leaves screen and new car shows

                                continue  # continue to next carid

                            else:  # calculate centroid deltas to compare to current frame position later

                                dx[i, j] = oldcxcy[0] - curcxcy[0]
                                dy[i, j] = oldcxcy[1] - curcxcy[1]

                    for j in range(len(carids)):  # loops through all current car ids

                        sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # sums the deltas wrt to car ids

                        # finds which index carid had the min difference and this is true index
                        correctindextrue = np.argmin(np.abs(sumsum))
                        minx_index = correctindextrue
                        miny_index = correctindextrue

                        # acquires delta values of the minimum deltas in order to check if it is within radius later on
                        mindx = dx[minx_index, j]
                        mindy = dy[miny_index, j]

                        if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                            # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                            # delta could be zero if centroid didn't move

                            continue  # continue to next carid

                        else:

                            # if delta values are less than maximum radius then add that centroid to that specific carid
                            if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:

                                # adds centroid to corresponding previously existing carid
                                df.at[int(framenumber), str(carids[j])] = [cxx[minx_index], cyy[miny_index]]
                                minx_index2.append(minx_index)  # appends all the indices that were added to previous carids
                                miny_index2.append(miny_index)

                    for i in range(len(cxx)):  # loops through all centroids

                        # if centroid is not in the minindex list then another car needs to be added
                        if i not in minx_index2 and miny_index2:

                            df[str(totalcars)] = ""  # create another column with total cars
                            totalcars = totalcars + 1  # adds another total car the count
                            t = totalcars - 1  # t is a placeholder to total cars
                            carids.append(t)  # append to list of car ids
                            df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # add centroid to the new car id

                        elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                            # checks if current centroid exists but previous centroid does not
                            # new car to be added in case minx_index2 is empty

                            df[str(totalcars)] = ""  # create another column with total cars
                            totalcars = totalcars + 1  # adds another total car the count
                            t = totalcars - 1  # t is a placeholder to total cars
                            carids.append(t)  # append to list of car ids
                            df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # add centroid to the new car id

            # The section below labels the centroids on screen

            currentcars = 0  # current cars on screen
            currentcarsindex = []  # current cars on screen carid index

            for i in range(len(carids)):  # loops through all carids

                if df.at[int(framenumber), str(carids[i])] != '':
                    # checks the current frame to see which car ids are active
                    # by checking in centroid exists on current frame for certain car id

                    currentcars = currentcars + 1  # adds another to current cars on screen
                    currentcarsindex.append(i)  # adds car ids to current cars on screen

            for i in range(currentcars):  # loops through all current car ids on screen

                # grabs centroid of certain carid for current frame
                curcent = df.iloc[int(framenumber)][str(carids[currentcarsindex[i]])]

                # grabs centroid of certain carid for previous frame
                oldcent = df.iloc[int(framenumber - 1)][str(carids[currentcarsindex[i]])]

                if curcent:  # if there is a current centroid

                    # On-screen text for current centroid
                    cv2.putText(image, "Centroid" + str(curcent[0]) + "," + str(curcent[1]),
                                (int(curcent[0]), int(curcent[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.putText(image, "ID:" + str(carids[currentcarsindex[i]]), (int(curcent[0]), int(curcent[1] - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

                    cv2.drawMarker(image, (int(curcent[0]), int(curcent[1])), (0, 0, 255), cv2.MARKER_STAR, markerSize=5,
                                thickness=1, line_type=cv2.LINE_AA)

                    if oldcent:  # checks if old centroid exists
                        # adds radius box from previous centroid to current centroid for visualization
                        xstart = oldcent[0] - maxrad
                        ystart = oldcent[1] - maxrad
                        xwidth = oldcent[0] + maxrad
                        yheight = oldcent[1] + maxrad
                        cv2.rectangle(image, (int(xstart), int(ystart)), (int(xwidth), int(yheight)), (0, 125, 0), 1)

                        # checks if old centroid is on or below line and curcent is on or above line
                        # to count cars and that car hasn't been counted yet
                        if oldcent[1] >= lineypos2 and curcent[1] <= lineypos2 and carids[
                            currentcarsindex[i]] not in caridscrossed:

                            carscrossedup = carscrossedup + 1
                            cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 0, 255), 5)
                            caridscrossed.append(
                                currentcarsindex[i])  # adds car id to list of count cars to prevent double counting

                        # checks if old centroid is on or above line and curcent is on or below line
                        # to count cars and that car hasn't been counted yet
                        elif oldcent[1] <= lineypos2 and curcent[1] >= lineypos2 and carids[
                            currentcarsindex[i]] not in caridscrossed:

                            carscrosseddown = carscrosseddown + 1
                            cv2.line(image, (0, lineypos2), (width, lineypos2), (0, 0, 125), 5)
                            caridscrossed.append(currentcarsindex[i])

            # Top left hand corner on-screen text
            cv2.rectangle(image, (0, 0), (250, 100), (255, 0, 0), -1)  # background rectangle for on-screen text

            cv2.putText(image, "Cars in Area: " + str(currentcars), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)

            cv2.putText(image, "Cars Crossed Up: " + str(carscrossedup), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0),
                        1)

            cv2.putText(image, "Cars Crossed Down: " + str(carscrosseddown), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 170, 0), 1)

            cv2.putText(image, "Total Cars Detected: " + str(len(carids)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 170, 0), 1)

            cv2.putText(image, "Frame: " + str(framenumber) + ' of ' + str(frames_count), (0, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 170, 0), 1)

            cv2.putText(image, 'Time: ' + str(round(framenumber / fps, 2)) + ' sec of ' + str(round(frames_count / fps, 2))
                        + ' sec', (0, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)

            # displays images and transformations
            cv2.imshow("countours", image)
        # cv2.moveWindow("countours", 0, 0)

        # cv2.imshow("fgmask", fgmask)
        # cv2.moveWindow("fgmask", int(width * ratio), 0)

        # cv2.imshow("closing", closing)
        # cv2.moveWindow("closing", width, 0)

        # cv2.imshow("opening", opening)
        # cv2.moveWindow("opening", 0, int(height * ratio))

        # cv2.imshow("dilation", dilation)
        # cv2.moveWindow("dilation", int(width * ratio), int(height * ratio))

        # cv2.imshow("binary", bins)
        # cv2.moveWindow("binary", width, int(height * ratio))

            video.write(image)  # save the current image to video file from earlier

            # adds to framecount
            framenumber = framenumber + 1

            k = cv2.waitKey(int(1000/fps)) & 0xff  # int(1000/fps) is normal speed since waitkey is in ms
            if k == 27:
                break

        else:  # if video is finished then break loop

            break

    cap.release()
    cv2.destroyAllWindows()

    # saves dataframe to csv file for later analysis
    # df.to_csv('traffic.csv', sep=',')













def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''





# @app.route("/")
# def hello():
#     return "Welcome to machine learning model APIs!"


@app.route("/tripsyDetection/<image>")
def tripsyDetection(image):
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path , "tripsyModel/yolo.h5"))
    detector.loadModel()
    
    uploads=os.path.join(app.config['Third_Eye_Media'])
    imageIn=os.path.join(uploads,'%s'%image)
    imageOut=os.path.join(uploads+"/out",'%s'%image)


    custom_objects = detector.CustomObjects(person=True, motorcycle=True)

    returned_image, detections, extracted_objects = detector.detectCustomObjectsFromImage(output_type="array", extract_detected_objects=True,input_image=os.path.join(execution_path, imageIn), output_image_path=os.path.join(execution_path , imageOut), custom_objects=custom_objects, minimum_percentage_probability=65)
    #, output_type="array", extract_detected_objects=True
    print("done, Continue!!")


    # print the output image first
    plt.imshow(returned_image)
    plt.show()
    print("--------------------------------")
    #  second is the classes which are detected
    # print(detections)
    for eachObject in detections:
        print(eachObject['name'] ,":", eachObject['percentage_probability'],":", eachObject['box_points'])
    print("--------------------------------")
    # third extracted images from the classes 
    for eachExtract in extracted_objects:
        plt.imshow(eachExtract)
        plt.show()
    #    print(eachExtract['box_points'])
    print("--------------------------------")


    cnt=0
    for indx in range(len(detections)):
        if 'motorcycle' in detections[indx]['name']:
            cnt+=1
    print(cnt)


    index=-1
    resultArray=[] 
    for indx in range(len(detections)):
        if 'motorcycle' in detections[indx]['name']:
            
            index=indx
            print(index)
    # capturing the moterbike in data with index to check with the peoples


    resultArray=[]                   #Peoples on the bike 

    for i in range(len(detections)):
        if 'motorcycle' != detections[index]['name']:
            pass
        else:
            if detections[i]['box_points'][0] >= detections[index]['box_points'][0] & detections[i]['box_points'][2]<=detections[index]['box_points'][2]:
                if (detections[i]['box_points'][0]-detections[index]['box_points'][0])>75 or (detections[i]['box_points'][3]-detections[index]['box_points'][3])<10:
                    if (detections[index]['box_points'][3]-detections[i]['box_points'][3])<60:
                        if ((detections[index]!=detections[i])):
    #                   not considering vehicle
                            resultArray.append(detections[i]['box_points'])
                            print(detections[i])
                            plt.imshow(extracted_objects[i])
                            plt.show()
                    else:
                        pass
                else:
                    pass
            else:
                pass
        


    print("Result Array is ",resultArray)
    coordinates=list()
    if len(resultArray)>=2:
        print("Tripsy")
    #print(resultArray)
        resultArray.sort()
    #sorting the array to get the min and max points in the images
        items=[resultArray[0][0],resultArray[0][1],resultArray[len(resultArray)-1][2],resultArray[len(resultArray)-1][3]]
        for item in items:
            coordinates.append(item)
    elif len(resultArray)==1:
        items=[resultArray[0][0],resultArray[0][1],resultArray[0][2],resultArray[0][3]]
        for item in items:
            coordinates.append(item)
        #Single
    else:
        items=[0,0,0,0]
        for item in items:
            coordinates.append(item)
    print("done")
    print("coordinates of rectangle",coordinates)
    # coordinates conntain the coordinates of the rectangle which will show the    *tripsy*    is here
    


    I = np.asarray(PIL.Image.open(imageIn, mode='r').convert('RGB')) 

    rslt=""
    if len(resultArray)>=2:
        print("Tripsy Found")
        I = np.asarray(PIL.Image.open(imageIn, mode='r').convert('RGB'))              #Open as a array and perform the operation
    # print(I)
        img = cv2.rectangle(I,(coordinates[0],coordinates[1]),(coordinates[2],coordinates[3]), (0,0,255),4)
        rslt=rslt+"Tripsy Found!"
        cv2.imshow("Image",img)
        outImg = Image.fromarray(img)
        outImg.save(imageOut)
        
        cv2.waitKey(0)
    else:
        img = cv2.rectangle(I,(coordinates[0],coordinates[1]),(coordinates[2],coordinates[3]), (0,255,0),4)
        cv2.imshow("Image",img)
        outImg = Image.fromarray(img)
        outImg.save(imageOut)
        print("No Tripsy Found")
        rslt=rslt+"Tripsy Not Found!"
        cv2.waitKey(0)
    return rslt

        






@app.route("/numberPlate/<video>")
def numberPlate(video):
    
    plate_cascade= cv2.CascadeClassifier('numberPlateModel/cascades/data/haarcascade_russian_plate_number.xml')

    videopath=os.path.join(app.config['Third_Eye_Media'],'%s'%video)

    cap = cv2.VideoCapture(videopath)
    print("\n\n\n",type(video))
    # cap = cv2.VideoCapture(0)
    # y_cord_mx=80
    y_cord_mn=20
    count = 0
    if cap.isOpened():
        while True:
            check, frame = cap.read()
            if check:
                gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('color Frame', frame)
                plates= plate_cascade.detectMultiScale(gray_frame, scaleFactor= 1.5, minNeighbors=5)
            
                
                for (x,y, w, h) in plates:
                    
                    region_of_interest_gray= gray_frame[y: y+h, x: x+w]
                    region_of_interest_gray_to_Display= gray_frame[y-20: y+h+20, x-20: x+w+20]
                    rect_color=(255, 0, 0)
                    stroke=2
                    cv2.rectangle(frame, (x, y), ( x+w , y+h ), rect_color, stroke )   #param: frame, co-ordinates, Height- width , color of rect. , Stroke
                    cv2.imshow("Tkinter and OpenCV", frame)
    #                 print(y)
                    if ((y_cord_mn<=y)):
                        count=count+1
                        # print(x, y, w, h)
                        region_image= app.config['Third_Eye_Plates_Out']+str(count)+'.png'
                        cv2.imshow("plates", region_of_interest_gray_to_Display)
                        cv2.imwrite(region_image, region_of_interest_gray_to_Display)               
                    
                key = cv2.waitKey(50)

                if key == ord('q'):
                    break
            else:
                print('Frame not available')
                print(cap.isOpened())

    cap.release()
    cv2.destroyAllWindows()
    return "Video Saved!"














@app.route("/helmet/<path:video>")                   # <video> is variable
def helmet(video):
    print("Working Wait It may Take Some Minutes! ")
    FILE_OUTPUT = app.config['Third_Eye_Out']+video
    
    # initialize the list of class labels MobileNet SSD was trained to detect
    # generate a set of bounding box colors for each class
    CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    #CLASSES = ['motorbike', 'person']
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('helmetModel/MobileNetSSD_deploy.prototxt.txt', 'helmetModel/MobileNetSSD_deploy.caffemodel')

    print('Loading helmet model...')
    loaded_model = load_model('helmetModel/new_helmet_model.h5')
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # initialize the video stream,
    print("[INFO] starting video stream...")

    # Loading the video file
    videopath=os.path.join(app.config['Third_Eye_Media'],'%s'%video)

    print(videopath)
    cap = cv2.VideoCapture(videopath) 

    currentFrame = 0

    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float


    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (int(width),int(height)))

    # time.sleep(2.0)

    # Starting the FPS calculation
    fps = FPS().start()

    # loop over the frames from the video stream
    # i = True
    while True:
        # i = not i
        # if i==True:

        try:
            # grab the frame from the threaded video stream and resize it
            # to have a maxm width and height of 600 pixels
            ret, frame = cap.read()

            # resizing the images
            frame = imutils.resize(frame, width=600, height=600)

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            
            # Resizing to a fixed 300x300 pixels and normalizing it.
            # Creating the blob from image to give input to the Caffe Model
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

            # pass the blob through the network and obtain the detections and predictions
            net.setInput(blob)

            detections = net.forward()  # getting the detections from the network
            
            persons = []
            person_roi = []
            motorbi = []
            
            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence associated with the prediction
                confidence = detections[0, 0, i, 2]
                
                # filter out weak detections by ensuring the confidence
                # is greater than minimum confidence
                if confidence > 0.5:
                    
                    # extract index of class label from the detections
                    idx = int(detections[0, 0, i, 1])
                    
                    if idx == 15:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        # roi = box[startX:endX, startY:endY/4] 
                        # person_roi.append(roi)
                        persons.append((startX, startY, endX, endY))

                    if idx == 14:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        motorbi.append((startX, startY, endX, endY))

            xsdiff = 0
            xediff = 0
            ysdiff = 0
            yediff = 0
            p = ()
            
            for i in motorbi:
                mi = float("Inf")
                for j in range(len(persons)):
                    xsdiff = abs(i[0] - persons[j][0])
                    xediff = abs(i[2] - persons[j][2])
                    ysdiff = abs(i[1] - persons[j][1])
                    yediff = abs(i[3] - persons[j][3])

                    if (xsdiff+xediff+ysdiff+yediff) < mi:
                        mi = xsdiff+xediff+ysdiff+yediff
                        p = persons[j]
                        # r = person_roi[j]


                if len(p) != 0:

                    # display the prediction
                    label = "{}".format(CLASSES[14])
    # 	            print("[INFO] {}".format(label))
    # 	            cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), COLORS[14], 2)            #Vehicle Body
                    y = i[1] - 15 if i[1] - 15 > 15 else i[1] + 15
    # 	            cv2.putText(frame, label, (i[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[14], 2)   
                    label = "{}".format(CLASSES[15])
    # 	            print("[INFO] {}".format(label))

    # 	            cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), COLORS[15], 2)           #Person Body
                    y = p[1] - 15 if p[1] - 15 > 15 else p[1] + 15

                    roi = frame[p[1]:p[1]+(p[3]-p[1])//4, p[0]:p[2]]
                    # print(roi)
                    if len(roi) != 0:
                        img_array = cv2.resize(roi, (50,50))
                        gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                        img = np.array(gray_img).reshape(1, 50, 50, 1)
                        img = img/255.0
                        # cv2.imshow("img",img)
                        # print(img)
                        prediction = loaded_model.predict_proba([img])
                        
                        cv2.rectangle(frame, (p[0], p[1]), (p[0]+(p[2]-p[0]), p[1]+(p[3]-p[1])//4), [0,0,255], 2)
                        if(round(prediction[0][0],2))<=.90:
                                cv2.putText(frame, "No Helmet", (p[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)
    #                             print(round(prediction[0][0],2))
                                        
    #             cv2.putText(frame, "Helmet", (p[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)
                
        except:
            pass

        cv2.imshow('Frame', frame)  # Displaying the frame
        # Saves for video
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): # if 'q' key is pressed, break from the loop
            break
        
        # update the FPS counter
        fps.update()
            

    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    cv2.destroyAllWindows()
    cap.release()   # Closing the video stream 
    out.release()
    return "Video Saved!"


if __name__ == '__main__':
    app.secret_key='Third_Eye'
    app.run(debug=True)
    # socketio.run(app)
    