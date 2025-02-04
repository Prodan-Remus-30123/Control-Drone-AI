import cv2
import numpy as np
from djitellopy import tello
import time

me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamon()
time.sleep(3)
me.takeoff()
me.move_up(90)
time.sleep(2)
me.send_rc_control(0, 0, 0, 0)

w,h=360,240 # setez width si height a imaginii de 360 si 240
#cap=cv2.VideoCapture(0)
fbRange=[5000,6800]

# File path
file_path = "integer_file.txt"
ok = 0

def findFace(img):
    faceCascade= cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    imgGray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray,1.2,8)

    myFaceListC = []
    myFaceListArea = []

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) #desenez patrat
        #iau punctul din centru si calculez aria
        cx = x+w // 2 # Impartire cu aproximare in jos 5/2=2.5 5//2=2
        cy = y+ h // 2
        area = w*h
        cv2.circle(img,(cx,cy),3,(0,255,0),cv2.FILLED)
        myFaceListC.append([cx,cy])
        myFaceListArea.append(area)
        #Trecem prin lista cu fete, ca sa luam cea mai apropiata fata = area cea mai mare
    if len(myFaceListArea) != 0:
        i=myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i],myFaceListArea[i]]
    else:
        return img,[[0,0],0]

def calculateDistance(info, ok):
    area = info[1]
    x, y = info[0]
    print(area)
    if ok == 0:
        if area > fbRange[0] and area < fbRange[1]:
            # Open the file in write mode ('w')
            with open(file_path, 'w') as file:
                # Write the integer to the file as a string
                file.write(str(area)+" ")
            ok = 1
            me.move_forward(50)
            time.sleep(3)
    if ok == 1:
        print("Dupa miscare: "+ str(area) )
        me.send_rc_control(0,0,0,0)


    return ok

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w, h))
    img, info = findFace(img)

    ok = calculateDistance(info, ok)

    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)


