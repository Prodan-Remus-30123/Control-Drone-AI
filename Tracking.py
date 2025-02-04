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
pid=[0.4,0.4,0.2]
pid2=[0.4,0.4,0]
pError=0
pErroru=0
iError = 0

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

def trackFace(info,w,pid,pid2,pError, pErroru,iError):
    area = info[1]
    x,y=info[0]
    fb = 0  #forward & backwards speed
    error = x-w//2 # x e valoarea pentru fata si w//2 inseamna mijlocul imaginii, eroarea e cat de daparte e fata de centrul imaginii

    iError += error * 0.1

    speed = pid[0] *error+ pid[1]*(error-pError) + pid[2] * iError
    speed = int(np.clip(speed,-100,100))#are voie sa se miste cu o viteza de maxim 100

    erroru = h//2 -y
    speedu = pid2[0] * erroru + pid2[1] * (erroru - pErroru)
    speedu = int(np.clip(speedu, -100, 100))

    #Miscare drona fata spate in functie de arie
    if area>fbRange[0] and area<fbRange[1]:
        fb = 0
    elif area>fbRange[1]:
        fb = -20
    elif area<fbRange[0] and area!=0:
        fb = 20

    if x == 0:
        speedu = 0
        erroru = 0

    #In caz ca nu avem centru
    if x == 0:
        speed = 0
        error = 0

    print(speed, fb, area)
    me.send_rc_control(0,fb,speedu,speed)
    return error, erroru, iError

while True:
   # _,img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img,(w,h))
    img,info=findFace(img)
    pError, pErroru,iError=trackFace( info, w, pid,pid2, pError, pErroru,iError)
    print("Center",info[0],"Area",info[1])

    cv2.imshow("Output",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break