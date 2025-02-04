import cv2
from djitellopy import tello
import KeyPressModule as kp
from time import sleep
import time
import matplotlib.pyplot as plt
import pandas as pd
import keyboard

excel_file_path = 'data_log.xlsx'

# Create a window with a trackbar
cv2.namedWindow('Trackbar Example')

# Initial value for the variable
speed = 0

# Callback function for the trackbar
def on_trackbar_change(value):
    global speed
    speed = value
    print("Variable Value:", speed)

# Create a trackbar
cv2.createTrackbar('Variable', 'Trackbar Example', speed, 100, on_trackbar_change)

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamon()
me.takeoff()
me.move_up(90)

w,h=360,240 # setez width si height a imaginii de 360 si 240
fbRange=[5000,6800]
def getKeyboardInput():
    lr,fb,ud,yw =0, 0, 0, 0
  #  speed=50
    if kp.getKey("LEFT"): lr=-speed
    elif kp.getKey("RIGHT"): lr=speed

    if kp.getKey("UP"): fb = -speed
    elif kp.getKey("DOWN"): fb=speed

    if kp.getKey("w"): ud = -speed
    elif kp.getKey("s"): ud = speed

    if kp.getKey("a"): yw = -speed
    elif kp.getKey("d"): yw = speed

    if kp.getKey("q"): me.land()
    if kp.getKey("e"): me.takeoff()

    return [lr,fb,ud,yw]


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


def plot(time,speed,distance):
    plt.clf() #clear
    plt.plot(time,speed,marker='o',linestyle='-',label='Speed (cm/s)')
    plt.plot(time,distance,marker='o',linestyle='-',label='Distance (cm)')
    plt.title('Speed and Distance in Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Value')
    plt.legend()
    plt.draw()
    plt.pause(0.05)

def get_current_time():
    return time.time()

time_data = []
speed_data = []
distance_data = []

start_time = get_current_time()

plot(time_data,speed_data,distance_data)

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w, h))
    img, info = findFace(img)
    cv2.imshow("Output", img)
    vals=getKeyboardInput()
    me.send_rc_control(vals[0],vals[1],vals[2],vals[3])
    print(f"lr={vals[0]}, fb={vals[1]}, ud={vals[2]}, yw={vals[3]} area={info[1]}")

    time_data.append(get_current_time() - start_time)
    speed_data.append(vals[1])
    distance_data.append(info[1])
    plot(time_data,speed_data,distance_data)

    #print(me.get_height(),me.get_speed_x(),me.get_speed_y(),me.get_speed_z())
    sleep(0.05)

    # Break the loop when the user presses 'Esc' key
    if cv2.waitKey(1) == 27:
        break
    if keyboard.is_pressed('r'):
        # Save data to Excel file when 'r' is pressed
        data_dict = {
            'Time': time_data,
            'Speed': speed_data,
            'Distance': distance_data
        }
        df = pd.DataFrame(data_dict)
        df.to_excel(excel_file_path, index=False)
        print(f'Data saved to {excel_file_path}')


plt.ioff()
plt.show()
# Release resources
cv2.destroyAllWindows()