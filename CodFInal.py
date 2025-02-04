import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
from djitellopy import tello
import time
import copy
import csv
import itertools

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list
def findFace(img):
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw a rectangle around the face
        # Calculate the center point and area
        cx = x + w // 2  # Integer division
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

def preprocess_image(img):
    # Check if imgCrop is not empty
    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        # Resize the image to your model's input size
        img = cv2.resize(img, (150, 150))
        # Convert image to required format (e.g., normalization, color space conversion)
        img = img / 255.0  # normalize to range [0, 1]
        img = np.expand_dims(img, axis=0)  # add batch dimension
    return img

def hand_in_box(point_list,rect_coords):
    x_tl, y_tl, x_br, y_br = rect_coords
    for point in point_list:
        x, y = point
        if not (x_tl <= x <= x_br and y_tl <= y <= y_br):
            return False
    return True

def predict_hand(detector, img_original, rect_coords, prev_gesture,k,img_copy):
    hands, img = detector.findHands(img_copy, draw=True)
    if hands:
        hand = hands[0]
        lmlist1 = hand["lmList"]  # lista de landmark-uri, 21 la numar

        # Preprocess and normalize landmarks
        normalized_landmarks = pre_process_landmark(lmlist1)

        knuckle_list=[lmlist1[5][:2],lmlist1[9][:2],lmlist1[13][:2],lmlist1[17][:2]]

        x, y, w, h = hand['bbox']
        if w is not None and h is not None:
            imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

            if hand_in_box(knuckle_list, rect_coords):
                # Preprocess the image for the model
                processed_img = preprocess_image(imgCrop)

                # Make prediction
                try:
                    normalized_landmarks = np.array(normalized_landmarks)  # Convert to NumPy array if not already
                    # Reshape the input data
                    normalized_landmarks = normalized_landmarks.reshape(1,-1)  # Reshape to have 1 sample and 63 features
                    prediction = model.predict(normalized_landmarks)

                    if prediction is not None and np.max(prediction) > 0.8:
                        # Get the index of the class with the highest probability
                        predicted_class = np.argmax(prediction)
                        gesture_label = gesture_labels[predicted_class]
                        if gesture_label == prev_gesture:
                            k=k+1
                        else:
                            k=0
                        prev_gesture = gesture_label
                        probability = prediction[0][predicted_class]
                        # Draw prediction text on the image
                        cv2.putText(img_original, f"Prediction: {gesture_label} (Probability: {probability:.2f})", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        print(k)
                except ValueError as e:
                    print("Error occurred during prediction:", e)

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                cv2.imshow('ImageCrop', imgCrop)

    return prev_gesture,k

def trackFace(info,w,pid,pid2,pError, pErroru):
    area = info[1]
    x,y=info[0]
    fb = 0  #forward & backwards speed
    p=3

    error = x-w//2 # x e valoarea pentru fata si w//2 inseamna mijlocul imaginii, eroarea e cat de daparte e fata de centrul imaginii
    speed = pid[0] *error+ pid[1]*(error-pError)
    speed = int(np.clip(speed,-100,100))#are voie sa se miste cu o viteza de maxim 100

    erroru = h//2 -y
    speedu = pid2[0] * erroru + pid2[1] * (erroru - pErroru)
    speedu = int(np.clip(speedu, -100, 100))

    #Miscare drona fata spate in functie de arie
    if area>fbRange[0] and area<fbRange[1]:
        fb = 0
        errorFB = 0
    elif area>fbRange[1]:
        errorFB = (fbRange[1] - area) // 535
        fb = p*errorFB
    elif area<fbRange[0] and area!=0:
        errorFB = (fbRange[0] - area) // 535
        fb = p * errorFB
    if x == 0:
        speedu = 0
        erroru = 0

    #In caz ca nu avem centru
    if x == 0:
        speed = 0
        error = 0
    if area == 0 or area is None:
        fb=0

    fb = int(np.clip(fb, -100, 100))
    print(speed, fb, area)
    me.send_rc_control(0,fb,speedu,speed)
    return error, erroru

def drawHandRectangle(info,img):
    # Draw a rectangle to the left and slightly above the detected face rectangle
    face_center, face_area = info
    if face_area != 0:
        x, y = face_center
        face_width = int(face_area ** 0.5)
        # Adjust the coordinates for the new rectangle
        new_x = x - 2 * face_width
        new_y = y - face_width // 2
        rect_coords = [new_x, new_y, new_x + face_width, new_y + face_width]
        cv2.rectangle(img, (new_x, new_y), (new_x + face_width, new_y + face_width), (255, 0, 0), 2)
        return rect_coords

#Init Tello
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
me.takeoff()
me.move_up(90)
me.send_rc_control(0, 0, 0, 0)

#Setting up Parameters
fbRange=[6000,7800]
pid=[0.4,0.4,0]
pid2=[0.4,0.4,0]
pError=0;
pErroru=0;

# Prepare hand tracking
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

cap = cv2.VideoCapture(0)
w, h = 500, 500  # Set the width and height of the image to 360 and 240

# Load your AI model (Replace this with your actual model loading code)
model = load_model('modelC9979.h5')

# Define gesture labels
gesture_labels = {0: "Down", 1: "Left", 2: "Right", 3: "Rock", 4: "Stay", 5: "Up"}
prev_gesture=4
k=0
l=0

while True:
    # success, img = cap.read()
    img = me.get_frame_read().frame

    # img = cv2.resize(img, (w, h))
    img = cv2.resize(img, (w, h))
    picture = img.copy()

    #Find and draw the rectangle for the Face
    img, info = findFace(img)

    # Create a copy of the original image without the blue rectangle
    img_copy = img.copy()


    #Get coords and draw the rectangle for the hand
    rect_coords = drawHandRectangle(info,img)

    #Detect the hand gesture
    prev_gesture,k= predict_hand(detector, img, rect_coords,prev_gesture,k,img_copy)

    if k<5:
        #Track and Follow the Face
        pError, pErroru = trackFace(info, w, pid, pid2, pError, pErroru)
        print("Center", info[0], "Area", info[1])

    if k>=5:
        print(prev_gesture)
        if prev_gesture == 'Down':
            me.move_down(20)
            k=0
        elif prev_gesture == 'Left':
            me.move_left(20)
            k=0
        elif prev_gesture == 'Right':
            me.move_right(20)
            k=0
        elif prev_gesture == 'Rock':
            cv2.imwrite(f'output_image{l}.jpg', picture)
            print("Image saved successfully.")
            l=l+1
            k=0
        elif prev_gesture == 'Stay':
            me.send_rc_control(0, 0, 0, 0)
            k=0
        elif prev_gesture == 'Up':
            me.move_up(20)
            k=0

    cv2.imshow("Output", img)  # Display the original image

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
