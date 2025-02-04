import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model

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

def predict_hand(detector, img_original, rect_coords,img_copy):
    hands, img = detector.findHands(img_copy, draw=True)
    if hands:
        hand = hands[0]
        lmlist1 = hand["lmList"]  # lista de landmark-uri, 21 la numar

        knuckle_list=[lmlist1[5][:2],lmlist1[9][:2],lmlist1[13][:2],lmlist1[17][:2]]

        x, y, w, h = hand['bbox']
        if w is not None and h is not None:
            imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

            if hand_in_box(knuckle_list, rect_coords):
                # Preprocess the image for the model
                processed_img = preprocess_image(imgCrop)

                # Make prediction
                try:
                    prediction = model.predict(processed_img)
                    if prediction is not None and np.max(prediction) > 0.8:
                        # Get the index of the class with the highest probability
                        predicted_class = np.argmax(prediction)
                        gesture_label = gesture_labels[predicted_class]
                        probability = prediction[0][predicted_class]
                        # Draw prediction text on the image
                        cv2.putText(img_original, f"Prediction: {gesture_label} (Probability: {probability:.2f})", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                except ValueError as e:
                    print("Error occurred during prediction:", e)

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                cv2.imshow('ImageCrop', imgCrop)


# Prepare hand tracking
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

cap = cv2.VideoCapture(0)
w, h = 500, 500  # Set the width and height of the image to 360 and 240

# Load your AI model (Replace this with your actual model loading code)
model = load_model('model1000.h5')

# Define gesture labels
gesture_labels = {0: "Down", 1: "Left", 2: "Right", 3: "Rock", 4: "Stay", 5: "Up"}

while True:
    success, img = cap.read()
    img = cv2.resize(img, (w, h))

    img, info = findFace(img)

    #Create a copy of the image so it doesn't have the blue rectangle
    img_copy = img.copy()

    # Draw a rectangle to the left and slightly above the detected face rectangle
    face_center, face_area = info
    if face_area != 0:
        x, y = face_center
        face_width = int(face_area ** 0.5)
        # Adjust the coordinates for the new rectangle
        new_x = x - 2 * face_width
        new_y = y - face_width // 2
        rect_coords = [new_x, new_y, new_x + face_width, new_y + face_width]

        # Draw the red rectangle on the original image
        cv2.rectangle(img, (new_x, new_y), (new_x + face_width, new_y + face_width), (255, 0, 0), 2)

        predict_hand(detector, img, rect_coords,img_copy)

    cv2.imshow("Output", img)  # Display the original image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
