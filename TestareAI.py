import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image

# Load the saved model
model = load_model('model8692.h5')


# Function to preprocess the image before prediction
def preprocess_image(img):
    # Resize image to match model input shape (100x100)
    resized_img = cv2.resize(img, (150, 150))
    return resized_img


# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the image
    preprocessed_img = preprocess_image(frame)
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)

    # Make prediction
    prediction = model.predict(preprocessed_img)

    # Check if the predicted probability is higher than 75%
    if np.max(prediction) > 0.75:
        # Get the index of the class with the highest probability
        predicted_class = np.argmax(prediction)

        # Print the prediction if the probability is higher than 75%
        print("Predicted class:", predicted_class)
        print("Probability:", np.max(prediction))

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()