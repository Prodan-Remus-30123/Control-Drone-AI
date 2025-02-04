import csv
import itertools
import copy
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Function to preprocess and normalize landmarks
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

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20

# Dictionary to keep track of the number of entries saved for each class
class_tracker = {str(i): 0 for i in range(10)}

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)
    if hands:
        hand = hands[0]
        lmlist1 = hand["lmList"]  # lista de landmark-uri, 21 la numar

        # Preprocess and normalize landmarks
        normalized_landmarks = pre_process_landmark(lmlist1)

        # Display normalized landmarks (for testing purposes)
        #print(normalized_landmarks)

        # Get keyboard input
        key = cv2.waitKey(1)
        if key >= 48 and key <= 57:  # Check if pressed key is a number from 0 to 9
            class_label = chr(key)  # Convert ASCII code to character (e.g., 48 -> '0', 49 -> '1', ..., 57 -> '9')

            # Write normalized landmarks to CSV file
            with open('Data.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([class_label] + normalized_landmarks)

            # Update class tracker
            class_tracker[class_label] += 1
            print("Data saved to Data.csv for class:", class_label)
            print("Entries for each class:", class_tracker)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
