from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load pre-trained models and data
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Debugging: Print shapes of loaded data
print('Shape of Faces matrix --> ', FACES.shape)
print('Length of Labels list --> ', len(LABELS))

# Ensure LABELS and FACES are of the same length
if FACES.shape[0] != len(LABELS):
    # Determine the minimum length of FACES and LABELS
    min_length = min(FACES.shape[0], len(LABELS))
    
    # Trim the longer array to match the shorter one
    FACES = FACES[:min_length]
    LABELS = LABELS[:min_length]

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image
imgBackground = cv2.imread("background.png")

# Column names for the CSV file
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)  # Consistent resizing and flattening
        output = knn.predict(resized_img)
        
        # Get timestamp
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        
        # Check if the attendance file already exists
        exist = os.path.isfile(f"Attendance/Attendance_{date}.csv")
        
        # Draw rectangles and text on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        
        # Prepare attendance record
        attendance = [str(output[0]), str(timestamp)]
    
    # Overlay the frame on the background image
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)
    
    k = cv2.waitKey(1)
    
    # Handle key presses
    if k == ord('o'):  # 'o' for marking attendance
        speak("Attendance Taken..")
        time.sleep(5)
        
        # Save attendance record
        with open(f"Attendance/Attendance_{date}.csv", "+a") as csvfile:
            writer = csv.writer(csvfile)
            if exist:
                writer.writerow(attendance)
            else:
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
    
    if k == ord('q'):  # 'q' to quit
        break

# Release resources
video.release()
cv2.destroyAllWindows()
