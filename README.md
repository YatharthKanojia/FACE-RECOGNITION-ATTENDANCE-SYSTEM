INTRODUCTION:
This project is a Face Recognition Attendance System that uses a webcam to detect and recognize faces, marking attendance in a CSV file. The system uses OpenCV for face detection, Scikit-Learn for face recognition, and Streamlit for displaying the attendance data in a web interface.

FEATURES:
* Face detection using OpenCV's Haar cascades.
*  Face recognition using a K-Nearest Neighbors (KNN) classifier.
* Real-time attendance marking.
* Visual interface for attendance using a background image.
* Streamlit app for displaying attendance data.

PREREQUISITIES:
* Python 3.x
* OpenCV
* Scikit-Learn
* Streamlit
* Pandas
* pywin32

'add_faces.py' :
This script is used to capture and store face data. It allows you to add new faces to the system by capturing images from the webcam and saving the data for later use in recognition.

'test.py' :
This script performs the face recognition and marks attendance. It uses the webcam to capture real-time video, detects faces, and recognizes them using a K-Nearest Neighbors (KNN) classifier. Recognized faces are marked in a CSV file with the current timestamp.

'app.py' :
This Streamlit application displays the attendance data in a web interface. It automatically refreshes to show the latest attendance records.

'data --->  haarcascade_frontalface_default.xml / faces_data.pkl / names_pkl' :
This directory contains the Haar cascades XML file for face detection and the pickled data files for storing face embeddings and labels.

'Attendance' ;
This directory contains the CSV files where attendance records are stored. Each file is named with the current date.

'Background Image' :
The background image used in the face recognition system should be of dimensions 1280x720 pixels. You can replace the existing background image by placing a new image with the same name (background.png).
