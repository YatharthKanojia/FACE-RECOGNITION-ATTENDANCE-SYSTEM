import cv2
import pickle
import numpy as np
import os

def resize_face_data(faces_data, target_shape):
    resized_faces_data = []
    for face in faces_data:
        try:
            face_reshaped = face.reshape((50, 50, 3))  # Assume initial shape is (50, 50, 3)
            face_resized = cv2.resize(face_reshaped, target_shape)
            resized_faces_data.append(face_resized.flatten())
        except Exception as e:
            print(f"[ERROR] Could not resize face data: {e}")
            continue
    return np.array(resized_faces_data)

def collect_faces():
    # Initialize the face detector and video capture
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    faces_data = []
    i = 0
    name = input("Enter Your Name: ")

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) < 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) >= 100:
            break

    video.release()
    cv2.destroyAllWindows()

    faces_data = np.array(faces_data)
    faces_data = faces_data.reshape(-1, 50*50*3)  # Ensure correct shape

    # Load existing data
    data_dir = 'data/'
    names_path = os.path.join(data_dir, 'names.pkl')
    faces_path = os.path.join(data_dir, 'faces_data.pkl')

    if os.path.exists(names_path):
        with open(names_path, 'rb') as f:
            names = pickle.load(f)
        names.extend([name] * 100)
    else:
        names = [name] * 100

    if os.path.exists(faces_path):
        with open(faces_path, 'rb') as f:
            existing_faces = pickle.load(f)

        # Convert existing_faces to a numpy array
        existing_faces = np.array(existing_faces)

        # Reshape existing_faces to ensure it has 2 dimensions
        if existing_faces.ndim == 1:
            existing_faces = existing_faces.reshape(-1, 50*50*3)
        elif existing_faces.ndim == 2:
            # Check if the second dimension is not equal to 50*50*3 and reshape if needed
            if existing_faces.shape[1] != 50*50*3:
                existing_faces = existing_faces.reshape(-1, 50*50*3)

        target_shape = (50, 50)
        faces_data = resize_face_data(faces_data, target_shape)

        all_faces = np.concatenate((existing_faces, faces_data), axis=0)
    else:
        all_faces = faces_data

    # Save updated data
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)
    with open(faces_path, 'wb') as f:
        pickle.dump(all_faces, f)

if __name__ == "__main__":
    collect_faces()
