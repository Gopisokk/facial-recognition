import cv2
import os
import numpy as np
from tkinter import *
from tkinter import messagebox
from datetime import datetime

# Set the path for the training images
TRAINING_IMAGES_PATH = r'C:\Users\kabilan\Documents\FaceRecognitionAttendanceSystem\training_images'
TRAINER_FILE = r'C:\Users\kabilan\Documents\FaceRecognitionAttendanceSystem\trainer.yml'

def load_images_and_labels(path):
    faces = []
    ids = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            id = int(os.path.split(filename)[-1].split('.')[1])  # Assuming format 'User.ID.jpg'
            faces.append(img)
            ids.append(id)
    return faces, ids

def train_images():
    try:
        faces, ids = load_images_and_labels(TRAINING_IMAGES_PATH)
        if len(faces) == 0:
            messagebox.showerror("Error", "No training images found!")
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        recognizer.save(TRAINER_FILE)
        messagebox.showinfo("Success", "Training completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def take_attendance():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_FILE)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                if confidence < 100:
                    name = f"User.{id}"  # Assuming name format is 'User.ID'
                    confidence_str = f"  {round(100 - confidence)}%"
                    cv2.putText(frame, f'ID: {id}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(frame, f'Name: {name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Log attendance
                    with open('attendance.csv', 'a') as f:
                        f.write(f'{id},{name},{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                else:
                    cv2.putText(frame, 'Unknown', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI
root = Tk()
root.title("Face Recognition Attendance System")

frame = Frame(root)
frame.pack(pady=20)

btn_train = Button(frame, text="Train Images", command=train_images)
btn_train.pack(side=LEFT, padx=10)

btn_attendance = Button(frame, text="Take Attendance", command=take_attendance)
btn_attendance.pack(side=LEFT, padx=10)

root.mainloop()
