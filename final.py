import tkinter as tk
from tkinter import Label, Entry, Button, Canvas
from PIL import Image, ImageTk
import cv2  
import os
import numpy as np
import mtcnn
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity as cs
from pymongo import MongoClient

# Initialize MongoDB Connection
client = MongoClient("mongodb://localhost:27017")
db = client["Face_Recognition"]
collection = db["embeddings"]

# Initialize OpenCV & MTCNN
cap = cv2.VideoCapture(0)
detector = mtcnn.MTCNN()

# Tkinter Window
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("700x600")

# Create a Canvas for webcam feed
canvas = Canvas(root, width=500, height=350)
canvas.pack()

# Entry field for user name
name_label = Label(root, text="For new Registeration Enter Your Name:")
name_label.pack()
name_entry = Entry(root)
name_entry.pack()

# Label to display results
result_label = Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

def update_frame():
    """ Updates the webcam feed inside Tkinter. """
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (500, 350))
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.img = img
    root.after(10, update_frame)

def register_new_user():
    """ Captures a face and stores it in the database. """
    user_name = name_entry.get()
    if not user_name:
        result_label.config(text="Please enter a name.")
        return

    ret, frame = cap.read()
    if ret:
        face = detector.detect_faces(frame)
        if face:
            x, y, w, h = face[0]['box']
            cropped_face = frame[y:y + h, x:x + w]

            # Save user image
            folder_name = "user_images"
            os.makedirs(folder_name, exist_ok=True)
            file_path = os.path.join(folder_name, f"{user_name}.jpg")
            cv2.imwrite(file_path, cropped_face)

            # Extract face embeddings
            face_info = DeepFace.represent(cropped_face, model_name="ArcFace")
            embedding = face_info[0]["embedding"]

            # Store in MongoDB
            collection.insert_one({"user_name": user_name, "embedding": embedding})
            result_label.config(text="User Registered Successfully!")

        else:
            result_label.config(text="No face detected!")
            
def recognize_face():
    """ Captures a face and recognizes it against stored users. """
    ret, frame = cap.read()
    if ret:
        face = detector.detect_faces(frame)
        if face:
            x, y, w, h = face[0]['box']
            cropped_face = frame[y:y + h, x:x + w]

            # Get face embedding
            face_info = DeepFace.represent(cropped_face, model_name="ArcFace")
            face_embedding = face_info[0]["embedding"]

            # Compare with stored embeddings
            records = collection.find()
            best_match = None
            highest_similarity = -1

            for record in records:
                stored_embedding = record["embedding"]
                similarity = cs(np.array(face_embedding).reshape(1, -1), np.array(stored_embedding).reshape(1, -1))[0][0]

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = record["user_name"]

            # Display result
            if highest_similarity >= 0.5:
                result_label.config(text=f"Face Recognized: {best_match}\nConfidence: {highest_similarity:.2f}")
            else:
                result_label.config(text="No Match Found!")

# Buttons
register_button = Button(root, text="Register", command=register_new_user, width=20, height=2)
register_button.pack(pady=5)

recognize_button = Button(root, text="Mark Attendance", command=recognize_face, width=20, height=2)
recognize_button.pack(pady=5)

exit_button = Button(root, text="Exit", command=root.quit, width=20, height=2)
exit_button.pack(pady=5)

update_frame()  # Start webcam feed
root.mainloop()

cap.release()
cv2.destroyAllWindows()
