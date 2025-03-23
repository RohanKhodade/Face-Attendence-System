import cv2
import numpy as np
import mtcnn
from deepface import DeepFace
from pymongo import MongoClient
import os

def register_new_user():
    detector = mtcnn.MTCNN()
    cap = cv2.VideoCapture(0)
    print("Detecting Face")
    
    while True:
        ret, frame = cap.read()
        face = detector.detect_faces(frame)
        if len(face) > 0:
            x, y, w, h = face[0]['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (25, 255, 0), 2)
            cv2.imshow("face", frame)
            cropped_face = frame[y:y + h + 10, x:x + w + 10]
        else:
            cv2.putText(frame, "detecting face", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 0, 255), 2)
            cv2.imshow("face", frame)

        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

    user_name = str(input("Enter User_name: "))
    folder_name = "user_images"
    os.makedirs(folder_name, exist_ok=True)  # Ensure folder exists
    file_path = os.path.join(folder_name, user_name + ".jpg")
    cv2.imwrite(file_path, cropped_face)

    print("Wait Until we process your image")

    face_info = DeepFace.represent(cropped_face, model_name="ArcFace")
    e1 = face_info[0]["embedding"]
    if len(e1) == 512:
        print("Face Processed Successfully")

    # Save in MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["Face_Recognition"]
    collection = db["embeddings"]
    data = {"user_name": user_name, "embedding": e1}
    collection.insert_one(data)

    print("Face data saved in MongoDB")
    return user_name," Registered Successfully"

if __name__ == "__main__":
    register_new_user()
