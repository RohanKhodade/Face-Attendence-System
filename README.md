# 🎥 Face Recognition Attendance System  

This is a **Face Recognition System with Attendance Marking** built using **Python, OpenCV, DeepFace, MTCNN, Tkinter, and MongoDB**.  
It allows you to:  
- Register new users by capturing their face and storing embeddings in MongoDB.  
- Recognize registered users in real-time via webcam.  
- Mark attendance automatically.  

---

## 🚀 Features  

- Live webcam feed inside a Tkinter GUI  
- New user registration with name and face embedding storage  
- Face recognition using **DeepFace (ArcFace model)**  
- Face detection with **MTCNN**  
- Similarity comparison with **Cosine Similarity**  
- MongoDB database integration for embeddings  
- Attendance marking with confidence score  

---

## 🛠️ Tech Stack  

- **Python**  
- **OpenCV** – Webcam feed & face image processing  
- **MTCNN** – Face detection  
- **DeepFace (ArcFace)** – Face embeddings  
- **MongoDB** – Database for embeddings & user info  
- **Tkinter** – GUI interface  
- **NumPy / Scikit-learn** – Cosine similarity  

---

## 📂 Project Structure  

📦 Face_Recognition_System
┣ 📜 main.py # Main Python script (GUI + Logic)
┣ 📂 user_images # Folder to store registered user face images
┣ 📜 README.md # Documentation


---

## ⚙️ Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition-attendance

## ⚙️ Installation  
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

install requirements.txt
tk
pillow
opencv-python
mtcnn
deepface
numpy
scikit-learn
pymongo

