from deepface import DeepFace
import cv2
import mtcnn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs
from deepface import DeepFace
print("initiating")

# Connection to mongo db
from pymongo import MongoClient
client=MongoClient("mongodb://localhost:27017")
db=client["Face_Recognition"]
collection=db["embeddings"]

print("connection successfull")

def recognize_face():
    cap=cv2.VideoCapture(0)
    detector=mtcnn.MTCNN()
    while True:
        ret,frame=cap.read()
        face=detector.detect_faces(frame)
        if len(face)>0:
            x,y,w,h=face[0]['box']
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            cv2.imshow("frame",frame)
            cropped_face=frame[y:y+h,x:x+h]
            
        if cv2.waitKey(1)==13:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    
    print("verifying")
    # create embeddings
    
    face_info=DeepFace.represent(cropped_face,model_name="ArcFace")
    face_embedding=face_info[0]["embedding"]

    # searching for high similarity in mongo db embeddings
   

    records=collection.find()
    high_similarity=-1
    verified=None
    for record in records:
        e1=record["embedding"]
        similarity=cs(np.array(face_embedding).reshape(1,-1),np.array(e1).reshape(1,-1))
        if similarity[0][0]>high_similarity:
            high_similarity=similarity
            verified=record["user_name"]
        

    if high_similarity[0][0]<0.5:
        print("no match found")
    else:
        print("face recognized :",verified)
        print("similarity :",high_similarity[0][0])
    return verified, high_similarity[0][0]
    
    
if __name__=="__main__":
    recognize_face()