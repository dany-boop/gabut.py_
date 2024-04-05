import cv2
import numpy as np
from deepface import DeepFace


face_dtct = cv2.CascadeClassifier('face_advnce.xml')
cam = cv2.VideoCapture(0)

def detect_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_dtct.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces, gray_frame

def main():
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        faces, gray_frame = detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_img = gray_frame[y:y + h, x:x + w]
            if face_img.size != 0:
               
                face_img_resized = cv2.resize(face_img, (48, 48))
                face_img_rgb = cv2.cvtColor(face_img_resized, cv2.COLOR_GRAY2RGB)
                results = DeepFace.analyze(face_img_rgb, actions=['emotion', 'age'], enforce_detection=False)
                if isinstance(results, list) and results:
                    result = results[0]
                
                age = int(result['age'])
                emotion = result['dominant_emotion']
                
                cv2.putText(frame, f'Age: {age}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Emotion: {emotion}', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    cam.release()
    cv2.destroyAllWindows()  
    

if __name__  == '__main__':
    main()



