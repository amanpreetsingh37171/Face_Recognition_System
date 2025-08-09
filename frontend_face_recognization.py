import cv2
import os
import numpy as np

face_dector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

loading_model = cv2.face.LBPHFaceRecognizer_create()
loading_model.read("face_recognization_system.yml")


actors_name = ['Angelina Jolie', 'Brad Pitt', 'Denzel Washington', 'Hugh Jackman', 'Jennifer Lawrence', 'Johnny Depp', 'Kate Winslet', 'Leonardo DiCaprio', 'Megan Fox', 'Natalie Portman', 'Nicole Kidman', 'Robert Downey Jr', 'Sandra Bullock', 'Scarlett Johansson', 'Tom Cruise', 'Tom Hanks', 'Will Smith']


cap = cv2.VideoCapture(0)

while True:
    isTrue, frame = cap.read()

    gray_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    face_roi = face_dector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in face_roi:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    






        crop_face = gray_face[y:y + h, x:x + w]
        crop_face = cv2.resize(crop_face, (200, 200))
        label, confidence = loading_model.predict(crop_face)
          


    
        cv2.putText(frame, f"{actors_name[label]}, confidence: {confidence}/-100", (x - 30, y - 30), cv2.FONT_HERSHEY_PLAIN, 2, (20, 240, 34), 3)




    cv2.imshow("Face Recognition System", frame)

    if cv2.waitKey(20) & 0xFF == ord('d'):

        break
