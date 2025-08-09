import cv2
import os
import numpy as np

face_dector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



# actors_name = []

# for i in os.listdir("images"):
#     actors_name.append(i)


actors_name = ['Angelina Jolie', 'Brad Pitt', 'Denzel Washington', 'Hugh Jackman', 'Jennifer Lawrence', 'Johnny Depp', 'Kate Winslet', 'Leonardo DiCaprio', 'Megan Fox', 'Natalie Portman', 'Nicole Kidman', 'Robert Downey Jr', 'Sandra Bullock', 'Scarlett Johansson', 'Tom Cruise', 'Tom Hanks', 'Will Smith']

path = "images"

lables = []
actor_faces =[]

for actors in actors_name:
    actor_folder = os.path.join(path, actors)
    actor_index = actors_name.index(actors)


    for images in os.listdir(actor_folder):
        actor_image_path = os.path.join(actor_folder, images)
        array_img = cv2.imread(actor_image_path)
        gray_img = cv2.cvtColor(array_img, cv2.COLOR_BGR2GRAY)


        face_roi = face_dector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for x, y, w, h in face_roi:
            # cv2.rectangle(array_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            crop_face = gray_img[y:y + h, x:x + w] 

        # cv2.imshow("img", crop_face)

        # cv2.waitKey(0)

        lables.append(actor_index)
        actor_faces.append(crop_face)


lables_array = np.array(lables)
actor_faces_array = np.array(actor_faces, dtype='object')


# Model ---> Algo

model = cv2.face.LBPHFaceRecognizer_create()

model.train(actor_faces_array, lables_array)

model.save("face_recognization_system.yml")