"""
Created on 26 Aug 2021
@author: akshatrunwal

"""
# Extracts all faces from the image frame
# Stores the face information into numpy array

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box
# 3.Flatten the largest face image and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

import cv2
import numpy as np

#Init web Cam
cap = cv2.VideoCapture(0)

#Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
dataset_path = "./data/"
file_name = input("Enter the name of person : ")
skip=0;

while True:
    ret, frame = cap.read()
    
    if ret == False:
        continue
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces = sorted(faces, key=lambda f:f[2]*f[3])
   
    #pick the last face (because it has largest area)
    face_section_list = [] 
    for face in faces[-1:]:
        #draw bounding box or the rectange
        
        x,y,w,h = face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (123, 88, 202), 2)
        
        #extract (crop out the required face): region of interest
        
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))
        face_section_list.append(face_section)
        
    # Store every 10th face.
        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_section))
    
    cv2.imshow("Frame", frame)
    for im in face_section_list:
        cv2.imshow("Face section frame",im)
         # Zero means "wait until a key is pressed"
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

#convert face data list into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("data saved successfully!!!")

cap.release()
cv2.destroyAllWindows()