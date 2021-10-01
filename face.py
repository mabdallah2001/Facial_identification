from posixpath import join
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
from openpyxl import load_workbook

wb = load_workbook ('FacialAttribs.xlsx')
ws = wb.active
ws.title = 'Sheet1'


def analyzeGender(frame):
    gender = cv2.dnn.readNetFromCaffe("gender.prototxt","gender.caffemodel")
    # img = cv2.imread(frame)
    plt.imshow(frame[:,:,::-1])

    detector_path  = "./venv/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(detector_path)

    faces = detector.detectMultiScale(frame, 1.3, 5)

    x,y,w,h = faces[0]

    detected_face = frame[int(y):int(y+h), int(x):int(x+w)]

    plt.imshow(detected_face[:,:,::-1])

    detected_face = cv2.resize(detected_face, (224,224))

    detected_face_blob = cv2.dnn.blobFromImage(detected_face)
    

    gender.setInput(detected_face_blob)
    gender_result = gender.forward()

    if np.argmax(gender_result[0]) == 0:
        gendertype = "female"

    else:
        gendertype = "male"
    
    return gendertype



def analyzeEmotion(frame):
    result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection=False)
    em = result['dominant_emotion']

   
    return em






faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    raise IOError("Cannot access webcam")


gender = ""
count = 10
count2 = 99
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
   
    ret,frame = cap.read()
    
    if(count == 10):
        em = analyzeEmotion(frame)
        count = 0
    if(count2 == 100):
        gender = analyzeGender(frame)
        print(gender)
        count2 = 0
    
    cv2.putText(frame, em, (50,50), font, 3, (0,0,255),2,cv2.LINE_4)
    cv2.putText(frame, gender, (50,100), font, 3, (0,0,255),2,cv2.LINE_4)

    cv2.imshow('Original video', frame)
    count +=1
    count2+=1
   

    if cv2.waitKey(20) &  0xFF == ord('q'):
        break

    ws.append([gender,em])

wb.save('FacialAttribs.xlsx')

print("Successfully printed to excel sheet")


cap.release()
cv2.destroyAllWindows()