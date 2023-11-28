import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from streamlit_webrtc import webrtc_streamer
import webbrowser
import time
from keras.models import model_from_json
#####################################
try:
    gender = np.load("gender.npy")
except:
    gender = ""

#####################################
# from keras_preprocessing.image import load_img
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

webcam=cv2.VideoCapture(0)
labels = {0 : 'sad', 1 : 'sad', 2 : 'sad', 3 : 'happy', 4 : 'sad', 5 : 'sad', 6 : 'happy'}

t_end = time.time() + 5
prediction_label = 'happy'
while time.time() < t_end:
    i,im=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im,1.3,5)
    try: 
        for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            # print("Predicted Output:", prediction_label)
            # cv2.putText(im,prediction_label)
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
        cv2.imshow("Output",im)
        cv2.waitKey(27)
    except cv2.error:
        pass
webcam.release()
st.title("Predicted Label for the image is {}".format(prediction_label))
if(prediction_label=='sad'):
    webbrowser.open(f"https://www.spotify.com")
else:
    webbrowser.open(f"https://www.youtube.com")    

#####################################

# lang = st.text_input("Language")
# # singer = st.text_input("Singer")

# btn = st.button("Recommend me Songs")

# if btn:
#     webbrowser.open(f"https://www.youtube.com")