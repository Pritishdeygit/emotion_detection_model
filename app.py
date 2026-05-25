import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

labels = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@st.cache_resource
def get_model():
    return load_model("FER_best_model.keras", compile=False)

model = get_model()

st.title("Emotion Detection")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (224,224))
        face = np.expand_dims(face.astype(np.float32), axis=0)
        face = preprocess_input(face)

        pred = model.predict(face)
        emotion = labels[np.argmax(pred)]
        conf = np.max(pred)*100

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,f"{emotion} ({conf:.1f}%)",(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    st.image(img)
