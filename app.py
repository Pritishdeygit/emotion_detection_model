import streamlit as st
import cv2
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load model
model = load_model("emotion_model.keras")

# Emotion labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# App title
st.title("Emotion Detection using ResNet50")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")

    # Convert image to numpy
    image_np = np.array(image)

    # Convert RGB to BGR
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) == 0:
        st.error("No face detected")

    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(
            img_bgr,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Crop face
        face = image_np[y:y+h, x:x+w]

        # Resize to 224x224
        face = cv2.resize(face, (224, 224))

        # Convert to float32
        face = face.astype(np.float32)

        # Expand dimensions
        face = np.expand_dims(face, axis=0)

        # Preprocess for ResNet50
        face = preprocess_input(face)

        # Predict
        prediction = model.predict(face)

        # Get prediction index
        emotion_index = np.argmax(prediction)

        # Get emotion label
        emotion = emotion_labels[emotion_index]

        # Confidence
        confidence = np.max(prediction) * 100

        # Put text on image
        cv2.putText(
            img_bgr,
            f"{emotion} ({confidence:.2f}%)",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        # Show prediction
        st.success(f"Emotion: {emotion}")
        st.info(f"Confidence: {confidence:.2f}%")

    # Convert back to RGB
    final_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Display final image
    st.image(
        final_img,
        caption="Prediction Result",
        use_container_width=True
    )