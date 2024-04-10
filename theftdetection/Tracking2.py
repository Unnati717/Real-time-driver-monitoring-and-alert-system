import cv2
import numpy as np
import streamlit as st
import pygame
import os
import face_recognition as frg
import yaml
from utils import recognize, build_dataset

# Initialize pygame mixer
pygame.mixer.init()

# Path: code\app.py

st.set_page_config(layout="wide")
# Config
cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
PICTURE_PROMPT = cfg['INFO']['PICTURE_PROMPT']
WEBCAM_PROMPT = cfg['INFO']['WEBCAM_PROMPT']

st.sidebar.title("Settings")

# Create a menu bar
menu = ["Picture", "Webcam"]
choice = st.sidebar.selectbox("Input type", menu)
# Put slide to adjust tolerance
TOLERANCE = st.sidebar.slider("Tolerance", 0.0, 1.0, 0.5, 0.01)
st.sidebar.info(
    "Tolerance is the threshold for face recognition. The lower the tolerance, the more strict the face recognition. The higher the tolerance, the more loose the face recognition."
)

# Information section
st.sidebar.title("Driver Information")
name_container = st.sidebar.empty()
id_container = st.sidebar.empty()
name_container.info('Name: Unknown')
id_container.success('ID: Unknown')

# Initialize alarm state
alarm_playing = False

if choice == "Picture":
    st.title("Theft Detection App")
    st.write(PICTURE_PROMPT)
    uploaded_images = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    if len(uploaded_images) != 0:
        # Read uploaded image with face_recognition
        for image in uploaded_images:
            image = frg.load_image_file(image)
            image, name, id = recognize(image, TOLERANCE)
            name_container.info(f"Name: {name}")
            id_container.success(f"ID: {id}")
            st.image(image)
            if name == 'Unknown':
                # Play the alarm sound only once when an unknown person is detected
                if not alarm_playing:
                    pygame.mixer.music.load('alarm.wav')
                    pygame.mixer.music.play()
                    alarm_playing = True
            else:
                # Reset the alarm state when a known person is detected
                alarm_playing = False
    else:
        st.info("Please upload an image")

elif choice == "Webcam":
    st.title("Theft Detection App")
    st.write(WEBCAM_PROMPT)
    # Camera Settings
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    FRAME_WINDOW = st.image([])

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to capture frame from the camera")
            st.info("Please turn off the other app that is using the camera and restart the app")
            st.stop()
        image, name, id = recognize(frame, TOLERANCE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Display name and ID of the person

        name_container.info(f"Name: {name}")
        id_container.success(f"ID: {id}")

        if name == 'Unknown':
            # Play the alarm sound only once when an unknown person is detected
            if not alarm_playing:
                pygame.mixer.music.load('alarm.wav')
                pygame.mixer.music.play()
                alarm_playing = True
        else:
            # Reset the alarm state when a known person is detected
            alarm_playing = False

        FRAME_WINDOW.image(image)

with st.sidebar.form(key='my_form'):
    st.title("Developer Section")
    submit_button = st.form_submit_button(label='REBUILD DATASET')
    if submit_button:
        with st.spinner("Rebuilding dataset..."):
            build_dataset()
        st.success("Dataset has been reset")
