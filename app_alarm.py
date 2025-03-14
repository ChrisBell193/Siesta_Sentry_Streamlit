from pathlib import Path
import streamlit as st
from ultralytics import YOLO
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam, play_webcam, play_webcam_alarm
#from app.utils import infer_uploaded_image, infer_uploaded_video, play_webcam_alarm
from PIL import Image
import torch
from ultralytics.nn.tasks import DetectionModel

# setting page layout
# st.set_page_config(
#     page_title="Interactive Interface for YOLOv8",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
#     )

st.set_page_config(
    page_title="Siesta Sentry",
    page_icon=":male-detective:",
    # layout="wide",
    initial_sidebar_state="expanded"
)

# Load the image for the title
title_image = Image.open("SIESTA8LOGO.png")

# Display the title image
st.image(title_image, use_column_width=True)

# sidebar
st.sidebar.header("Model Config")

model = YOLO('yolov8s.pt')
model = YOLO('best.pt')
#model.model = torch.load('yolov8s.pt', weights_only=False)
#model.model = torch.load('best.pt', weights_only=False)

# model = torch.load('yolov8s.pt')

# image/video options
st.sidebar.header("Input Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    ["Webcam", "Image", "Video"]
)

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

source_img = None
if source_selectbox == "Image": # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == "Video": # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == "Webcam": # Webcam
    play_webcam_alarm(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")
