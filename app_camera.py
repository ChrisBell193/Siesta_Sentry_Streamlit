from pathlib import Path
import streamlit as st
from ultralytics import YOLO
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam


# setting page layout
st.set_page_config(
    page_title="Interactive Interface for YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )
# main page heading
st.title("Interactive Interface for YOLOv8")

# sidebar
st.sidebar.header("DL Model Config")
model = YOLO('yolov8s.pt')
model = YOLO('best.pt')

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    ["Image", "Video", "Webcam"]
)

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

source_img = None
if source_selectbox == "Image": # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == "Video": # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == "Webcam": # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")
