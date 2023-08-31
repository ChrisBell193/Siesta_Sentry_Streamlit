import threading

import cv2
import streamlit as st
from matplotlib import pyplot as plt
from utils import load_model, infer_uploaded_webcam
# from streamlit import script_run_context


from streamlit_webrtc import webrtc_streamer


# webrtc_streamer(key="sample")


##########################

from ultralytics import YOLO
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
from PIL import Image
import av
import cv2
from pytube import YouTube

# import settings



def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model




def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLO object detection model.

    Returns:
        None

    Raises:
        None
    """
    st.sidebar.title("Webcam Object Detection")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")


        orig_h, orig_w = image.shape[0:2]
        width = 720  # Set the desired width for processing

        # cv2.resize used in a forked thread may cause memory leaks
        processed_image = np.asarray(Image.fromarray(image).resize((width, int(width * orig_h / orig_w))))

        if model is not None:
            # Perform object detection using YOLO model
            res = model.predict(processed_image, conf=conf)
            # print(f'resboxes: {res.boxes}')

            # Plot the detected objects on the video frame
            res_plotted = res[0].plot()
            # print(f'resplotted: {res_plotted}')


        return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")


    webrtc_streamer(
        key="example",
        # video_transformer_factory=lambda: MyVideoTransformer(conf, model),
        video_frame_callback = video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

# class MyVideoTransformer(VideoTransformerBase):
#     def __init__(self, conf, model):
#         self.conf = conf
#         self.model = model

#     def recv(self, frame):
#         image = frame.to_ndarray(format="bgr24")
#         processed_image = self._display_detected_frames(image)
#         st.image(processed_image, caption='Detected Video', channels="BGR", use_column_width=True)

#     def _display_detected_frames(self, image):
#         orig_h, orig_w = image.shape[0:2]
#         width = 720  # Set the desired width for processing

#         # cv2.resize used in a forked thread may cause memory leaks
#         input = np.asarray(Image.fromarray(image).resize((width, int(width * orig_h / orig_w))))

#         if self.model is not None:
#             # Perform object detection using YOLO model
#             res = self.model.predict(input, conf=self.conf)
#             # print(f'resboxes: {res.boxes}')

#             # Plot the detected objects on the video frame
#             res_plotted = res[0].plot()
#             return res_plotted

#         return input

conf = .2
# model = load_model('should be best fine tuned/weights/best.pt')
model = load_model('should be best fine tuned/weights/best.pt')

play_webcam(conf, model)
