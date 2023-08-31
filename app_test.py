import streamlit as st
import numpy as np
import base64



# Load the alert sound
alert_sound_path = "alarm.wav"
audio_file = open("alarm.wav", 'rb')
audio_bytes = audio_file.read()

st.audio(audio_bytes, format='audio/wav', start_time= 5)





def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


st.write("# Auto-playing Audio!")

autoplay_audio(alert_sound_path)
