import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import requests
import base64
import time

st.title("Smart Quality Vision Testing System")
st.sidebar.title("Settings")

camera = cv2.VideoCapture(0)

# Configure timeout and security settings
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3

def check_food_quality(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    try:
        response = requests.post('http://localhost:5000/process_frame', 
                                files={'frame': img_encoded.tobytes()},
                                timeout=REQUEST_TIMEOUT,
                                headers={'Content-Type': 'multipart/form-data'})
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        img_base64 = data['frame']
        img_bytes = base64.b64decode(img_base64)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        quality = data['text']
        return frame, quality
    except requests.exceptions.Timeout:
        st.error("Request timeout. Please check your connection.")
        return frame, "Timeout Error"
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the backend: {e}")
        return frame, "Error"
    except ValueError as e: