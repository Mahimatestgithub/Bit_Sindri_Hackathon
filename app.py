import streamlit as st
import cv2
import numpy as np
import pytesseract
import os
import tempfile
from PIL import Image
try:
    from keras.src.saving.saving_api import load_model
except ImportError:
    from keras.saving import load_model
from ultralytics import YOLO

# Configure the Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'D:/Program Files/Tesseract-OCR/tesseract.exe'

# Load models
object_model = YOLO("yolov8s.pt")
logo_model = YOLO("logo.pt")
try:
    fresh_model = load_model('rottenvsfresh.h5')
except Exception as e:
    st.error(f"Failed to load fresh model: {e}")
    fresh_model = None

# Load class names
if object_model:
    object_class_names = object_model.names
if logo_model:
    logo_class_names = logo_model.names

def print_fresh(res):
    if res < 0.10:
        return "FRESH", (0, 255, 0)  # Green
    elif 0.10 <= res < 0.35:
        return "MEDIUM FRESH", (0, 255, 255)  # Yellow
    else:
        return "NOT FRESH", (0, 0, 255)  # Red

def pre_proc_img(image):
    img = cv2.resize(image, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def evaluate_rotten_vs_fresh(image):
    if fresh_model is None:
        return 0.5  # Return neutral value if model not loaded
    try:
        prediction = fresh_model.predict(pre_proc_img(image))
        return prediction[0][0]
    except Exception as e:
        st.error(f"Error in freshness prediction: {e}")
        return 0.5

def draw_boxes(frame, results, class_names, color):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if confidence > 0.5:
                x1, y1, x2, y2 = map(int, box)
                label = f'{class_names[int(class_id)]}: {confidence:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if class_names[int(class_id)] in ['apple', 'banana', 'orange']:
                    crop_img = frame[y1:y2, x1:x2]
                    freshness_score = evaluate_rotten_vs_fresh(crop_img)
                    freshness_label, freshness_color = print_fresh(freshness_score)
                    cv2.putText(frame, freshness_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, freshness_color, 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), freshness_color, 2)
    return frame

def perform_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray).strip()
    st.write(f"Text: {text}")  # Debugging information
    return text

def validate_file_path(file_path):
    """Validate file path to prevent path traversal attacks"""
    if not file_path:
        return False
    # Check if path is absolute or contains directory traversal
    if os.path.isabs(file_path) or '..' in file_path or ':' in file_path:
        return False
    # Check if file exists and is in allowed directory
    allowed_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(allowed_dir, file_path)
    return os.path.exists(full_path) and os.path.commonpath([allowed_dir, full_path]) == allowed_dir

def main():
    st.title("Smart Quality Vision Testing System")

    # Add mode selection
    mode = st.radio("Select Mode", ("Image Upload", "Live Camera Feed"))

    if mode == "Image Upload":
        st.write("Upload an image to analyze.")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Object detection
            object_results = object_model(frame)
            frame_with_objects = draw_boxes(frame.copy(), object_results, object_class_names, (255, 255, 255))

            # Logo detection
            logo_results = logo_model(frame)
            frame_with_logos = draw_boxes(frame_with_objects.copy(), logo_results, logo_class_names, (255, 0, 0))

            # OCR
            detected_text = perform_ocr(frame_with_logos)
            st.subheader("Detected Text (OCR)")
            st.write(detected_text)

            st.image(cv2.cvtColor(frame_with_logos, cv2.COLOR_BGR2RGB), caption='Processed Image', use_container_width=True)

    elif mode == "Live Camera Feed":
        st.write("Analyzing live camera feed. Press 'Q' to exit.")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        stframe = st.empty()  # Streamlit frame for displaying the video

        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Unable to access the camera. Check your device.")
                break

            # Object detection
            object_results = object_model(frame)
            frame_with_objects = draw_boxes(frame.copy(), object_results, object_class_names, (0, 0, 0))

            # Logo detection
            logo_results = logo_model(frame)
            frame_with_logos = draw_boxes(frame_with_objects.copy(), logo_results, logo_class_names, (255, 0, 0))

            # OCR
            detected_text = perform_ocr(frame_with_logos)
            stframe.text(f"Detected Text: {detected_text}")

            # Display video frame
            stframe.image(cv2.cvtColor(frame_with_logos, cv2.COLOR_BGR2RGB), use_container_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()