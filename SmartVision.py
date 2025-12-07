from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import os
import numpy as np
import pytesseract
from ultralytics import YOLO
import base64
import logging
from werkzeug.utils import secure_filename, tempfile

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

pytesseract.pytesseract.tesseract_cmd = r'D:/Program Files/Tesseract-OCR/tesseract.exe'

object_model = YOLO("yolov8s.pt")
object_class_names = object_model.names

if not os.path.exists("yolov8s.pt"):
    raise FileNotFoundError("YOLOv8 model file not found")

def pre_proc_img(image):
    img = cv2.resize(image, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def draw_boxes(frame, results, class_names, color):
    h, w, _ = frame.shape
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
                    
    return frame

def perform_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray).strip()
    return text

def overlay_text(frame, text):
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.rectangle(frame, (10, 30 - text_height - baseline), (10 + text_width, 30 + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return frame

@app.before_request
def security_checks():
    logger.info(f"Request: {request.method} {request.url}")

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        file = request.files['frame']
        if file.filename == '':
            return jsonify({'error': 'No frame selected'}), 400
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            tmp_filename = tmp_file.name
        
        try:
            frame = cv2.imread(tmp_filename)
            if frame is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            results = object_model(frame)
            frame = draw_boxes(frame, results, object_class_names, (0, 255, 0))
            text = perform_ocr(frame)
            frame = overlay_text(frame, text)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({'frame': frame_base64, 'text': text})
        finally:
            os.unlink(tmp_filename)
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.after_request
def after_request(response):
    response.headers.add('X-Content-Type-Options', 'nosniff')
    response.headers.add('X-Frame-Options', 'DENY')
    response.headers.add('X-XSS-Protection', '1; mode=block')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)