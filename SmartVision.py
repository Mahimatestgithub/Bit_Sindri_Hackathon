from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import base64
import logging
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Security configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure Tesseract OCR - Use environment variable for security
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_PATH', '/usr/bin/tesseract')

# Load YOLOv8 models
object_model = YOLO("yolov8s.pt")

# Load class names for object detection
object_class_names = object_model.names

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_data(img_data):
    # Basic validation to prevent malicious data
    if len(img_data) == 0 or len(img_data) > 16 * 1024 * 1024:  # 16MB limit
        return False
    return True

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

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        file = request.files['frame']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read image data
        img_data = file.read()
        if not validate_image_data(img_data):
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Process frame
        results = object_model(frame)
        frame = draw_boxes(frame, results, object_class_names, (0, 255, 0))
        text = perform_ocr(frame)
        frame = overlay_text(frame, text)
        
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = jsonify({
            'frame': frame_base64,
            'text': text
        })
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        return response
    except Exception as e:
        logging.error(f"Error processing frame: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)