# backend/routes.py
# All Flask API route definitions

import os
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, Response

from backend.detection_service import DetectionService
from utils.config import (
    UPLOAD_FOLDER,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    MODEL_PATH,
    CAMERA_ID
)

# Create blueprint
api = Blueprint('api', __name__)

# Initialize detection service (singleton)
service = DetectionService(model_path=MODEL_PATH)


# ─── Helpers ──────────────────────────────────────────────

def allowed_file(filename: str, allowed_set: set) -> bool:
    """Check if file extension is allowed."""
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in allowed_set
    )

def get_conf_from_request() -> float:
    """Extract confidence threshold from request args or form."""
    try:
        return float(
            request.args.get('conf') or
            request.form.get('conf') or
            0.5
        )
    except ValueError:
        return 0.5


# ─── Health Check ─────────────────────────────────────────

@api.route('/health', methods=['GET'])
def health():
    """Check if API is running."""
    return jsonify({
        'status': 'ok',
        'device': service.device,
        'model': service.model_path
    })


# ─── Image Detection ──────────────────────────────────────

@api.route('/detect/image', methods=['POST'])
def detect_image():
    """
    Upload an image and get back:
    - base64 annotated image
    - list of detections (class, confidence, bbox)
    - object count
    - FPS
    """
    # Validate file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({
            'error': f'Invalid file type. Allowed: {ALLOWED_IMAGE_EXTENSIONS}'
        }), 400

    # Read image
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Could not decode image'}), 400

    # Run detection
    conf = get_conf_from_request()
    result = service.detect_image(image, conf=conf)

    return jsonify(result)


# ─── Video Upload Detection ────────────────────────────────

@api.route('/detect/video', methods=['POST'])
def detect_video():
    """
    Upload a video and stream back annotated frames
    as MJPEG stream.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({
            'error': f'Invalid file type. Allowed: {ALLOWED_VIDEO_EXTENSIONS}'
        }), 400

    # Save uploaded video
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    conf = get_conf_from_request()

    return Response(
        service.generate_video_frames(save_path, conf=conf),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ─── Webcam Stream ─────────────────────────────────────────

@api.route('/stream/webcam')
def webcam_stream():
    """
    Stream live annotated webcam frames as MJPEG.
    Access directly in <img src="/api/stream/webcam">
    """
    conf = get_conf_from_request()

    return Response(
        service.generate_webcam_frames(
            camera_id=CAMERA_ID,
            conf=conf
        ),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ─── Model Switching ──────────────────────────────────────

@api.route('/model/switch', methods=['POST'])
def switch_model():
    """
    Switch between models at runtime.
    Body: { "model": "yolov8m.pt" } or { "model": "yolov8n.pt" }
    """
    data = request.get_json()

    if not data or 'model' not in data:
        return jsonify({'error': 'Provide model path in JSON body'}), 400

    model_path = data['model']

    if not os.path.exists(model_path) and model_path not in [
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt',
        'yolov8l.pt', 'yolov8x.pt'
    ]:
        return jsonify({'error': f'Model not found: {model_path}'}), 404

    service.switch_model(model_path)
    return jsonify({
        'status': 'switched',
        'model': model_path
    })


# ─── Get Available Models ─────────────────────────────────

@api.route('/model/list', methods=['GET'])
def list_models():
    """Return list of available models."""
    pretrained = [
        'yolov8n.pt', 'yolov8s.pt',
        'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
    ]
    # Also check for locally trained models
    trained = []
    runs_path = 'model/runs'
    if os.path.exists(runs_path):
        for root, dirs, files in os.walk(runs_path):
            for f in files:
                if f.endswith('.pt'):
                    trained.append(os.path.join(root, f))

    return jsonify({
        'pretrained': pretrained,
        'trained': trained,
        'current': service.model_path
    })


# ─── Unknown Detections ───────────────────────────────────

@api.route('/detections/unknown', methods=['GET'])
def list_unknown_detections():
    """List saved low-confidence frames for retraining."""
    from utils.config import UNKNOWN_DETECTIONS_FOLDER
    files = []
    if os.path.exists(UNKNOWN_DETECTIONS_FOLDER):
        files = os.listdir(UNKNOWN_DETECTIONS_FOLDER)

    return jsonify({
        'count': len(files),
        'files': files
    })