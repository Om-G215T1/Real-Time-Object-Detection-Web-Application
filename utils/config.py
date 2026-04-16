# utils/config.py
# Central configuration file for the entire project

import os
from dotenv import load_dotenv

load_dotenv()  # Loads from .env file

# ─── Model ────────────────────────────────────────
MODEL_PATH = os.getenv('MODEL_PATH', 'yolov8m.pt')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
LOW_CONF_THRESHOLD = float(os.getenv('LOW_CONF_THRESHOLD', 0.25))

# ─── Camera ───────────────────────────────────────
CAMERA_ID = int(os.getenv('CAMERA_ID', 0))
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# ─── Flask ────────────────────────────────────────
FLASK_HOST = '0.0.0.0'
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
FLASK_DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'

# ─── Paths ────────────────────────────────────────
UPLOAD_FOLDER = 'uploads'
UNKNOWN_DETECTIONS_FOLDER = 'unknown_detections/low_conf_frames'
RETRAIN_QUEUE_FOLDER = 'unknown_detections/retrain_queue'

# Create folders if they don't exist
for folder in [UPLOAD_FOLDER, UNKNOWN_DETECTIONS_FOLDER, RETRAIN_QUEUE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ─── Processing ───────────────────────────────────
MAX_UPLOAD_SIZE_MB = 100
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
FRAME_RESIZE = 640      # Resize frames to this before inference