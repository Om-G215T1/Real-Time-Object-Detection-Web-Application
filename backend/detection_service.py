# backend/detection_service.py
# Core detection logic — used by all Flask routes
# Singleton pattern ensures model is loaded only once

import cv2
import numpy as np
import base64
import torch
from ultralytics import YOLO

from utils.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    LOW_CONF_THRESHOLD,
    UNKNOWN_DETECTIONS_FOLDER
)
from utils.draw_boxes import draw_all_detections, draw_info_bar
from utils.fps_counter import FPSCounter
import time
import os


class DetectionService:
    _instance = None  # Singleton instance

    def __new__(cls, model_path=MODEL_PATH):
        """Ensure only one instance of model is loaded."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path=MODEL_PATH):
        if self._initialized:
            return  # Already initialized — skip

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.fps_counter = FPSCounter(buffer_size=30)

        print(f"\nLoading detection model...")
        print(f"  Path   : {model_path}")
        print(f"  Device : {self.device}")

        self.model = YOLO(model_path)
        self.model.to(self.device)
        self._initialized = True

        print("Detection service ready ✅\n")

    def switch_model(self, model_path: str):
        """
        Switch to a different model at runtime.
        Supports YOLOv8 / SSD switching.
        """
        print(f"Switching model to: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model_path = model_path
        print("Model switched ✅")

    def _save_low_confidence(self, frame: np.ndarray, result):
        """Save low-confidence frames for future retraining."""
        if result.boxes is None:
            return
        for box in result.boxes:
            conf_val = float(box.conf[0])
            if LOW_CONF_THRESHOLD <= conf_val < CONFIDENCE_THRESHOLD:
                timestamp = int(time.time() * 1000)
                path = os.path.join(
                    UNKNOWN_DETECTIONS_FOLDER,
                    f"low_conf_{timestamp}.jpg"
                )
                cv2.imwrite(path, frame)
                break

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert OpenCV frame to base64 string for JSON response."""
        _, buffer = cv2.imencode(
            '.jpg', frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        return base64.b64encode(buffer).decode('utf-8')

    def _extract_detections(self, result) -> list:
        """Extract detection data into a clean list of dicts."""
        detections = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            detections.append({
                'class': self.model.names[int(box.cls[0])],
                'confidence': round(float(box.conf[0]), 3),
                'bbox': box.xyxy[0].tolist()
            })
        return detections

    def detect_image(
        self,
        image: np.ndarray,
        conf: float = CONFIDENCE_THRESHOLD
    ) -> dict:
        """
        Run detection on a single image.

        Args:
            image : OpenCV BGR image (numpy array)
            conf  : confidence threshold

        Returns:
            dict with base64 annotated image and detections list
        """
        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf,
            verbose=False
        )
        result = results[0]

        # Draw detections
        annotated = draw_all_detections(
            image.copy(),
            result,
            self.model.names
        )

        # Update FPS
        self.fps_counter.update()
        fps = self.fps_counter.get_fps()

        # Draw info bar
        annotated = draw_info_bar(
            annotated,
            fps=fps,
            object_count=len(result.boxes) if result.boxes else 0,
            device=self.device.upper()
        )

        # Save low confidence frames
        self._save_low_confidence(image, result)

        return {
            'image': self._frame_to_base64(annotated),
            'detections': self._extract_detections(result),
            'count': len(result.boxes) if result.boxes else 0,
            'fps': round(fps, 1),
            'device': self.device
        }

    def generate_webcam_frames(
        self,
        camera_id: int = 0,
        conf: float = CONFIDENCE_THRESHOLD
    ):
        """
        Generator for real-time webcam MJPEG stream.
        Used by Flask streaming route.
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result_data = self.detect_image(frame, conf=conf)

                # Decode base64 back to frame for streaming
                img_bytes = base64.b64decode(result_data['image'])
                img_array = np.frombuffer(img_bytes, np.uint8)
                annotated = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                _, buffer = cv2.imencode(
                    '.jpg', annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 85]
                )

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + buffer.tobytes()
                       + b'\r\n')
        finally:
            cap.release()

    def generate_video_frames(
        self,
        video_path: str,
        conf: float = CONFIDENCE_THRESHOLD
    ):
        """
        Generator for video file MJPEG stream.
        Used by Flask video upload route.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result_data = self.detect_image(frame, conf=conf)

                img_bytes = base64.b64decode(result_data['image'])
                img_array = np.frombuffer(img_bytes, np.uint8)
                annotated = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                _, buffer = cv2.imencode(
                    '.jpg', annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 85]
                )

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + buffer.tobytes()
                       + b'\r\n')
        finally:
            cap.release()