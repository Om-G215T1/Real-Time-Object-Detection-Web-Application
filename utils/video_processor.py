# utils/video_processor.py
# Real-time video processing pipeline with FPS optimization
# and low-confidence frame saving for self-learning

import cv2
import numpy as np
import os
import time
import torch
from ultralytics import YOLO

from utils.fps_counter import FPSCounter
from utils.draw_boxes import draw_all_detections, draw_info_bar
from utils.config import (
    CONFIDENCE_THRESHOLD,
    LOW_CONF_THRESHOLD,
    UNKNOWN_DETECTIONS_FOLDER,
    FRAME_RESIZE
)

class VideoProcessor:
    def __init__(
        self,
        model_path: str,
        conf: float = CONFIDENCE_THRESHOLD,
        low_conf: float = LOW_CONF_THRESHOLD
    ):
        """
        Initialize the video processor.

        Args:
            model_path : path to YOLOv8 .pt model
            conf       : confidence threshold for display
            low_conf   : below this = save for retraining
        """
        self.conf = conf
        self.low_conf = low_conf
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fps_counter = FPSCounter(buffer_size=30)

        print(f"Loading model: {model_path} on {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print("Model loaded ✅")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame for faster inference.
        Maintains aspect ratio.
        """
        h, w = frame.shape[:2]
        scale = FRAME_RESIZE / max(h, w)
        if scale < 1:  # Only downscale, never upscale
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h),
                               interpolation=cv2.INTER_LINEAR)
        return frame

    def detect(self, frame: np.ndarray):
        """
        Run YOLOv8 inference on a frame.

        Returns:
            YOLOv8 results object
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            verbose=False,
            stream=False
        )
        return results[0]

    def save_low_confidence(self, frame: np.ndarray, result):
        """
        Save frames that have low-confidence detections.
        These are used later for retraining (self-learning).

        Saves if any detection has:
            low_conf_threshold <= confidence < conf_threshold
        """
        if result.boxes is None:
            return

        for box in result.boxes:
            conf_val = float(box.conf[0])
            if self.low_conf <= conf_val < self.conf:
                # Save frame with timestamp as filename
                timestamp = int(time.time() * 1000)
                save_path = os.path.join(
                    UNKNOWN_DETECTIONS_FOLDER,
                    f"low_conf_{timestamp}.jpg"
                )
                cv2.imwrite(save_path, frame)
                break  # Only save once per frame

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Full pipeline for a single frame:
        preprocess → detect → draw → save unknowns → update FPS

        Returns:
            annotated_frame : frame with boxes drawn
            result          : raw YOLOv8 result
            fps             : current FPS
        """
        # Step 1: Preprocess
        small_frame = self.preprocess(frame)

        # Step 2: Detect
        result = self.detect(small_frame)

        # Step 3: Draw detections
        annotated = draw_all_detections(
            small_frame.copy(),
            result,
            self.model.names
        )

        # Step 4: Save low confidence detections
        self.save_low_confidence(small_frame, result)

        # Step 5: Update FPS
        self.fps_counter.update()
        fps = self.fps_counter.get_fps()

        # Step 6: Draw info bar (FPS, object count, device)
        annotated = draw_info_bar(
            annotated,
            fps=fps,
            object_count=len(result.boxes) if result.boxes else 0,
            device=self.device.upper()
        )

        return annotated, result, fps

    def process_webcam(self, camera_id: int = 0):
        """
        Run real-time detection on webcam feed.
        Press Q to quit.
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return

        print("Webcam started. Press Q to quit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, result, fps = self.process_frame(frame)

            cv2.imshow("YOLOv8 Real-Time Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nAverage FPS: {self.fps_counter.get_fps():.1f}")

    def process_video_file(self, video_path: str, output_path: str = None):
        """
        Run detection on a video file frame by frame.

        Args:
            video_path  : input video path
            output_path : save annotated video here (optional)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return

        # Setup video writer if output path given
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps_out = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps_out, (w, h))

        frame_count = 0
        print(f"Processing video: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, result, fps = self.process_frame(frame)
            frame_count += 1

            if writer:
                # Resize back to original before saving
                out_frame = cv2.resize(
                    annotated,
                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                )
                writer.write(out_frame)

            cv2.imshow("YOLOv8 Video Processing", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print(f"\n✅ Video processing complete")
        print(f"   Frames processed : {frame_count}")
        print(f"   Average FPS      : {self.fps_counter.get_fps():.1f}")
        if output_path:
            print(f"   Saved to         : {output_path}")

    def generate_frames(self, camera_id: int = 0):
        """
        Generator function for Flask MJPEG streaming.
        Yields JPEG bytes for each annotated frame.
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, result, fps = self.process_frame(frame)

            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', annotated,
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Yield as multipart frame for Flask streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + buffer.tobytes()
                   + b'\r\n')

        cap.release()