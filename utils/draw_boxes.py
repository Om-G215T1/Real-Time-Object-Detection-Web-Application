# utils/draw_boxes.py
# Draws bounding boxes, labels, and confidence scores on frames

import cv2
import numpy as np

# Color palette for 80 COCO classes
# Each class gets a unique consistent color
COLORS = np.random.default_rng(42).uniform(0, 255, size=(80, 3))

def draw_detection(
    frame: np.ndarray,
    box: list,
    label: str,
    confidence: float,
    class_id: int
) -> np.ndarray:
    """
    Draw a single bounding box with label and confidence.

    Args:
        frame      : original frame (BGR)
        box        : [x1, y1, x2, y2]
        label      : class name
        confidence : confidence score (0-1)
        class_id   : class index for color selection
    """
    x1, y1, x2, y2 = map(int, box)
    color = COLORS[class_id % 80].tolist()

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Prepare label text
    text = f"{label} {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Draw label background
    cv2.rectangle(frame,
                  (x1, y1 - text_h - 8),
                  (x1 + text_w + 4, y1),
                  color, -1)

    # Draw label text
    cv2.putText(frame, text,
                (x1 + 2, y1 - 4),
                font, font_scale,
                (255, 255, 255),
                thickness)

    return frame

def draw_all_detections(
    frame: np.ndarray,
    results,
    model_names: dict
) -> np.ndarray:
    """
    Draw all detections from a YOLOv8 result object.

    Args:
        frame       : original frame (BGR)
        results     : YOLOv8 result object
        model_names : model.names dict {id: label}
    """
    if results.boxes is None:
        return frame

    for box in results.boxes:
        class_id = int(box.cls[0])
        label = model_names[class_id]
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        frame = draw_detection(frame, coords, label, confidence, class_id)

    return frame

def draw_info_bar(
    frame: np.ndarray,
    fps: float,
    object_count: int,
    device: str = 'GPU'
) -> np.ndarray:
    """
    Draw top info bar showing FPS, object count, device.

    Args:
        frame        : original frame
        fps          : current FPS
        object_count : number of detected objects
        device       : 'GPU' or 'CPU'
    """
    # Semi-transparent bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Info text
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Objects: {object_count}",
                (150, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Device: {device}",
                (320, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    return frame