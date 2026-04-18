# model/test.py
# Test YOLOv8 on image, video, and webcam

import cv2
import torch
import time
import os
from ultralytics import YOLO

# Use best.pt if available, else fall back to pretrained
MODEL_PATH = (
    'model/runs/yolov8m_coco/weights/best.pt'
    if os.path.exists('model/runs/yolov8m_coco/weights/best.pt')
    else 'yolov8m.pt'
)

def load_model(model_path=MODEL_PATH):
    """Load YOLOv8 model onto GPU if available."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nLoading model from: {model_path}")
    print(f"Device: {device}")
    model = YOLO(model_path)
    model.to(device)
    return model

def test_image(image_path: str, conf: float = 0.5):
    """
    Run detection on a single image.
    Saves annotated result to runs/detect/
    """
    print(f"\nTesting on image: {image_path}")
    model = load_model()

    results = model.predict(
        source=image_path,
        conf=conf,
        save=True,              # Saves annotated image
        project='model/runs',
        name='test_image'
    )

    result = results[0]
    print(f"Detected {len(result.boxes)} objects:")
    for box in result.boxes:
        label = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        print(f"  {label:<20} {confidence:.2%}")

    return results

def test_video(video_path: str, conf: float = 0.5):
    """
    Run detection on a video file.
    Displays FPS in real time. Press Q to quit.
    """
    print(f"\nTesting on video: {video_path}")
    model = load_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    prev_time = time.time()
    frame_count = 0
    fps_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model.predict(source=frame, conf=conf, verbose=False)
        annotated = results[0].plot()

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_list.append(fps)
        frame_count += 1

        # Draw FPS on frame
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Objects: {len(results[0].boxes)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Video Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"\n✅ Video test complete")
    print(f"   Frames processed : {frame_count}")
    print(f"   Average FPS      : {avg_fps:.1f}")

def test_webcam(conf: float = 0.5, camera_id: int = 0):
    """
    Real-time webcam detection.
    Press Q to quit.
    """
    print(f"\nStarting webcam detection (Camera ID: {camera_id})")
    print("Press Q to quit\n")

    model = load_model()
    cap = cv2.VideoCapture(camera_id)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    prev_time = time.time()
    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Run detection
        results = model.predict(source=frame, conf=conf, verbose=False)
        annotated = results[0].plot()

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_list.append(fps)

        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Objects: {len(results[0].boxes)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Webcam Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"\n✅ Webcam test complete")
    print(f"   Average FPS : {avg_fps:.1f}")

if __name__ == '__main__':
    # Change to test_image() or test_video() as needed
    test_webcam()