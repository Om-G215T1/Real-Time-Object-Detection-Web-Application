# deployment/gradio_app.py
# Gradio app for Hugging Face Spaces deployment
# Converts Flask app to Gradio interface

import os
import sys
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from ultralytics import YOLO

# ─── Model Loading ────────────────────────────────────────

# On Hugging Face — uses CPU (no GPU available on free tier)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = os.getenv('MODEL_PATH', 'yolov8m.pt')

print(f"Loading model: {MODEL_PATH} on {DEVICE}")
model = YOLO(MODEL_PATH)
model.to(DEVICE)
print("Model loaded ✅")

# ─── Detection Functions ──────────────────────────────────

def detect_image(
    image: Image.Image,
    conf_threshold: float = 0.5,
    model_name: str = 'yolov8m.pt'
) -> tuple:
    """
    Run YOLOv8 detection on an uploaded image.

    Args:
        image          : PIL Image from Gradio
        conf_threshold : confidence threshold slider value
        model_name     : selected model name

    Returns:
        annotated PIL image, detection summary string
    """
    if image is None:
        return None, "No image provided"

    # Switch model if changed
    global model, MODEL_PATH
    if model_name != MODEL_PATH:
        print(f"Switching to {model_name}")
        model = YOLO(model_name)
        model.to(DEVICE)
        MODEL_PATH = model_name

    # Convert PIL to OpenCV BGR
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Run inference
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        verbose=False
    )
    result = results[0]

    # Draw detections
    annotated = result.plot()

    # Convert back to PIL RGB
    annotated_pil = Image.fromarray(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    )

    # Build detection summary
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            conf  = float(box.conf[0])
            detections.append(f"• {label:<20} {conf:.1%}")

    summary = f"Found {len(detections)} object(s):\n\n"
    summary += "\n".join(detections) if detections else "No objects detected"

    return annotated_pil, summary


def detect_video(
    video_path: str,
    conf_threshold: float = 0.5
) -> str:
    """
    Run YOLOv8 detection on an uploaded video.
    Returns path to annotated output video.

    Args:
        video_path     : path to uploaded video file
        conf_threshold : confidence threshold

    Returns:
        path to output video
    """
    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Output video setup
    output_path = video_path.replace('.mp4', '_detected.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    max_frames  = 300  # Limit to 300 frames on HF free tier

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster inference on CPU
        small = cv2.resize(frame, (640, 360))
        results = model.predict(
            source=small,
            conf=conf_threshold,
            verbose=False
        )
        annotated_small = results[0].plot()

        # Resize back to original
        annotated = cv2.resize(annotated_small, (width, height))
        writer.write(annotated)
        frame_count += 1

    cap.release()
    writer.release()

    return output_path


def detect_webcam(
    frame: np.ndarray,
    conf_threshold: float = 0.5
) -> np.ndarray:
    """
    Run detection on a single webcam frame.
    Used by Gradio's live webcam component.

    Args:
        frame          : numpy array from webcam
        conf_threshold : confidence threshold

    Returns:
        annotated frame as numpy array
    """
    if frame is None:
        return None

    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=bgr,
        conf=conf_threshold,
        verbose=False
    )
    annotated = results[0].plot()

    # Convert back to RGB for Gradio
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# ─── Gradio UI ────────────────────────────────────────────

def build_ui():
    """Build the complete Gradio interface."""

    # Custom CSS
    css = """
        .title { text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #888; margin-bottom: 20px; }
        footer { display: none !important; }
    """

    with gr.Blocks(
        title="YOLOv8 Object Detection",
        css=css,
        theme=gr.themes.Soft()
    ) as demo:

        # Header
        gr.Markdown(
            "# 🎯 YOLOv8 Real-Time Object Detection",
            elem_classes="title"
        )
        gr.Markdown(
            "Detect 80 COCO classes in images, videos, and live webcam",
            elem_classes="subtitle"
        )

        # Global controls
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=[
                    'yolov8n.pt',
                    'yolov8s.pt',
                    'yolov8m.pt',
                    'yolov8l.pt',
                    'yolov8x.pt'
                ],
                value='yolov8m.pt',
                label="Model (n=fastest, x=most accurate)"
            )
            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.95,
                value=0.5,
                step=0.05,
                label="Confidence Threshold"
            )

        # Tabs
        with gr.Tabs():

            # ── Image Tab ───────────────────────────────
            with gr.Tab("🖼️ Image Detection"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload Image",
                            type="pil"
                        )
                        image_btn = gr.Button(
                            "🔍 Detect Objects",
                            variant="primary"
                        )

                    with gr.Column():
                        image_output = gr.Image(
                            label="Detection Result"
                        )
                        detection_text = gr.Textbox(
                            label="Detected Objects",
                            lines=10,
                            interactive=False
                        )

                # Examples
                gr.Examples(
                    examples=[],  # Add sample images here
                    inputs=image_input
                )

                image_btn.click(
                    fn=detect_image,
                    inputs=[image_input, conf_slider, model_dropdown],
                    outputs=[image_output, detection_text]
                )

            # ── Video Tab ───────────────────────────────
            with gr.Tab("🎬 Video Detection"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(
                            label="Upload Video (max 300 frames on free tier)"
                        )
                        video_btn = gr.Button(
                            "▶ Process Video",
                            variant="primary"
                        )

                    with gr.Column():
                        video_output = gr.Video(
                            label="Processed Video"
                        )

                gr.Markdown(
                    "⚠️ Video processing is limited to 300 frames on "
                    "Hugging Face free tier (CPU only)."
                )

                video_btn.click(
                    fn=detect_video,
                    inputs=[video_input, conf_slider],
                    outputs=video_output
                )

            # ── Webcam Tab ──────────────────────────────
            with gr.Tab("📷 Live Webcam"):
                gr.Markdown(
                    "### Live webcam detection\n"
                    "Allow browser camera access when prompted."
                )
                with gr.Row():
                    webcam_input = gr.Image(
                        label="Webcam Input",
                        sources=["webcam"],
                        streaming=True
                    )
                    webcam_output = gr.Image(
                        label="Detection Output",
                        streaming=True
                    )

                webcam_input.stream(
                    fn=detect_webcam,
                    inputs=[webcam_input, conf_slider],
                    outputs=webcam_output,
                    time_limit=60     # 60 second stream limit
                )

            # ── About Tab ───────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
## About This App

This app uses **YOLOv8** (You Only Look Once v8) for real-time object detection.

### Features
- Detects **80 COCO classes** (people, cars, animals, household items, etc.)
- Supports **image upload**, **video upload**, and **live webcam**
- Adjustable **confidence threshold**
- Multiple **model sizes** (nano to extra-large)

### Models
| Model | Speed | Accuracy |
|-------|-------|----------|
| YOLOv8n | ⚡⚡⚡⚡⚡ | ⭐⭐ |
| YOLOv8s | ⚡⚡⚡⚡ | ⭐⭐⭐ |
| YOLOv8m | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| YOLOv8l | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| YOLOv8x | ⚡ | ⭐⭐⭐⭐⭐ |

### Built With
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Gradio](https://gradio.app)
- [OpenCV](https://opencv.org)
- [PyTorch](https://pytorch.org)

### Source Code
[GitHub Repository](https://github.com/Om-G215T1/Real-Time-Object-Detection-Web-Application)
                """)

    return demo

# ─── Launch ───────────────────────────────────────────────

if __name__ == '__main__':
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False            # Set True to get public link
    )