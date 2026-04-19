---
title: YOLOv8 Object Detection
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
---

# 🎯 YOLOv8 Real-Time Object Detection

Real-time object detection using YOLOv8 — detects 80 COCO classes
in images, videos, and live webcam.

## Features
- 🖼️ Image upload detection
- 🎬 Video upload detection
- 📷 Live webcam detection
- 🎛️ Adjustable confidence threshold
- 🔄 Multiple model sizes

## How to Use
1. Choose a tab — Image, Video, or Webcam
2. Select model size and confidence threshold
3. Upload your file or start webcam
4. View detections with bounding boxes

## Local Setup
```bash
git clone https://github.com/Om-G215T1/Real-Time-Object-Detection-Web-Application
cd Real-Time-Object-Detection-Web-Application
pip install -r requirements.txt
python backend/app.py
```

## Built With
- YOLOv8 by Ultralytics
- Gradio
- OpenCV
- PyTorch