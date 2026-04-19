# Hugging Face Spaces — Deployment Guide

## Step 1 — Create a Hugging Face account
Go to https://huggingface.co and sign up (free).

## Step 2 — Create a new Space
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in:
   - Owner: your username
   - Space name: yolov8-object-detection
   - License: MIT
   - SDK: Gradio
   - Hardware: CPU Basic (free)
4. Click "Create Space"

## Step 3 — Clone the Space repo
```bash
# Install git-lfs first (for large files)
git lfs install

# Clone your HF Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/yolov8-object-detection
cd yolov8-object-detection
```

## Step 4 — Copy deployment files
```bash
# Copy these files into the Space repo:
# - deployment/app.py         → app.py
# - deployment/requirements.txt → requirements.txt
# - deployment/README.md      → README.md
# - deployment/gradio_app.py  → gradio_app.py
```

## Step 5 — Push to Hugging Face
```bash
git add .
git commit -m "Deploy YOLOv8 object detection app"
git push
```

## Step 6 — Monitor build
- Go to your Space URL
- Click "App" tab to see logs
- Build takes ~3-5 minutes
- App goes live automatically

## Notes
- Free tier = CPU only (slower than local GPU)
- yolov8n.pt recommended for free tier (fastest)
- Video limited to 300 frames to avoid timeout
- Model weights auto-download on first run (~6-25MB)