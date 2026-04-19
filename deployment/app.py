# deployment/app.py
# Entry point for Hugging Face Spaces
# HF requires the main file to be named app.py

from deployment.gradio_app import build_ui

demo = build_ui()

# HF Spaces launches with these settings automatically
demo.launch()