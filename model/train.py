# model/train.py
# YOLOv8 Training Script — Optimized for RTX 4060 (8GB VRAM)

import torch
from ultralytics import YOLO
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.augmentation import get_augmentation_config

def check_gpu():
    """Check and display GPU information."""
    print("\n--- GPU Check ---")
    if torch.cuda.is_available():
        print(f"  CUDA Available    ✅")
        print(f"  GPU               {torch.cuda.get_device_name(0)}")
        print(f"  VRAM              {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  CUDA Version      {torch.version.cuda}")
        return 'cuda'
    else:
        print("  CUDA              ❌ Not available — training on CPU (slow!)")
        return 'cpu'
    print("-----------------\n")

def train_model(
    data_yaml='dataset/data.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    resume=False
):
    """
    Train YOLOv8m on COCO dataset.

    Args:
        data_yaml : path to data.yaml
        epochs    : number of training epochs
        batch     : batch size (reduce to 8 if VRAM runs out)
        imgsz     : input image size
        resume    : resume from last checkpoint
    """
    device = check_gpu()

    # Load pretrained YOLOv8m model
    # First run will auto-download yolov8m.pt (~50MB)
    print("\nLoading YOLOv8m pretrained model...")
    model = YOLO('yolov8m.pt')

    # Get augmentation settings
    aug_config = get_augmentation_config()

    print("\nStarting training...")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch}")
    print(f"  Image size : {imgsz}")
    print(f"  Device     : {device}")
    print(f"  Data       : {data_yaml}\n")

    # Train the model
    results = model.train(
        # Dataset
        data=data_yaml,

        # Training duration
        epochs=epochs,
        patience=20,          # Early stopping if no improvement for 20 epochs

        # Hardware
        device=device,
        workers=4,            # Data loading workers
        batch=batch,

        # Image settings
        imgsz=imgsz,
        cache=True,           # Cache images in RAM for faster training

        # Optimization
        optimizer='AdamW',
        lr0=0.01,             # Initial learning rate
        lrf=0.01,             # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,      # Warmup for first 3 epochs

        # Performance
        amp=True,             # Mixed precision — crucial for RTX 4060

        # Saving
        project='model/runs',
        name='yolov8m_coco',
        save=True,
        save_period=10,       # Save checkpoint every 10 epochs

        # Logging
        plots=True,           # Save training plots
        verbose=True,

        # Resume
        resume=resume,

        # Augmentation
        **aug_config
    )

    print(f"\n✅ Training complete!")
    print(f"   Best model : {results.save_dir}/weights/best.pt")
    print(f"   Last model : {results.save_dir}/weights/last.pt")

    return results

if __name__ == '__main__':
    # For quick testing use coco128
    # Change to 'dataset/data.yaml' for full COCO training
    train_model(
        data_yaml='dataset/data.yaml',
        epochs=100,
        batch=16,
        imgsz=640
    )