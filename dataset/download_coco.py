# dataset/download_coco.py
# Downloads COCO128 (sample dataset) for testing
# Use full COCO only when ready for final training

from ultralytics.utils.downloads import download
import os

def download_coco128():
    """
    Downloads COCO128 — a small 128-image sample of COCO.
    Perfect for testing your training pipeline before full training.
    Full COCO dataset is 20GB+ — only download when ready.
    """
    print("Downloading COCO128 sample dataset...")
    download(
        url='https://ultralytics.com/assets/coco128.zip',
        dir='dataset/'
    )
    print("COCO128 downloaded successfully!")
    print("Location: dataset/coco128/")

def download_full_coco():
    """
    Downloads full COCO dataset (~20GB).
    Only run this when you are ready for final training.
    Requires stable internet and enough disk space.
    """
    print("WARNING: This will download ~20GB of data!")
    confirm = input("Are you sure? (yes/no): ")
    if confirm.lower() == 'yes':
        from ultralytics import YOLO
        model = YOLO('yolov8m.pt')
        model.train(data='coco.yaml', epochs=1, imgsz=640)
    else:
        print("Cancelled.")

if __name__ == '__main__':
    download_coco128()