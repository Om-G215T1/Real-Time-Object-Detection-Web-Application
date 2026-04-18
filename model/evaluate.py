# model/evaluate.py
# Evaluate YOLOv8 model using mAP, precision, recall

import os
from ultralytics import YOLO

MODEL_PATH = (
    'model/runs/yolov8m_coco/weights/best.pt'
    if os.path.exists('model/runs/yolov8m_coco/weights/best.pt')
    else 'yolov8m.pt'
)

def evaluate_model(data_yaml='dataset/data.yaml'):
    """
    Evaluate model and print:
    - mAP50
    - mAP50-95
    - Precision
    - Recall
    """
    print(f"\nEvaluating model: {MODEL_PATH}")
    print(f"Dataset         : {data_yaml}\n")

    model = YOLO(MODEL_PATH)

    # Run validation
    metrics = model.val(
        data=data_yaml,
        imgsz=640,
        conf=0.5,
        iou=0.6,
        project='model/runs',
        name='evaluation',
        verbose=True
    )

    # Print results
    print("\n--- Evaluation Results ---")
    print(f"  mAP50       : {metrics.box.map50:.4f}")
    print(f"  mAP50-95    : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    print("--------------------------\n")

    return metrics

if __name__ == '__main__':
    evaluate_model()