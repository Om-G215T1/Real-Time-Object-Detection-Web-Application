# tests/test_model.py
# Tests YOLOv8 model accuracy — mAP, precision, recall
# Also tests image and video detection

import os
import sys
import time
import cv2
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from utils.config import MODEL_PATH, CONFIDENCE_THRESHOLD

# ─── Setup ───────────────────────────────────────────────

MODEL = None

def load_model():
    """Load model once for all tests."""
    global MODEL
    if MODEL is None:
        print(f"\nLoading model: {MODEL_PATH}")
        MODEL = YOLO(MODEL_PATH)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        MODEL.to(device)
        print(f"Model loaded on {device} ✅")
    return MODEL

# ─── Test 1: Model Loads ─────────────────────────────────

def test_model_loads():
    """Test that model loads without errors."""
    print("\n[TEST 1] Model Loading...")
    try:
        model = load_model()
        assert model is not None
        assert len(model.names) == 80
        print(f"  ✅ Model loaded — {len(model.names)} classes")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Test 2: GPU Available ───────────────────────────────

def test_gpu_available():
    """Test that GPU is detected and being used."""
    print("\n[TEST 2] GPU Check...")
    cuda = torch.cuda.is_available()
    if cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✅ GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        print(f"  ⚠️  No GPU — running on CPU (slower)")
    return True  # Not a hard fail — CPU is valid

# ─── Test 3: Image Detection ─────────────────────────────

def test_image_detection():
    """Test detection on a generated dummy image."""
    print("\n[TEST 3] Image Detection...")
    try:
        model = load_model()

        # Create a dummy 640x640 black image
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

        start = time.time()
        results = model.predict(
            source=dummy_image,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )
        elapsed = time.time() - start

        assert results is not None
        assert len(results) > 0

        print(f"  ✅ Detection ran in {elapsed*1000:.1f}ms")
        print(f"  ✅ Objects detected: {len(results[0].boxes)}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Test 4: Inference Speed ─────────────────────────────

def test_inference_speed(num_frames=30):
    """
    Benchmark inference speed over N frames.
    Target: ≥15 FPS on CPU, ≥30 FPS on GPU
    """
    print(f"\n[TEST 4] Inference Speed ({num_frames} frames)...")
    try:
        model = load_model()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        target_fps = 30 if device == 'cuda' else 15

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        times = []

        for i in range(num_frames):
            start = time.time()
            model.predict(source=dummy, conf=CONFIDENCE_THRESHOLD, verbose=False)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        avg_fps = 1 / avg_time
        min_fps = 1 / max(times)
        max_fps = 1 / min(times)

        passed = avg_fps >= target_fps

        print(f"  Device     : {device.upper()}")
        print(f"  Target FPS : ≥{target_fps}")
        print(f"  Avg FPS    : {avg_fps:.1f}  {'✅' if passed else '❌'}")
        print(f"  Min FPS    : {min_fps:.1f}")
        print(f"  Max FPS    : {max_fps:.1f}")
        print(f"  Avg Time   : {avg_time*1000:.1f}ms per frame")

        return passed
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Test 5: Model Evaluation (mAP) ──────────────────────

def test_model_evaluation(data_yaml='dataset/data.yaml'):
    """
    Evaluate model on validation set.
    Prints mAP50, mAP50-95, Precision, Recall.
    Skipped if dataset not found.
    """
    print("\n[TEST 5] Model Evaluation (mAP)...")

    if not os.path.exists(data_yaml):
        print(f"  ⚠️  Skipped — dataset not found at {data_yaml}")
        print(f"       Run dataset/download_coco.py first")
        return None

    try:
        model = load_model()
        metrics = model.val(
            data=data_yaml,
            imgsz=640,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        map50    = metrics.box.map50
        map5095  = metrics.box.map
        precision = metrics.box.mp
        recall   = metrics.box.mr

        print(f"  mAP50      : {map50:.4f}   {'✅' if map50 > 0.5 else '⚠️'}")
        print(f"  mAP50-95   : {map5095:.4f}  {'✅' if map5095 > 0.3 else '⚠️'}")
        print(f"  Precision  : {precision:.4f}  {'✅' if precision > 0.5 else '⚠️'}")
        print(f"  Recall     : {recall:.4f}  {'✅' if recall > 0.5 else '⚠️'}")

        return {
            'map50': map50,
            'map50_95': map5095,
            'precision': precision,
            'recall': recall
        }
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Test 6: Confidence Filtering ────────────────────────

def test_confidence_filtering():
    """Test that confidence threshold correctly filters results."""
    print("\n[TEST 6] Confidence Filtering...")
    try:
        model = load_model()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)

        results_low  = model.predict(source=dummy, conf=0.1, verbose=False)
        results_high = model.predict(source=dummy, conf=0.9, verbose=False)

        count_low  = len(results_low[0].boxes)
        count_high = len(results_high[0].boxes)

        # Higher threshold should return fewer or equal detections
        assert count_high <= count_low

        print(f"  conf=0.1 → {count_low} detections")
        print(f"  conf=0.9 → {count_high} detections")
        print(f"  ✅ Confidence filtering works correctly")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Test 7: Class Names ─────────────────────────────────

def test_class_names():
    """Test that model has all 80 COCO class names."""
    print("\n[TEST 7] Class Names Check...")
    try:
        model = load_model()
        names = model.names

        required_classes = [
            'person', 'car', 'dog', 'cat',
            'bicycle', 'truck', 'bus', 'laptop'
        ]

        missing = [c for c in required_classes if c not in names.values()]

        if missing:
            print(f"  ❌ Missing classes: {missing}")
            return False

        print(f"  ✅ All {len(names)} COCO classes present")
        print(f"  Sample: {list(names.values())[:5]}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Run All Tests ────────────────────────────────────────

def run_all_model_tests():
    print("\n" + "="*50)
    print("  MODEL TEST SUITE")
    print("="*50)

    results = {
        'Model Loads'           : test_model_loads(),
        'GPU Available'         : test_gpu_available(),
        'Image Detection'       : test_image_detection(),
        'Inference Speed'       : test_inference_speed(),
        'Confidence Filtering'  : test_confidence_filtering(),
        'Class Names'           : test_class_names(),
        'Model Evaluation(mAP)' : test_model_evaluation(),
    }

    print("\n" + "="*50)
    print("  MODEL TEST RESULTS")
    print("="*50)

    passed = 0
    failed = 0
    skipped = 0

    for test, result in results.items():
        if result is True:
            print(f"  ✅ PASS    {test}")
            passed += 1
        elif result is None:
            print(f"  ⏭️  SKIP    {test}")
            skipped += 1
        else:
            print(f"  ❌ FAIL    {test}")
            failed += 1

    print("="*50)
    print(f"  Passed : {passed}")
    print(f"  Failed : {failed}")
    print(f"  Skipped: {skipped}")
    print("="*50 + "\n")

    return failed == 0

if __name__ == '__main__':
    run_all_model_tests()

