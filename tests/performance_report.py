# tests/performance_report.py
# Generates a full performance report — FPS, memory, GPU usage

import os
import sys
import time
import json
import torch
import numpy as np
import cv2
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from utils.config import MODEL_PATH, CONFIDENCE_THRESHOLD

def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved  = torch.cuda.memory_reserved() / 1e6
        total     = torch.cuda.get_device_properties(0).total_memory / 1e6
        return {
            'allocated_mb' : round(allocated, 1),
            'reserved_mb'  : round(reserved, 1),
            'total_mb'     : round(total, 1),
            'free_mb'      : round(total - allocated, 1)
        }
    return {'error': 'No GPU'}

def benchmark_fps(model, image_size=640, num_frames=50):
    """Benchmark FPS at a given image size."""
    dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    times = []

    # Warmup
    for _ in range(5):
        model.predict(source=dummy, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # Benchmark
    for _ in range(num_frames):
        start = time.perf_counter()
        model.predict(source=dummy, conf=CONFIDENCE_THRESHOLD, verbose=False)
        times.append(time.perf_counter() - start)

    avg_fps = 1 / (sum(times) / len(times))
    return {
        'avg_fps'    : round(avg_fps, 1),
        'min_fps'    : round(1 / max(times), 1),
        'max_fps'    : round(1 / min(times), 1),
        'avg_ms'     : round((sum(times) / len(times)) * 1000, 1)
    }

def generate_report():
    """Generate and save full performance report."""
    print("\n" + "="*55)
    print("  PERFORMANCE REPORT — YOLOv8 Object Detection")
    print("="*55)

    report = {
        'generated_at' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model'        : MODEL_PATH,
        'device'       : 'cuda' if torch.cuda.is_available() else 'cpu',
        'gpu_name'     : torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
    }

    # Load model
    print("\nLoading model...")
    model = YOLO(MODEL_PATH)
    device = report['device']
    model.to(device)

    print(f"  Model  : {MODEL_PATH}")
    print(f"  Device : {device.upper()}")
    if device == 'cuda':
        print(f"  GPU    : {report['gpu_name']}")

    # GPU Memory before
    print("\n--- GPU Memory (Before Inference) ---")
    mem_before = get_gpu_memory()
    for k, v in mem_before.items():
        print(f"  {k:<20} {v}")
    report['gpu_memory_before'] = mem_before

    # FPS benchmarks at different resolutions
    print("\n--- FPS Benchmark ---")
    fps_results = {}
    for size in [320, 480, 640]:
        print(f"  Benchmarking at {size}x{size}...")
        result = benchmark_fps(model, image_size=size, num_frames=30)
        fps_results[f'{size}x{size}'] = result
        print(f"    Avg FPS : {result['avg_fps']}")
        print(f"    Avg ms  : {result['avg_ms']}ms")

    report['fps_benchmarks'] = fps_results

    # GPU Memory after
    print("\n--- GPU Memory (After Inference) ---")
    mem_after = get_gpu_memory()
    for k, v in mem_after.items():
        print(f"  {k:<20} {v}")
    report['gpu_memory_after'] = mem_after

    # Performance summary
    best_fps = fps_results.get('640x640', {}).get('avg_fps', 0)
    target   = 30 if device == 'cuda' else 15
    passed   = best_fps >= target

    print("\n--- Summary ---")
    print(f"  Best FPS (640x640) : {best_fps}")
    print(f"  Target FPS         : ≥{target}")
    print(f"  Performance        : {'✅ PASS' if passed else '❌ FAIL'}")

    report['summary'] = {
        'best_fps'   : best_fps,
        'target_fps' : target,
        'passed'     : passed
    }

    # Save report to JSON
    os.makedirs('tests/reports', exist_ok=True)
    report_path = f"tests/reports/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved: {report_path}")
    print("="*55 + "\n")

    return report

if __name__ == '__main__':
    generate_report()