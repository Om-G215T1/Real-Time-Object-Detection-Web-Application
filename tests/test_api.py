# tests/test_api.py
# Tests all Flask API endpoints

import os
import sys
import requests
import numpy as np
import cv2
import io
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = 'http://localhost:5000/api'

def create_dummy_image_bytes() -> bytes:
    """Create a small dummy JPEG image in memory."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some color so it's not completely black
    cv2.rectangle(img, (100, 100), (300, 300), (255, 100, 50), -1)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

# ─── Test 1: Health Check ─────────────────────────────────

def test_health():
    """Test /api/health returns 200 and correct keys."""
    print("\n[TEST 1] Health Check...")
    try:
        res = requests.get(f"{BASE_URL}/health", timeout=5)
        assert res.status_code == 200

        data = res.json()
        assert 'status' in data
        assert data['status'] == 'ok'
        assert 'device' in data

        print(f"  ✅ API is online")
        print(f"  ✅ Device: {data['device'].upper()}")
        return True
    except requests.ConnectionError:
        print(f"  ❌ Cannot connect — is Flask running?")
        print(f"       Run: python backend/app.py")
        return False
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Test 2: Image Detection ─────────────────────────────

def test_image_detection():
    """Test POST /api/detect/image with a dummy image."""
    print("\n[TEST 2] Image Detection API...")
    try:
        img_bytes = create_dummy_image_bytes()

        res = requests.post(
            f"{BASE_URL}/detect/image",
            files={'file': ('test.jpg', img_bytes, 'image/jpeg')},
            data={'conf': '0.5'},
            timeout=30
        )

        assert res.status_code == 200
        data = res.json()

        assert 'image' in data
        assert 'detections' in data
        assert 'count' in data
        assert 'fps' in data

        # Validate base64 image is returned
        assert len(data['image']) > 0

        print(f"  ✅ Status: {res.status_code}")
        print(f"  ✅ Objects detected: {data['count']}")
        print(f"  ✅ FPS: {data['fps']}")
        print(f"  ✅ Response keys: {list(data.keys())}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Test 3: Invalid File Type ────────────────────────────

def test_invalid_file_type():
    """Test that invalid file types are rejected."""
    print("\n[TEST 3] Invalid File Type Rejection...")
    try:
        res = requests.post(
            f"{BASE_URL}/detect/image",
            files={'file': ('test.txt', b'hello world', 'text/plain')},
            timeout=10
        )

        assert res.status_code == 400
        data = res.json()
        assert 'error' in data

        print(f"  ✅ Rejected with status {res.status_code}")
        print(f"  ✅ Error message: {data['error']}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Test 4: No File Provided ─────────────────────────────

def test_no_file():
    """Test that missing file returns proper error."""
    print("\n[TEST 4] Missing File Error...")
    try:
        res = requests.post(
            f"{BASE_URL}/detect/image",
            timeout=10
        )
        assert res.status_code == 400
        data = res.json()
        assert 'error' in data

        print(f"  ✅ Rejected with status {res.status_code}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Test 5: Model List ───────────────────────────────────

def test_model_list():
    """Test GET /api/model/list returns model options."""
    print("\n[TEST 5] Model List API...")
    try:
        res = requests.get(f"{BASE_URL}/model/list", timeout=10)
        assert res.status_code == 200

        data = res.json()
        assert 'pretrained' in data
        assert 'trained' in data
        assert 'current' in data

        print(f"  ✅ Pretrained models: {data['pretrained']}")
        print(f"  ✅ Current model: {data['current']}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Test 6: Model Switch ─────────────────────────────────

def test_model_switch():
    """Test POST /api/model/switch changes the model."""
    print("\n[TEST 6] Model Switch API...")
    try:
        res = requests.post(
            f"{BASE_URL}/model/switch",
            json={'model': 'yolov8n.pt'},
            timeout=30
        )
        assert res.status_code == 200
        data = res.json()
        assert data['status'] == 'switched'

        print(f"  ✅ Switched to: {data['model']}")

        # Switch back
        requests.post(
            f"{BASE_URL}/model/switch",
            json={'model': 'yolov8m.pt'},
            timeout=30
        )
        print(f"  ✅ Switched back to yolov8m.pt")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Test 7: Confidence Threshold ────────────────────────

def test_confidence_threshold():
    """Test that different confidence values affect results."""
    print("\n[TEST 7] Confidence Threshold API...")
    try:
        img_bytes = create_dummy_image_bytes()

        # Low confidence
        res_low = requests.post(
            f"{BASE_URL}/detect/image",
            files={'file': ('test.jpg', img_bytes, 'image/jpeg')},
            data={'conf': '0.1'},
            timeout=30
        )

        # High confidence
        res_high = requests.post(
            f"{BASE_URL}/detect/image",
            files={'file': ('test.jpg', img_bytes, 'image/jpeg')},
            data={'conf': '0.9'},
            timeout=30
        )

        assert res_low.status_code == 200
        assert res_high.status_code == 200

        count_low  = res_low.json()['count']
        count_high = res_high.json()['count']

        assert count_high <= count_low

        print(f"  ✅ conf=0.1 → {count_low} detections")
        print(f"  ✅ conf=0.9 → {count_high} detections")
        print(f"  ✅ Confidence filtering works via API")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Test 8: API Response Time ────────────────────────────

def test_response_time(num_requests=5):
    """Benchmark API response time over N requests."""
    print(f"\n[TEST 8] API Response Time ({num_requests} requests)...")
    try:
        img_bytes = create_dummy_image_bytes()
        times = []

        for i in range(num_requests):
            start = time.time()
            res = requests.post(
                f"{BASE_URL}/detect/image",
                files={'file': ('test.jpg', img_bytes, 'image/jpeg')},
                data={'conf': '0.5'},
                timeout=30
            )
            elapsed = time.time() - start
            times.append(elapsed)
            assert res.status_code == 200

        avg = sum(times) / len(times)
        print(f"  ✅ Avg response time : {avg*1000:.0f}ms")
        print(f"  ✅ Min response time : {min(times)*1000:.0f}ms")
        print(f"  ✅ Max response time : {max(times)*1000:.0f}ms")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Test 9: Unknown Detections Endpoint ─────────────────

def test_unknown_detections():
    """Test GET /api/detections/unknown."""
    print("\n[TEST 9] Unknown Detections API...")
    try:
        res = requests.get(
            f"{BASE_URL}/detections/unknown",
            timeout=10
        )
        assert res.status_code == 200
        data = res.json()
        assert 'count' in data
        assert 'files' in data

        print(f"  ✅ Saved low-conf frames: {data['count']}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

# ─── Run All API Tests ────────────────────────────────────

def run_all_api_tests():
    print("\n" + "="*50)
    print("  API TEST SUITE")
    print("="*50)

    results = {
        'Health Check'           : test_health(),
        'Image Detection'        : test_image_detection(),
        'Invalid File Rejection' : test_invalid_file_type(),
        'Missing File Error'     : test_no_file(),
        'Model List'             : test_model_list(),
        'Model Switch'           : test_model_switch(),
        'Confidence Threshold'   : test_confidence_threshold(),
        'API Response Time'      : test_response_time(),
        'Unknown Detections'     : test_unknown_detections(),
    }

    print("\n" + "="*50)
    print("  API TEST RESULTS")
    print("="*50)

    passed = failed = 0
    for test, result in results.items():
        if result:
            print(f"  ✅ PASS    {test}")
            passed += 1
        else:
            print(f"  ❌ FAIL    {test}")
            failed += 1

    print("="*50)
    print(f"  Passed : {passed}")
    print(f"  Failed : {failed}")
    print("="*50 + "\n")

    return failed == 0

if __name__ == '__main__':
    run_all_api_tests()