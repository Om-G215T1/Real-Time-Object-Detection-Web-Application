# tests/run_all_tests.py
# Master test runner — runs all test suites in order

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_model import run_all_model_tests
from tests.test_api import run_all_api_tests
from tests.performance_report import generate_report
from datetime import datetime

def run_all():
    print("\n" + "🔷"*25)
    print("  FULL TEST SUITE — YOLOv8 Object Detection")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔷"*25)

    results = {}

    # 1. Model tests
    print("\n📦 Running Model Tests...")
    results['model'] = run_all_model_tests()

    # 2. API tests (requires Flask running)
    print("\n🌐 Running API Tests...")
    print("   ⚠️  Make sure Flask is running: python backend/app.py")
    results['api'] = run_all_api_tests()

    # 3. Performance report
    print("\n⚡ Generating Performance Report...")
    report = generate_report()
    results['performance'] = report['summary']['passed']

    # Final summary
    print("\n" + "="*50)
    print("  FULL TEST SUITE RESULTS")
    print("="*50)
    for suite, passed in results.items():
        status = '✅ PASS' if passed else '❌ FAIL'
        print(f"  {status}    {suite.upper()} TESTS")
    print("="*50 + "\n")

if __name__ == '__main__':
    run_all()