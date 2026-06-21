"""
test_predict.py — Quick manual test for the /predict endpoint.

BUG FIX: The original script sent heart disease features (age, sex, cp, trestbps…)
to an endpoint that expects Pima Indians Diabetes features. This caused the server
to silently default all values to 0, producing meaningless predictions.

Corrected to use the proper Pima dataset feature names and point to localhost.

Usage::

    # With server running:
    python test_predict.py

    # Against a remote server:
    python test_predict.py --server http://192.168.1.100:5000
"""

import argparse
import sys

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)


def run_test(server_url: str) -> None:
    """Send a sample diabetes patient record and print the prediction.

    Args:
        server_url: Base URL of the FastAPI server (e.g. 'http://localhost:5000').
    """
    # High-risk profile (elevated glucose, high BMI, multiple pregnancies)
    high_risk = {
        "Pregnancies": 8,
        "Glucose": 183,
        "BloodPressure": 64,
        "SkinThickness": 0,
        "Insulin": 0,
        "BMI": 35.2,
        "DiabetesPedigreeFunction": 0.672,
        "Age": 50,
    }

    # Low-risk profile (healthy values)
    low_risk = {
        "Pregnancies": 1,
        "Glucose": 89,
        "BloodPressure": 66,
        "SkinThickness": 23,
        "Insulin": 94,
        "BMI": 22.5,
        "DiabetesPedigreeFunction": 0.167,
        "Age": 25,
    }

    endpoint = f"{server_url.rstrip('/')}/predict"
    print(f"\n🌐 Testing endpoint: {endpoint}\n{'='*55}")

    for label, payload in [("High-Risk Patient", high_risk), ("Low-Risk Patient", low_risk)]:
        print(f"\n🔍 {label}")
        print(f"   Input: {payload}")
        try:
            resp = requests.post(endpoint, json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            print(f"   ✅ Status       : {result.get('status')}")
            print(f"   📊 Risk Level   : {result.get('risk_level')}")
            print(f"   📈 Risk %       : {result.get('risk_percentage')}")
            print(f"   💬 Interpretation: {result.get('interpretation')}")
            print(f"   🩺 Recommendation: {result.get('recommendation')}")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Cannot connect to {endpoint}. Is the server running?")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "="*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the /predict endpoint of the Diabetes Risk Prediction server."
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:5000",
        help="Server base URL (default: http://localhost:5000)",
    )
    args = parser.parse_args()
    run_test(args.server)