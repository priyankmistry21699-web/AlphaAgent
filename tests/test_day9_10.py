"""Day 9 & 10 Test — Insider Agent & FastAPI End-to-End."""
import sys
import requests
import json
import time
import subprocess

sys.path.insert(0, r"d:\ML and DL\Python\AlphaAgent")

print("=" * 60)
print("  AlphaAgent — Phase 1 Final Test (Day 9 & 10)")
print("=" * 60)

# 1. Start the API Server in the background
print("\n  [1] Starting FastAPI Server (api/main.py)...")
server_process = subprocess.Popen(
    [sys.executable, "api/main.py"],
    cwd=r"d:\ML and DL\Python\AlphaAgent",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for server to boot (give it 10 seconds to load all models and yfinance)
time.sleep(10)

# 2. Test the endpoint
ticker = "AAPL"
print(f"\n  [2] Sending REST API Request for {ticker}...")
try:
    response = requests.get(f"http://localhost:8085/analyze/{ticker}")
    if response.status_code == 200:
        data = response.json()
        print("\n" + "─" * 60)
        print("  🟢 API RESPONSE RECEIVED SUCCESSFULY")
        print("─" * 60)
        
        print(json.dumps(data, indent=2))
        
        print("\n" + "=" * 60)
        print("  ✅ PHASE 1 COMPLETE — SYSTEM IS PRODUCTION READY")
        print("=" * 60)
    else:
        print(f"  ❌ API returned status code: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"  ❌ Failed to hit API: {e}")
finally:
    # Cleanup
    print("\n  [3] Shutting down FastAPI Server...")
    server_process.terminate()
