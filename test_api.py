"""
test_api.py — Tests both MoodLens API endpoints.

Usage:
  python test_api.py                                   # uses week1-4.json by default
  python test_api.py w1.json w2.json w3.json w4.json  # custom files
"""
import sys, json, requests
from datetime import datetime

BASE_URL = "http://127.0.0.1:5000"
HEADERS  = {"Content-Type": "application/json", "x-api-key": "mood_flow_model_key"}

# ── Load the 4 weekly files ───────────────────────────────────
default_files = ["week1.json", "week2.json", "week3.json", "week4.json"]
week_files    = sys.argv[1:] if len(sys.argv) == 5 else default_files
ts            = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Loading: {week_files}\n")

weekly_payloads = []
for path in week_files:
    with open(path) as f:
        data = json.load(f)
    weekly_payloads.append(data[0] if isinstance(data, list) else data)

# ── Test 1: /predict_all  (week 1 sanity check) ──────────────
print("=" * 55)
print("TEST 1: /predict_all  (week 1 sanity check)")
print("=" * 55)
r = requests.post(f"{BASE_URL}/predict_all",
                  json=[weekly_payloads[0]], headers=HEADERS)
print(f"Status : {r.status_code}")
print(json.dumps(r.json(), indent=4))

# ── Test 2: /predict_report_weekly  (all 4 weeks) ────────────
print("\n" + "=" * 55)
print("TEST 2: /predict_report_weekly  (4-week longitudinal)")
print("=" * 55)
r = requests.post(f"{BASE_URL}/predict_report_weekly",
                  json={"weeks": weekly_payloads}, headers=HEADERS)
print(f"Status : {r.status_code}")
if r.status_code == 200:
    out = f"moodlens_weekly_report_{ts}.pdf"
    with open(out, "wb") as f:
        f.write(r.content)
    print(f"Saved  : {out}  ({len(r.content):,} bytes)")
else:
    print("Error  :", r.text)