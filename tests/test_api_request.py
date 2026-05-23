"""
Run after starting API:

    uvicorn api.main:app --reload

Then test:

    python tests/test_api_request.py path/to/leaf.jpg
"""

import sys
import requests

if len(sys.argv) < 2:
    print("Usage: python tests/test_api_request.py path/to/leaf.jpg")
    raise SystemExit(1)

image_path = sys.argv[1]

with open(image_path, "rb") as f:
    files = {"file": (image_path, f, "image/jpeg")}
    response = requests.post("http://127.0.0.1:8000/predict/image", files=files)

print(response.status_code)
print(response.json())
