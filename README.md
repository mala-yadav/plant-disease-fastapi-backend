# Soybean Plant Disease Classifier - FastAPI Backend

This is **Stage 2** of your deployment:

```text
Android/Web frontend
        ↓
FastAPI backend
        ↓
PyTorch model: best_residual_model.pt
        ↓
JSON prediction result
```

## Features

- Image prediction from gallery upload
- Image prediction from camera-captured image
- Video prediction from gallery upload
- Live recorded video prediction from browser test page
- Video limit: only up to 10 seconds
- Video sampling: every 10th frame
- Maximum analyzed frames: 30
- Top-3 predictions
- 60% confidence threshold
- Disease care suggestions

## Project structure

```text
plant_disease_api_stage2/
├── api/
│   ├── main.py
│   └── inference.py
├── model/
│   ├── model.py
│   ├── best_residual_model.pt
│   └── class_names.json
├── static/
│   └── index.html
├── tests/
│   └── test_api_request.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Your classes

```json
{
  "Healthy_Soyabean": 0,
  "Soyabean Semilooper and Caterpillar_Pest_Attack": 1,
  "Soyabean_Mosaic": 2,
  "Soyabean_Rust": 3
}
```

## Run locally

### 1. Create virtual environment

Windows PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Linux/Mac:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Start API

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

For API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## API endpoints

### Health

```http
GET /health
```

### Classes

```http
GET /classes
```

### Image prediction

```http
POST /predict/image
```

Form-data:

```text
file = leaf image
```

Works for:

```text
gallery image
camera captured image
Android app image upload
```

Example response:

```json
{
  "predicted_class": "Soyabean_Rust",
  "confidence_percent": 97.42,
  "is_low_confidence": false,
  "confidence_threshold_percent": 60,
  "care_suggestion": "Possible soybean rust...",
  "top_predictions": [
    {
      "class_name": "Soyabean_Rust",
      "probability": 0.9742,
      "confidence_percent": 97.42,
      "care_suggestion": "Possible soybean rust..."
    }
  ]
}
```

### Video prediction

```http
POST /predict/video
```

Form-data:

```text
file = short video
```

Rules:

```text
Only accept videos up to 10 seconds
Analyze every 10th frame
Analyze maximum 30 frames
```

## Important concept: camera is frontend responsibility

FastAPI does **not** open the camera itself.

Correct flow:

```text
Android app/browser opens camera
        ↓
captures photo/video
        ↓
sends file to FastAPI
        ↓
FastAPI predicts
        ↓
Android/browser displays result
```

## Test with Python request

Start server:

```bash
uvicorn api.main:app --reload
```

Then run:

```bash
python tests/test_api_request.py sample_leaf.jpg
```

## Docker run locally

Build:

```bash
docker build -t plant-disease-api .
```

Run:

```bash
docker run -p 8000:8000 plant-disease-api
```

Open:

```text
http://127.0.0.1:8000
```

## Deploy to cloud

Good beginner options:

```text
Render
Railway
Hugging Face Spaces with Docker
AWS EC2
```

For Render/Railway, push this folder to GitHub and use:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

For Docker-based deployment, use the included Dockerfile.

## Common problems and fixes

### 1. Model loading error

Cause:

```text
wrong class_names.json
wrong weight filename
model.py architecture mismatch
```

Fix:

```text
Keep best_residual_model.pt inside model/
Keep class_names.json same as training
Do not change model.py unless retraining
```

### 2. Camera not opening

Cause:

```text
Browser allows camera only on localhost or HTTPS
```

Fix:

```text
Use localhost during testing
Use HTTPS cloud URL after deployment
Allow camera permission in browser/app
```

### 3. Android cannot call API

Cause:

```text
wrong URL
HTTP blocked
server sleeping
CORS issue
```

Fix:

```text
Use HTTPS URL
Test /health first
Keep CORS enabled during development
```

### 4. Video fails

Cause:

```text
unsupported codec
video longer than 10 seconds
large file
OpenCV cannot read format
```

Fix:

```text
Use MP4/H.264 if possible
Keep video <= 10 seconds
Use camera/photo prediction for most reliable result
```

### 5. Slow prediction

Cause:

```text
CPU server
large uploaded image
video too many frames
```

Fix:

```text
Model loads once at startup
No internet download needed because API uses use_pretrained_backbone=False for inference architecture creation
Backend resizes to 224x224
Video samples every 10th frame only
Use better server later
```

## Next stage

After this API works locally, build Android app:

```text
Android app
↓
Camera/gallery image
↓
POST /predict/image
↓
Show disease, confidence, top-3, care suggestion
```

For video:

```text
Android records max 10 sec
↓
POST /predict/video
↓
Show final video-level prediction
```
