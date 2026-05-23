from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.inference import (
    DEFAULT_CLASSES_PATH,
    FRAME_STEP,
    MAX_FRAMES_TO_ANALYZE,
    MAX_VIDEO_SECONDS,
    image_from_bytes,
    load_class_names,
    load_model_once,
    predict_pil_image,
    predict_video_file,
    save_upload_to_temp,
)

APP_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_DIR / "static"

app = FastAPI(
    title="Soybean Plant Disease Classifier API",
    description=(
        "FastAPI backend for soybean plant disease classification. "
        "Supports gallery image upload, camera-captured image upload, gallery video upload, "
        "and recorded short video upload."
    ),
    version="1.0.0",
)

# For development and Android testing, allow all origins.
# In production, replace '*' with your website/app domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def startup_load_model() -> None:
    """Load model when API starts."""
    load_model_once()


@app.get("/")
def home() -> FileResponse:
    """Browser test page for image upload, camera capture, video upload, and 10-sec video recording."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> Dict[str, Any]:
    bundle = load_model_once()
    return {
        "status": "ok",
        "model_loaded": bundle.loaded,
        "device": str(bundle.device),
        "num_classes": len(bundle.class_names),
        "classes": bundle.class_names,
    }


@app.get("/classes")
def classes() -> Dict[str, Any]:
    class_names = load_class_names(DEFAULT_CLASSES_PATH)
    return {"num_classes": len(class_names), "classes": class_names}


@app.get("/video-settings")
def video_settings() -> Dict[str, Any]:
    return {
        "max_video_seconds": MAX_VIDEO_SECONDS,
        "frame_step": FRAME_STEP,
        "max_frames_to_analyze": MAX_FRAMES_TO_ANALYZE,
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Predict disease from an image.

    Works for:
    - Gallery image selected by user
    - Camera image captured by Android/web frontend
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        file_bytes = await file.read()
        image = image_from_bytes(file_bytes)
        result = predict_pil_image(image, top_k=3)
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)) -> JSONResponse:
    """
    Predict disease from a short video.

    Works for:
    - Gallery video selected by user
    - 10-second recorded video from Android/web frontend

    Limits:
    - Only accept videos up to 10 seconds
    - Analyze every 10th frame
    - Analyze maximum 30 frames
    """
    content_type = file.content_type or ""
    filename = file.filename or "video.mp4"
    allowed = content_type.startswith("video/") or filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))

    if not allowed:
        raise HTTPException(status_code=400, detail="Please upload a valid video file.")

    suffix = Path(filename).suffix.lower() or ".mp4"
    temp_path = ""

    try:
        file_bytes = await file.read()
        temp_path = save_upload_to_temp(file_bytes, suffix=suffix)
        result = predict_video_file(temp_path)
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video prediction failed: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
