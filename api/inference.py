"Inference utilities for FastAPI Plant Disease backend."

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast
import json
import tempfile

import cv2
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model.model import DenseNetResidualMQXA

APP_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = APP_DIR / "model"
DEFAULT_WEIGHTS_PATH = MODEL_DIR / "best_residual_model.pt"
DEFAULT_CLASSES_PATH = MODEL_DIR / "class_names.json"

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CONFIDENCE_THRESHOLD = 0.60

MAX_VIDEO_SECONDS = 10
FRAME_STEP = 10
MAX_FRAMES_TO_ANALYZE = 30

CARE_SUGGESTIONS: Dict[str, str] = {
    "Healthy_Soyabean": (
        "Leaf appears healthy. Continue regular monitoring, balanced irrigation, "
        "proper spacing, and routine field inspection."
    ),
    "Soyabean Semilooper and Caterpillar_Pest_Attack": (
        "Possible pest attack. Inspect the underside of leaves for larvae or eggs. "
        "Remove heavily damaged leaves when practical, use pheromone/light traps if available, "
        "and consult a local agriculture expert before pesticide use."
    ),
    "Soyabean_Mosaic": (
        "Possible soybean mosaic or viral symptoms. Avoid using infected seed material, "
        "control insect vectors, remove severely infected plants if confirmed, and consult an agriculture expert."
    ),
    "Soyabean_Rust": (
        "Possible soybean rust. Avoid prolonged leaf wetness, improve field airflow, monitor spread, "
        "and consult a local agriculture expert for suitable fungicide timing and dosage."
    ),
}


class ModelBundle:
    """Keeps loaded model, class names, and device together."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.class_names: List[str] = []
        self.device: torch.device = torch.device("cpu")

    @property
    def loaded(self) -> bool:
        return self.model is not None and bool(self.class_names)


MODEL_BUNDLE = ModelBundle()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names(path: Path = DEFAULT_CLASSES_PATH) -> List[str]:
    """Supports both list format and class_to_idx dict format."""
    if not path.exists():
        raise FileNotFoundError(f"Class file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [name for name, _ in sorted(data.items(), key=lambda item: item[1])]

    if isinstance(data, list):
        return [str(x) for x in data]

    raise ValueError("class_names.json must be either a list or a class_to_index dictionary.")


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    tensor = cast(torch.Tensor, build_transform()(image))
    return tensor.unsqueeze(0)


def _extract_state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def _remove_module_prefix(state_dict: Any) -> Any:
    if not isinstance(state_dict, dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_model_once(
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    classes_path: Path = DEFAULT_CLASSES_PATH,
) -> ModelBundle:
    """Load model once at API startup, not on every request."""
    if MODEL_BUNDLE.loaded:
        return MODEL_BUNDLE

    class_names = load_class_names(classes_path)
    device = get_device()

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weight file not found: {weights_path}. Paste your trained weight file inside model/."
        )

    model = DenseNetResidualMQXA(
        num_classes=len(class_names),
        token_dim=128,
        num_heads=4,
        dropout=0.10,
        use_pretrained_backbone=False,
    )

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = _remove_module_prefix(_extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    MODEL_BUNDLE.model = model
    MODEL_BUNDLE.class_names = class_names
    MODEL_BUNDLE.device = device
    return MODEL_BUNDLE


def image_from_bytes(file_bytes: bytes) -> Image.Image:
    try:
        from io import BytesIO
        return Image.open(BytesIO(file_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc


@torch.no_grad()
def predict_pil_image(image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
    bundle = load_model_once()
    assert bundle.model is not None

    tensor = preprocess_image(image).to(bundle.device)
    logits = bundle.model(tensor)
    probabilities = F.softmax(logits, dim=1).squeeze(0).cpu()

    top_k = min(top_k, len(bundle.class_names))
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    predictions: List[Dict[str, Any]] = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label = bundle.class_names[idx]
        predictions.append({
            "class_name": label,
            "probability": round(float(prob), 6),
            "confidence_percent": round(float(prob) * 100, 2),
            "care_suggestion": CARE_SUGGESTIONS.get(
                label,
                "Consult a plant pathology or agriculture expert for confirmation.",
            ),
        })

    top = predictions[0]
    is_low_confidence = top["probability"] < CONFIDENCE_THRESHOLD
    return {
        "predicted_class": top["class_name"],
        "confidence_percent": top["confidence_percent"],
        "is_low_confidence": is_low_confidence,
        "confidence_threshold_percent": int(CONFIDENCE_THRESHOLD * 100),
        "care_suggestion": top["care_suggestion"],
        "top_predictions": predictions,
    }


def _summarize_frame_predictions(frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not frame_results:
        raise ValueError("No valid frames were analyzed from the video.")

    votes: Counter[str] = Counter()
    prob_sum: defaultdict[str, float] = defaultdict(float)

    for result in frame_results:
        top = result["top_predictions"][0]
        label = top["class_name"]
        votes[label] += 1
        prob_sum[label] += float(top["probability"])

    def rank_key(label: str) -> Tuple[int, float]:
        return votes[label], prob_sum[label] / votes[label]

    final_label = max(votes.keys(), key=rank_key)
    avg_prob = prob_sum[final_label] / votes[final_label]

    return {
        "predicted_class": final_label,
        "confidence_percent": round(avg_prob * 100, 2),
        "is_low_confidence": avg_prob < CONFIDENCE_THRESHOLD,
        "confidence_threshold_percent": int(CONFIDENCE_THRESHOLD * 100),
        "care_suggestion": CARE_SUGGESTIONS.get(
            final_label,
            "Consult a plant pathology or agriculture expert for confirmation.",
        ),
        "analyzed_frames": len(frame_results),
        "vote_summary": dict(votes),
        "frame_predictions": frame_results,
    }


def predict_video_file(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not read video. Try MP4 format with H.264 encoding.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps and fps > 0 else 0

    if duration > MAX_VIDEO_SECONDS:
        cap.release()
        raise ValueError(
            f"Video is too long: {duration:.2f} seconds. "
            f"Please upload a video up to {MAX_VIDEO_SECONDS} seconds only."
        )

    frame_no = 0
    analyzed = 0
    frame_results: List[Dict[str, Any]] = []

    while True:
        success, frame_bgr = cap.read()
        if not success:
            break

        if frame_no % FRAME_STEP == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            pred = predict_pil_image(image, top_k=3)
            pred["frame_number"] = frame_no
            frame_results.append(pred)
            analyzed += 1

            if analyzed >= MAX_FRAMES_TO_ANALYZE:
                break

        frame_no += 1

    cap.release()

    summary = _summarize_frame_predictions(frame_results)
    summary["video_info"] = {
        "duration_seconds": round(duration, 2),
        "total_frames": total_frames,
        "fps": round(float(fps), 2) if fps else None,
        "frame_step": FRAME_STEP,
        "max_frames_to_analyze": MAX_FRAMES_TO_ANALYZE,
        "max_video_seconds": MAX_VIDEO_SECONDS,
    }
    return summary


def save_upload_to_temp(upload_bytes: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload_bytes)
        return tmp.name
