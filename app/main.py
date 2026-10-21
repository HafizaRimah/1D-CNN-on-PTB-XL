"""
FastAPI Backend for PTB-XL 12-Lead ECG Diagnostic Web App.
Features:
- File upload (CSV, NPY, JSON, TXT)
- Pre-loaded clinical test cases (NORM, MI, HYP, CD)
- Multi-label prediction (NORM, MI, STTC, CD, HYP)
- Dynamic per-lead attention extraction (interpretable AI)
- Fully styled with user color palette
"""

import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.samples import STANDARD_LEAD_NAMES, generate_synthetic_ecg
from src.dataset import SUPERCLASSES
from src.models import EarlyFusionCNN, LeadAttentionCNN

app = FastAPI(title="PTB-XL 12-Lead ECG Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
device = torch.device("cpu")
early_fusion_model = EarlyFusionCNN(in_channels=12, num_classes=5).to(device)
lead_attention_model = LeadAttentionCNN(num_leads=12, num_classes=5).to(device)

# Load checkpoints if available
ckpt_ef = ROOT_DIR / "checkpoints/early_fusion_1d_cnn_(baseline)_seed_42.pt"
ckpt_la = ROOT_DIR / "checkpoints/lead-wise_attention_1d_cnn_variant_seed_42.pt"

if ckpt_ef.exists():
    try:
        early_fusion_model.load_state_dict(torch.load(ckpt_ef, map_location=device))
        print("Loaded EarlyFusionCNN checkpoint.")
    except Exception as e:
        print(f"Notice: initialized EarlyFusionCNN with base weights ({e})")

if ckpt_la.exists():
    try:
        lead_attention_model.load_state_dict(torch.load(ckpt_la, map_location=device))
        print("Loaded LeadAttentionCNN checkpoint.")
    except Exception as e:
        print(f"Notice: initialized LeadAttentionCNN with base weights ({e})")

early_fusion_model.eval()
lead_attention_model.eval()


def parse_ecg_file(file_bytes: bytes, filename: str) -> np.ndarray:
    """Parses an uploaded ECG file into a (12, 1000) numpy array."""
    ext = filename.lower().split(".")[-1]

    if ext == "npy":
        arr = np.load(io.BytesIO(file_bytes))
    elif ext in ["csv", "txt"]:
        # Try comma, tab, or whitespace delimiter
        try:
            arr = np.loadtxt(io.BytesIO(file_bytes), delimiter=",")
        except Exception:
            arr = np.loadtxt(io.BytesIO(file_bytes))
    elif ext == "json":
        data = json.loads(file_bytes.decode("utf-8"))
        if isinstance(data, dict) and "signal" in data:
            arr = np.array(data["signal"])
        else:
            arr = np.array(data)
    else:
        raise ValueError(f"Unsupported file format '.{ext}'. Use .csv, .npy, .json, or .txt.")

    # Shape sanity check
    if arr.ndim != 2:
        raise ValueError(f"ECG array must be 2D, but got shape {arr.shape}.")

    # Transpose if given as (time_steps, leads) e.g. (1000, 12)
    if arr.shape[0] != 12 and arr.shape[1] == 12:
        arr = arr.T

    # Resize or truncate to 1000 time steps if needed
    if arr.shape[0] != 12:
        raise ValueError(f"Expected 12 leads, but found {arr.shape[0]} channels.")

    if arr.shape[1] < 100:
        raise ValueError(f"Signal is too short ({arr.shape[1]} samples). Minimum is 100 samples.")

    if arr.shape[1] != 1000:
        # Interpolate to 1000 samples
        from scipy.ndimage import zoom
        zoom_factor = 1000.0 / arr.shape[1]
        arr = zoom(arr, (1, zoom_factor))[:, :1000]

    return arr.astype(np.float32)


def standardize_signal(signal: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance per lead."""
    mean = np.mean(signal, axis=1, keepdims=True)
    std = np.std(signal, axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (signal - mean) / std


@app.post("/api/predict")
async def predict_ecg(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
):
    try:
        if file is not None and file.filename != "":
            contents = await file.read()
            raw_signal = parse_ecg_file(contents, file.filename)
            source_name = file.filename
        elif sample_id:
            raw_signal = generate_synthetic_ecg(condition=sample_id)
            source_name = f"Preloaded Case: {sample_id}"
        else:
            # Default to NORM sample
            raw_signal = generate_synthetic_ecg(condition="NORM")
            source_name = "Default Sample (NORM)"

        # Standardize for neural network
        norm_signal = standardize_signal(raw_signal)

        # Tensor conversion: (1, 12, 1000)
        x_tensor = torch.from_numpy(norm_signal).unsqueeze(0).to(device)

        with torch.no_grad():
            # Lead-Attention Model
            logits_attn, attn_weights = lead_attention_model(x_tensor, return_attention=True)
            probs_attn = torch.sigmoid(logits_attn).squeeze(0).cpu().numpy()
            attn_weights = attn_weights.squeeze(0).cpu().numpy()

            # Early Fusion Baseline
            logits_ef = early_fusion_model(x_tensor)
            probs_ef = torch.sigmoid(logits_ef).squeeze(0).cpu().numpy()

        # Formatting results
        results_attn = {cls: float(np.round(p, 4)) for cls, p in zip(SUPERCLASSES, probs_attn)}
        results_ef = {cls: float(np.round(p, 4)) for cls, p in zip(SUPERCLASSES, probs_ef)}

        lead_attn_dict = {
            lead: float(np.round(w, 4)) for lead, w in zip(STANDARD_LEAD_NAMES, attn_weights)
        }

        # Downsample waveform for snappy frontend rendering (e.g. 500 points)
        downsampled = raw_signal[:, ::2].tolist()

        return {
            "status": "success",
            "source": source_name,
            "classes": SUPERCLASSES,
            "predictions": {
                "lead_attention": results_attn,
                "early_fusion": results_ef,
            },
            "lead_attention_weights": lead_attn_dict,
            "waveforms": {
                "leads": STANDARD_LEAD_NAMES,
                "data": downsampled,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = Path(__file__).resolve().parent / "static" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Index HTML not found.</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
