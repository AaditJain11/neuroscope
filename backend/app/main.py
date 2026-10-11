"""FastAPI Application for NeuroScope."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.adapters.keras import KerasAdapter
from app.adapters.tree import TreeAdapter
from app.core.format_sniffer import sniff_format
from app.schemas import ModelView

app = FastAPI(title="NeuroScope Backend API", version="0.1.0")

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for loaded adapters
LOADED_MODELS: Dict[str, Any] = {}
TEMP_FILES: List[str] = []


class PredictRequest(BaseModel):
    data: List[Any]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    TEMP_FILES.append(tmp_path)

    try:
        fmt = sniff_format(tmp_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=f"Sniffing error: {str(e)}")

    if fmt in ["keras_zip", "h5"]:
        try:
            adapter = KerasAdapter(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed loading Keras model: {str(e)}")
    elif fmt == "json":
        try:
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
            adapter = TreeAdapter(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed loading tree model: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

    model_id = str(uuid.uuid4())
    LOADED_MODELS[model_id] = adapter

    return {
        "model_id": model_id,
        "format": fmt,
        "filename": file.filename,
    }


@app.get("/api/models/{model_id}", response_model=ModelView)
def inspect_model(model_id: str):
    adapter = LOADED_MODELS.get(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        return adapter.inspect()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inspecting model: {str(e)}")


@app.post("/api/models/{model_id}/predict")
def predict_model(model_id: str, req: PredictRequest):
    adapter = LOADED_MODELS.get(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        preds = adapter.predict(req.data)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/models/{model_id}/activations")
def get_model_activations(model_id: str, req: PredictRequest):
    adapter = LOADED_MODELS.get(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        acts = adapter.get_activations(req.data)
        return {"activations": acts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Activation error: {str(e)}")