"""FastAPI Application for NeuroScope."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from app.adapters.keras import KerasAdapter
from app.adapters.tree import TreeAdapter
from app.core.format_sniffer import sniff_format
from app.schemas import ModelView

app = FastAPI(title="NeuroScope Backend API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOADED_MODELS: Dict[str, Any] = {}


class PredictRequest(BaseModel):
    data: List[Any]


class WeightUpdateRequest(BaseModel):
    layer_name: str
    src_idx: int
    target_idx: int
    weight_value: float


class RetrainRequest(BaseModel):
    epochs: int = 1
    learning_rate: float = 0.001


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

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


@app.get("/api/models/{model_id}/tensors/{layer_name}")
def get_tensor_slice(
    model_id: str,
    layer_name: str,
    src_start: int = 0,
    src_end: int = 31,
    target_start: int = 0,
    target_end: int = 31,
):
    adapter = LOADED_MODELS.get(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Model not found")
    if hasattr(adapter, "get_tensor_slice"):
        return adapter.get_tensor_slice(layer_name, src_start, src_end, target_start, target_end)
    raise HTTPException(status_code=400, detail="Adapter does not support tensor slicing")


@app.get("/api/models/{model_id}/heatmap/{layer_name}")
def get_layer_heatmap(model_id: str, layer_name: str, resolution: int = 32):
    adapter = LOADED_MODELS.get(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Model not found")
    if hasattr(adapter, "get_heatmap_thumbnail"):
        return adapter.get_heatmap_thumbnail(layer_name, resolution)
    raise HTTPException(status_code=400, detail="Adapter does not support heatmap downsampling")


@app.get("/api/models/{model_id}/synapse-weight")
def get_synapse_weight(model_id: str, layer: str, src: int, target: int):
    adapter = LOADED_MODELS.get(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Model not found")
    if hasattr(adapter, "get_synapse_weight"):
        w = adapter.get_synapse_weight(layer, src, target)
        return {"weight": w}
    return {"weight": 0.0}


@app.post("/api/models/{model_id}/weights")
def update_model_weight(model_id: str, req: WeightUpdateRequest):
    adapter = LOADED_MODELS.get(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        old_val, impact = adapter.update_weight(
            req.layer_name, req.src_idx, req.target_idx, req.weight_value
        )
        return {
            "status": "success",
            "updated": req.dict(),
            "impact": impact,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed updating weight: {str(e)}")


@app.post("/api/models/{model_id}/retrain")
def retrain_model(model_id: str, req: RetrainRequest):
    adapter = LOADED_MODELS.get(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Model not found")
    if not hasattr(adapter, "retrain_step"):
        raise HTTPException(status_code=400, detail="Adapter does not support retraining")
    try:
        result = adapter.retrain_step(epochs=req.epochs, lr=req.learning_rate)
        return {"status": "success", "metrics": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")


@app.get("/api/models/{model_id}/export")
def export_model_file(model_id: str):
    adapter = LOADED_MODELS.get(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Model not found")
    if not hasattr(adapter, "export_model"):
        raise HTTPException(status_code=400, detail="Adapter does not support export")

    export_path = os.path.join(tempfile.gettempdir(), f"neuroscope_export_{model_id[:8]}.keras")
    adapter.export_model(export_path)
    return FileResponse(
        export_path,
        media_type="application/octet-stream",
        filename="neuroscope_model.keras",
    )


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