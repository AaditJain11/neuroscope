"""NeuroScope FastAPI Backend Server."""

from __future__ import annotations

import uuid
from typing import Any, Dict
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.adapters.keras import KerasAdapter
from app.adapters.tree import TreeAdapter
from app.formats import detect_format

app = FastAPI(title="NeuroScope API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_store: Dict[str, Any] = {}


class InferenceRequest(BaseModel):
    data: Any


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "NeuroScope"}


@app.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        detection = detect_format(contents, file.filename or "")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    fmt = detection.format
    if fmt in ("keras", "keras_h5"):
        try:
            adapter = KerasAdapter.from_bytes(contents, file.filename or "model", fmt)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse Keras model: {str(e)}")
    elif fmt == "json_tree":
        try:
            adapter = TreeAdapter.from_json(contents.decode("utf-8"))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse Tree model: {str(e)}")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Refused or unsupported format '{fmt}': {detection.note}",
        )

    model_id = str(uuid.uuid4())
    models_store[model_id] = adapter

    return {
        "model_id": model_id,
        "format": fmt,
        "filename": file.filename,
        "note": detection.note,
    }


@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    adapter = models_store.get(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Model not found.")
    return adapter.inspect().to_dict()


@app.post("/api/models/{model_id}/predict")
def run_predict(model_id: str, req: InferenceRequest):
    adapter = models_store.get(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Model not found.")

    try:
        arr = np.array(req.data, dtype=np.float32)
        preds = adapter.predict(arr)
        return {"predictions": preds.tolist()}
    except NotImplementedError:
        raise HTTPException(status_code=400, detail="Inference not supported for this model.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


@app.post("/api/models/{model_id}/activations")
def extract_activations(model_id: str, req: InferenceRequest):
    adapter = models_store.get(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Model not found.")

    try:
        arr = np.array(req.data, dtype=np.float32)
        trace_data = adapter.trace(arr)
        return {"activations": trace_data}
    except NotImplementedError:
        raise HTTPException(status_code=400, detail="Activation tracing not supported for this model.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tracing error: {str(e)}")