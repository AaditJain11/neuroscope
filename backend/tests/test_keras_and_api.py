import numpy as np
import pytest
import tensorflow as tf
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture
def sample_keras_h5(tmp_path):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(8, activation="relu", name="hidden_layer"),
            tf.keras.layers.Dense(2, activation="softmax", name="output_layer"),
        ]
    )
    file_path = tmp_path / "test_model.h5"
    model.save(str(file_path))
    return file_path


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_keras_upload_inspect_inference(sample_keras_h5):
    with open(sample_keras_h5, "rb") as f:
        upload_res = client.post(
            "/api/models/upload",
            files={"file": ("test_model.h5", f, "application/octet-stream")},
        )
    assert upload_res.status_code == 200
    data = upload_res.json()
    assert "model_id" in data
    model_id = data["model_id"]

    # Inspect architecture
    view_res = client.get(f"/api/models/{model_id}")
    assert view_res.status_code == 200
    view = view_res.json()
    assert len(view["nodes"]) >= 2
    assert view["capabilities"]["inference"]["status"] == "supported"
    assert view["capabilities"]["intermediates"]["status"] == "supported"

    # Run inference
    dummy_input = [[0.5, -1.2, 0.3, 2.1]]
    pred_res = client.post(
        f"/api/models/{model_id}/predict", json={"data": dummy_input}
    )
    assert pred_res.status_code == 200
    preds = pred_res.json()["predictions"]
    assert len(preds) == 1
    assert len(preds[0]) == 2
    assert np.isclose(sum(preds[0]), 1.0, atol=1e-4)

    # Extract activations
    act_res = client.post(
        f"/api/models/{model_id}/activations", json={"data": dummy_input}
    )
    assert act_res.status_code == 200
    acts = act_res.json()["activations"]
    assert "hidden_layer" in acts
    assert len(acts["hidden_layer"][0]) == 8