"""Adapter for Keras models (.keras, .h5)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import tensorflow as tf

from app.adapters.base import Adapter
from app.schema import (
    CAPABILITIES,
    SUPPORTED,
    UNSUPPORTED,
    Capability,
    Edge,
    ModelView,
    Node,
)


class KerasAdapter(Adapter):
    format_ids = ("keras", "keras_h5")

    def __init__(self, model: tf.keras.Model, name: str = "Keras Model") -> None:
        self.model = model
        self.name = name

    @classmethod
    def from_bytes(cls, data: bytes, filename: str, fmt: str) -> "KerasAdapter":
        suffix = ".keras" if fmt == "keras" else ".h5"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        try:
            model = tf.keras.models.load_model(str(tmp_path), compile=False)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return cls(model=model, name=filename)

    def _capabilities(self) -> dict[str, Capability]:
        caps = {c: Capability(status=UNSUPPORTED, why="Not implemented") for c in CAPABILITIES}
        caps["architecture"] = Capability(status=SUPPORTED, why="Keras layer hierarchy")
        caps["graph"] = Capability(status=SUPPORTED, why="Computation flow")
        caps["parameters"] = Capability(status=SUPPORTED, why="Layer weights and biases")
        caps["inference"] = Capability(status=SUPPORTED, why="tf.keras forward pass")
        caps["intermediates"] = Capability(status=SUPPORTED, why="Layer activation trace")
        caps["param_editing"] = Capability(status=UNSUPPORTED, why="Weight mutation not yet enabled")
        caps["retraining"] = Capability(status=UNSUPPORTED, why="Retraining not supported")
        caps["export"] = Capability(status=UNSUPPORTED, why="Exporting not supported")
        return caps

    def inspect(self) -> ModelView:
        nodes: list[Node] = []
        edges: list[Edge] = []

        total_params = int(self.model.count_params())
        trainable_params = int(sum(np.prod(w.shape) for w in self.model.trainable_weights))
        non_trainable_params = total_params - trainable_params

        prev_node_id: Optional[str] = None
        for idx, layer in enumerate(self.model.layers):
            node_id = f"layer_{idx}_{layer.name}"
            layer_type = layer.__class__.__name__

            shape = None
            try:
                if hasattr(layer, "output_shape"):
                    shape = layer.output_shape
                elif hasattr(layer, "output"):
                    shape = [int(d) if d is not None else None for d in layer.output.shape]
            except Exception:
                pass

            weights_meta = {}
            for weight in layer.weights:
                weights_meta[weight.name] = {
                    "shape": list(weight.shape),
                    "param_count": int(np.prod(weight.shape)),
                }

            nodes.append(
                Node(
                    id=node_id,
                    label=layer.name,
                    type=layer_type,
                    attrs={
                        "shape": shape,
                        "trainable": layer.trainable,
                        "weights": weights_meta,
                    },
                )
            )

            if prev_node_id is not None:
                edges.append(Edge(source=prev_node_id, target=node_id))
            prev_node_id = node_id

        input_shapes = []
        try:
            if hasattr(self.model, "inputs") and self.model.inputs:
                input_shapes = [[int(d) if d is not None else None for d in inp.shape] for inp in self.model.inputs]
        except Exception:
            pass

        output_shapes = []
        try:
            if hasattr(self.model, "outputs") and self.model.outputs:
                output_shapes = [[int(d) if d is not None else None for d in out.shape] for out in self.model.outputs]
        except Exception:
            pass

        return ModelView(
            kind="dense",
            name=self.name,
            nodes=nodes,
            edges=edges,
            meta={
                "total_params": total_params,
                "trainable_params": trainable_params,
                "non_trainable_params": non_trainable_params,
                "input_shapes": input_shapes,
                "output_shapes": output_shapes,
            },
            capabilities=self._capabilities(),
        )

    def parameters(self) -> dict[str, np.ndarray]:
        params = {}
        for layer in self.model.layers:
            for w in layer.weights:
                params[w.name] = w.numpy()
        return params

    def predict(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1 and hasattr(self.model, "input_shape"):
            expected = self.model.input_shape
            if expected and len(expected) == 2 and arr.shape[0] == expected[1]:
                arr = np.expand_dims(arr, axis=0)

        out = self.model(arr, training=False)
        if isinstance(out, (list, tuple)):
            return np.concatenate([o.numpy() for o in out], axis=-1)
        return out.numpy()

    def trace(self, x: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1 and hasattr(self.model, "input_shape"):
            expected = self.model.input_shape
            if expected and len(expected) == 2 and arr.shape[0] == expected[1]:
                arr = np.expand_dims(arr, axis=0)

        layer_names = [layer.name for layer in self.model.layers]
        layer_outputs = [layer.output for layer in self.model.layers]
        submodel = tf.keras.Model(inputs=self.model.inputs, outputs=layer_outputs)

        outputs = submodel(arr, training=False)
        if not isinstance(outputs, list):
            outputs = [outputs]

        return {name: out.numpy().tolist() for name, out in zip(layer_names, outputs)}