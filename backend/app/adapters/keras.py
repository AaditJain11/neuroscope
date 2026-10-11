"""Keras/HDF5 adapter implementing canonical ModelAdapter interface."""

from __future__ import annotations

import os
from typing import Any, Dict, List
import numpy as np

from app.adapters.base import Adapter
from app.schemas import ModelView, ModelNode, ModelEdge, Capability


class KerasAdapter(Adapter):
    """Adapter for inspecting and executing TensorFlow Keras models."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        import tensorflow as tf

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        self.model = tf.keras.models.load_model(self.model_path)

    def inspect(self) -> ModelView:
        """Parse Keras graph into canonical ModelView schema."""
        nodes: List[ModelNode] = []
        edges: List[ModelEdge] = []

        total_params = int(self.model.count_params())
        trainable_params = int(
            sum(
                np.prod(p.shape)
                for p in self.model.trainable_weights
            )
        )
        non_trainable_params = total_params - trainable_params

        for idx, layer in enumerate(self.model.layers):
            node_id = f"layer_{idx}_{layer.name}"

            # Extract weights and raw numeric matrices for individual neuron synapses
            weights_meta: Dict[str, Any] = {}
            for weight in layer.weights:
                val = weight.numpy()
                weights_meta[weight.name] = {
                    "shape": list(weight.shape),
                    "param_count": int(np.prod(weight.shape)),
                    "values": val.tolist() if val.size <= 5000 else None,
                }

            # Accurately extract layer dimensions and units
            shape_info = None
            units = getattr(layer, "units", None)
            try:
                if hasattr(layer, "output_shape") and layer.output_shape is not None:
                    shape_info = [None if d is None else int(d) for d in layer.output_shape]
            except Exception:
                pass

            if not shape_info and units is not None:
                shape_info = [None, int(units)]

            nodes.append(
                ModelNode(
                    id=node_id,
                    label=layer.name,
                    type=layer.__class__.__name__,
                    attrs={
                        "shape": shape_info,
                        "units": units,
                        "trainable": layer.trainable,
                        "weights": weights_meta,
                    },
                )
            )

        # Build sequential connections
        for i in range(len(nodes) - 1):
            edges.append(
                ModelEdge(
                    source=nodes[i].id,
                    target=nodes[i + 1].id,
                )
            )

        # Extract input shapes
        input_shapes = []
        try:
            if hasattr(self.model, "input_shape") and self.model.input_shape:
                input_shapes.append([None if d is None else int(d) for d in self.model.input_shape])
        except Exception:
            pass

        return ModelView(
            kind="dense",
            name=os.path.basename(self.model_path),
            nodes=nodes,
            edges=edges,
            meta={
                "total_params": total_params,
                "trainable_params": trainable_params,
                "non_trainable_params": non_trainable_params,
                "input_shapes": input_shapes,
            },
            capabilities={
                "architecture": Capability(status="supported", why="Full graph topology parsed"),
                "parameters": Capability(status="supported", why="Layer tensors extracted"),
                "graph": Capability(status="supported", why="DAG connection list ready"),
                "inference": Capability(status="supported", why="Forward pass enabled"),
                "intermediates": Capability(status="supported", why="Feature map tracing supported"),
                "param_editing": Capability(status="unsupported", why="Immutable in current release"),
                "retraining": Capability(status="unsupported", why="Dataset binding not implemented"),
                "export": Capability(status="unsupported", why="Model export disabled"),
            },
        )

    def predict(self, input_data: Any) -> Any:
        """Run batch forward inference."""
        arr = np.array(input_data, dtype=np.float32)
        predictions = self.model.predict(arr, verbose=0)
        return predictions.tolist()

    def get_activations(self, input_data: Any) -> Dict[str, Any]:
        """Extract intermediate activations for every layer."""
        import tensorflow as tf

        arr = np.array(input_data, dtype=np.float32)
        activations_dict: Dict[str, Any] = {}

        layer_outputs = [layer.output for layer in self.model.layers]
        activation_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=layer_outputs,
        )

        results = activation_model.predict(arr, verbose=0)
        if not isinstance(results, list):
            results = [results]

        for layer, output_tensor in zip(self.model.layers, results):
            activations_dict[layer.name] = output_tensor.tolist()

        return activations_dict