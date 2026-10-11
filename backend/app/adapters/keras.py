"""Keras/HDF5 adapter implementing chunked streaming, evaluations, and retraining."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple
import numpy as np

from app.adapters.base import Adapter
from app.schemas import ModelView, ModelNode, ModelEdge, Capability


class KerasAdapter(Adapter):
    """Adapter for inspecting, executing, editing, and streaming Keras models."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = None
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.baseline_metrics: Dict[str, float] = {}
        self._load_model()
        self._prepare_benchmark_dataset()

    def _load_model(self) -> None:
        import tensorflow as tf

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        self.model = tf.keras.models.load_model(self.model_path)

    def _prepare_benchmark_dataset(self) -> None:
        """Create separate train and validation datasets to guarantee measurable learning."""
        import tensorflow as tf

        in_shape = self.model.input_shape
        if isinstance(in_shape, list):
            in_shape = in_shape[0]

        out_shape = self.model.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        out_dim = out_shape[-1] if out_shape and out_shape[-1] is not None else 10

        loaded = False
        is_mnist = in_shape and len(in_shape) >= 2 and in_shape[-1] in [784, 28]

        if is_mnist:
            try:
                (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
                x_tr = (x_tr / 255.0).astype(np.float32)
                x_te = (x_te / 255.0).astype(np.float32)
                if in_shape[-1] == 784:
                    x_tr = x_tr.reshape(-1, 784)
                    x_te = x_te.reshape(-1, 784)
                self.train_x = x_tr[:1500]
                self.train_y = y_tr[:1500]
                self.val_x = x_te[:500]
                self.val_y = y_te[:500]
                loaded = True
            except Exception:
                loaded = False

        if not loaded:
            # Deterministic, structured multi-class pattern dataset
            np.random.seed(42)
            n_samples = 1500
            flat_dim = int(np.prod([d for d in in_shape[1:] if d is not None]))

            centroids = np.random.randn(out_dim, flat_dim).astype(np.float32) * 2.0
            y_all = np.random.randint(0, out_dim, size=(n_samples,))
            x_all = centroids[y_all] + np.random.randn(n_samples, flat_dim).astype(np.float32) * 0.8

            target_shape = [n_samples] + [d if d is not None else 28 for d in in_shape[1:]]
            x_all = x_all.reshape(target_shape)

            self.train_x = x_all[:1000]
            self.train_y = y_all[:1000]
            self.val_x = x_all[1000:1500]
            self.val_y = y_all[1000:1500]

        self._ensure_compiled()
        self.baseline_metrics = self.evaluate()

    def _ensure_compiled(self) -> None:
        import tensorflow as tf

        if not hasattr(self.model, "optimizer") or self.model.optimizer is None:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=["accuracy"],
            )

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model against validation benchmark."""
        self._ensure_compiled()
        res = self.model.evaluate(self.val_x, self.val_y, batch_size=64, verbose=0)
        loss = float(res[0]) if isinstance(res, list) else float(res)
        acc = float(res[1]) if isinstance(res, list) and len(res) > 1 else 0.0
        return {
            "loss": round(loss, 5),
            "accuracy": round(acc, 5),
        }

    def get_synapse_weight(self, layer_name: str, src_idx: int, target_idx: int) -> float:
        for layer in self.model.layers:
            if layer.name == layer_name:
                weights = layer.get_weights()
                if len(weights) > 0:
                    kernel = weights[0]
                    if src_idx < kernel.shape[0] and target_idx < kernel.shape[1]:
                        return float(kernel[src_idx, target_idx])
        return 0.0

    def get_tensor_slice(
        self,
        layer_name: str,
        src_start: int,
        src_end: int,
        target_start: int,
        target_end: int,
    ) -> Dict[str, Any]:
        for layer in self.model.layers:
            if layer.name == layer_name:
                weights = layer.get_weights()
                if len(weights) > 0:
                    kernel = weights[0]
                    rows, cols = kernel.shape
                    s_start = max(0, min(src_start, rows - 1))
                    s_end = max(s_start + 1, min(src_end + 1, rows))
                    t_start = max(0, min(target_start, cols - 1))
                    t_end = max(t_start + 1, min(target_end + 1, cols))

                    sliced = kernel[s_start:s_end, t_start:t_end]
                    return {
                        "shape": [rows, cols],
                        "slice_shape": list(sliced.shape),
                        "src_range": [s_start, s_end - 1],
                        "target_range": [t_start, t_end - 1],
                        "values": sliced.tolist(),
                    }
        return {"error": "Layer or weights not found"}

    def get_heatmap_thumbnail(self, layer_name: str, resolution: int = 32) -> Dict[str, Any]:
        for layer in self.model.layers:
            if layer.name == layer_name:
                weights = layer.get_weights()
                if len(weights) > 0:
                    kernel = weights[0]
                    rows, cols = kernel.shape
                    step_r = max(1, rows // resolution)
                    step_c = max(1, cols // resolution)
                    sub = kernel[::step_r, ::step_c][:resolution, :resolution]
                    return {
                        "shape": [rows, cols],
                        "res": list(sub.shape),
                        "min": float(np.min(kernel)),
                        "max": float(np.max(kernel)),
                        "mean": float(np.mean(kernel)),
                        "grid": sub.tolist(),
                    }
        return {"error": "Layer not found"}

    def inspect(self) -> ModelView:
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

            shape_info = None
            units = getattr(layer, "units", None)
            try:
                if hasattr(layer, "output_shape") and layer.output_shape is not None:
                    out = layer.output_shape
                    if isinstance(out, list) and len(out) > 0:
                        out = out[0]
                    shape_info = [None if d is None else int(d) for d in out]
                    if units is None and len(shape_info) >= 2 and shape_info[-1] is not None:
                        non_none = [d for d in shape_info[1:] if d is not None]
                        units = int(np.prod(non_none)) if len(non_none) > 0 else None
            except Exception:
                pass

            if not shape_info and units is not None:
                shape_info = [None, int(units)]

            weights_meta: Dict[str, Any] = {}
            for weight in layer.weights:
                weights_meta[weight.name] = {
                    "shape": list(weight.shape),
                    "param_count": int(np.prod(weight.shape)),
                }

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

        for i in range(len(nodes) - 1):
            edges.append(
                ModelEdge(
                    source=nodes[i].id,
                    target=nodes[i + 1].id,
                )
            )

        input_shapes = []
        try:
            if hasattr(self.model, "input_shape") and self.model.input_shape:
                inp = self.model.input_shape
                if isinstance(inp, list) and len(inp) > 0:
                    inp = inp[0]
                input_shapes.append([None if d is None else int(d) for d in inp])
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
                "baseline_accuracy": self.baseline_metrics.get("accuracy", 0.0),
                "baseline_loss": self.baseline_metrics.get("loss", 0.0),
            },
            capabilities={
                "architecture": Capability(status="supported", why="Full topology parsed"),
                "parameters": Capability(status="supported", why="On-demand streaming tensors"),
                "graph": Capability(status="supported", why="DAG connection list ready"),
                "inference": Capability(status="supported", why="Forward pass enabled"),
                "intermediates": Capability(status="supported", why="Activations traceable"),
                "param_editing": Capability(status="supported", why="Synapse weight mutation with live metric delta"),
                "retraining": Capability(status="supported", why="Fine-tuning with before/after metric verification"),
                "export": Capability(status="supported", why="Export .keras supported"),
            },
        )

    def update_weight(
        self, layer_name: str, src_idx: int, target_idx: int, new_val: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Mutate a connection weight and return evaluation delta alongside prediction shift."""
        # Record baseline predictions on sample batch
        preds_before = self.model.predict(self.val_x[:100], verbose=0)

        old_val = 0.0
        found = False
        for layer in self.model.layers:
            if layer.name == layer_name:
                weights = layer.get_weights()
                if len(weights) > 0:
                    kernel = weights[0]
                    if src_idx < kernel.shape[0] and target_idx < kernel.shape[1]:
                        old_val = float(kernel[src_idx, target_idx])
                        kernel[src_idx, target_idx] = float(new_val)
                        weights[0] = kernel
                        layer.set_weights(weights)
                        found = True
                        break

        if not found:
            raise ValueError(f"Target connection ({src_idx} -> {target_idx}) not found in layer '{layer_name}'")

        # Measure output confidence shift across test batch
        preds_after = self.model.predict(self.val_x[:100], verbose=0)
        output_shift = float(np.mean(np.abs(preds_after - preds_before)))

        current_metrics = self.evaluate()
        delta_acc = current_metrics["accuracy"] - self.baseline_metrics.get("accuracy", 0.0)
        delta_loss = current_metrics["loss"] - self.baseline_metrics.get("loss", 0.0)

        impact = {
            "current_accuracy": current_metrics["accuracy"],
            "current_loss": current_metrics["loss"],
            "delta_accuracy": round(delta_acc, 5),
            "delta_loss": round(delta_loss, 5),
            "output_shift": round(output_shift, 5),
            "baseline_accuracy": self.baseline_metrics.get("accuracy", 0.0),
            "baseline_loss": self.baseline_metrics.get("loss", 0.0),
            "old_weight": round(old_val, 5),
            "new_weight": round(new_val, 5),
        }
        return old_val, impact

    def retrain_step(self, epochs: int = 1, lr: float = 0.001) -> Dict[str, Any]:
        """Perform fine-tuning training loop step and return validated metric improvements."""
        import tensorflow as tf

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        before = self.evaluate()
        history = self.model.fit(
            self.train_x, self.train_y, epochs=epochs, batch_size=32, verbose=0
        )
        after = self.evaluate()

        return {
            "epochs": epochs,
            "learning_rate": lr,
            "train_loss": float(history.history["loss"][-1]),
            "train_accuracy": float(history.history.get("accuracy", [0.0])[-1]),
            "before_accuracy": before["accuracy"],
            "before_loss": before["loss"],
            "after_accuracy": after["accuracy"],
            "after_loss": after["loss"],
            "delta_accuracy": round(after["accuracy"] - before["accuracy"], 5),
            "delta_loss": round(after["loss"] - before["loss"], 5),
        }

    def export_model(self, dest_path: str) -> str:
        self.model.save(dest_path)
        return dest_path

    def predict(self, input_data: Any) -> Any:
        arr = np.array(input_data, dtype=np.float32)
        predictions = self.model.predict(arr, verbose=0)
        return predictions.tolist()

    def get_activations(self, input_data: Any) -> Dict[str, Any]:
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