"""Decision-tree adapter (scikit-learn DecisionTree/RandomForest-style estimators).
Works on an in-memory estimator, or on a safe JSON export (no pickle)."""
import json
import numpy as np
from .base import Adapter
from ..schema import *

class _Tree:
    """Plain-array tree, independent of sklearn so it can come from JSON."""
    def __init__(self, left, right, feature, threshold, value):
        self.left, self.right = np.asarray(left), np.asarray(right)
        self.feature, self.threshold = np.asarray(feature), np.asarray(threshold, dtype=float)
        self.value = np.asarray(value, dtype=float)       # shape (n_nodes, n_outputs)
    def is_leaf(self, i): return self.left[i] == -1
    def predict_one(self, x):
        return self.value[self.path(x)[-1]]
    def path(self, x):
        i, p = 0, [0]
        while not self.is_leaf(i):
            i = int(self.left[i] if x[self.feature[i]] <= self.threshold[i] else self.right[i])
            p.append(i)
        return p

class TreeAdapter(Adapter):
    format_ids = ("json_tree", "sklearn_tree")

    def __init__(self, tree: _Tree, feature_names=None, name="decision_tree"):
        self.tree, self.name = tree, name
        nf = int(tree.feature.max()) + 1 if (tree.feature >= 0).any() else 1
        self.feature_names = feature_names or [f"x{i}" for i in range(nf)]

    @classmethod
    def from_sklearn(cls, est, feature_names=None):
        t = est.tree_
        return cls(_Tree(t.children_left, t.children_right, t.feature, t.threshold, t.value[:, 0, :]),
                   feature_names, type(est).__name__)

    @classmethod
    def from_json(cls, raw: bytes):
        d = json.loads(raw)
        if d.get("neuroscope") != "tree-v1":
            raise ValueError("Not a NeuroScope tree-v1 JSON file.")
        return cls(_Tree(d["left"], d["right"], d["feature"], d["threshold"], d["value"]),
                   d.get("feature_names"), d.get("name", "decision_tree"))

    def inspect(self) -> ModelView:
        t, nodes, edges = self.tree, [], []
        for i in range(len(t.left)):
            if t.is_leaf(i):
                nodes.append(Node(str(i), f"leaf {i}", "leaf", {"value": t.value[i].tolist()}))
            else:
                f = self.feature_names[int(t.feature[i])]
                nodes.append(Node(str(i), f"{f} <= {t.threshold[i]:.4g}", "split",
                                  {"feature": f, "threshold": float(t.threshold[i])}))
                edges += [Edge(str(i), str(int(t.left[i]))), Edge(str(i), str(int(t.right[i])))]
        caps = {
            "architecture": Capability(SUPPORTED, "Real split structure."),
            "parameters": Capability(SUPPORTED, "Thresholds and leaf values."),
            "graph": Capability(SUPPORTED, "Tree view."),
            "inference": Capability(SUPPORTED, "Traverses the stored tree."),
            "intermediates": Capability(SUPPORTED, "Decision path per sample."),
            "param_editing": Capability(SUPPORTED, "Thresholds/leaf values in a copy."),
            "retraining": Capability(UNSUPPORTED, "Not implemented."),
            "export": Capability(UNSUPPORTED, "Not implemented."),
        }
        return ModelView("tree", self.name, nodes, edges, {"n_nodes": len(nodes)}, caps)

    def parameters(self):
        m = self.tree.left != -1
        return {"thresholds": self.tree.threshold[m], "leaf_values": self.tree.value[~m]}

    def predict(self, x):
        x = np.atleast_2d(x)
        return np.stack([self.tree.predict_one(r) for r in x])

    def trace(self, x):
        p = self.tree.path(np.asarray(x, dtype=float))
        return {"path": p, "steps": [
            {"node": i, "feature": self.feature_names[int(self.tree.feature[i])],
             "value": float(x[self.tree.feature[i]]), "threshold": float(self.tree.threshold[i]),
             "went": "left" if x[self.tree.feature[i]] <= self.tree.threshold[i] else "right"}
            for i in p[:-1]]}

    def with_edit(self, name, index, value):
        t = self.tree
        new = _Tree(t.left.copy(), t.right.copy(), t.feature.copy(), t.threshold.copy(), t.value.copy())
        if name == "threshold":
            if new.is_leaf(index[0]): raise ValueError("Leaf nodes have no threshold.")
            new.threshold[index[0]] = value
        elif name == "leaf_value":
            if not new.is_leaf(index[0]): raise ValueError("Only leaves have values.")
            new.value[index[0], index[1]] = value
        else:
            raise ValueError("Unknown parameter.")
        return TreeAdapter(new, self.feature_names, self.name)
