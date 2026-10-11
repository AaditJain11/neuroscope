import io, json, unittest, zipfile
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from app.formats import detect_format
from app.adapters.tree import TreeAdapter

def _zip(names):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n in names: z.writestr(n, "x")
    return b.getvalue()

class FormatTests(unittest.TestCase):
    def test_keras_zip(self): self.assertEqual(detect_format(_zip(["config.json","metadata.json"]), "a.keras").format, "keras")
    def test_h5(self): self.assertEqual(detect_format(b"\x89HDF\r\n\x1a\n"+b"0"*20, "m.h5").format, "keras_h5")
    def test_tflite(self): self.assertEqual(detect_format(b"\x1c\x00\x00\x00TFL3"+b"0"*20, "m.tflite").format, "tflite")
    def test_torch_pickle_refused_note(self):
        d = detect_format(_zip(["m/data.pkl"]), "m.pt"); self.assertEqual(d.format, "torch_pickle"); self.assertIn("Not loaded", d.note)
    def test_pickle(self): self.assertEqual(detect_format(b"\x80\x04abc", "m.pkl").format, "pickle")
    def test_extension_is_not_trusted(self): self.assertEqual(detect_format(b"hello world", "m.onnx").format, "unknown")
    def test_empty(self):
        with self.assertRaises(ValueError): detect_format(b"")

class TreeTests(unittest.TestCase):
    def setUp(self):
        X, y = load_iris(return_X_y=True); self.X = X
        self.est = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
        self.ad = TreeAdapter.from_sklearn(self.est)
    def test_predictions_match_sklearn(self):
        ours = self.ad.predict(self.X).argmax(1)
        self.assertTrue((ours == self.est.predict(self.X)).all())
    def test_path_matches_sklearn(self):
        x = self.X[100]; ref = self.est.decision_path(x.reshape(1,-1)).indices.tolist()
        self.assertEqual(self.ad.trace(x)["path"], ref)
    def test_graph_counts(self):
        v = self.ad.inspect(); self.assertEqual(len(v.nodes), self.est.tree_.node_count); self.assertEqual(len(v.edges), len(v.nodes)-1)
    def test_edit_changes_copy_only(self):
        orig = self.ad.parameters()["thresholds"].copy()
        new = self.ad.with_edit("threshold", (0,), 99.0)
        self.assertTrue((self.ad.parameters()["thresholds"] == orig).all())
        self.assertTrue((new.predict(self.X).argmax(1) != self.ad.predict(self.X).argmax(1)).any())
    def test_edit_leaf_rejected(self):
        leaf = int(np.where(self.ad.tree.left == -1)[0][0])
        with self.assertRaises(ValueError): self.ad.with_edit("threshold", (leaf,), 1.0)
    def test_json_roundtrip(self):
        t = self.ad.tree
        raw = json.dumps({"neuroscope":"tree-v1","left":t.left.tolist(),"right":t.right.tolist(),"feature":t.feature.tolist(),
                          "threshold":t.threshold.tolist(),"value":t.value.tolist()}).encode()
        self.assertTrue((TreeAdapter.from_json(raw).predict(self.X) == self.ad.predict(self.X)).all())
    def test_bad_json(self):
        with self.assertRaises(ValueError): TreeAdapter.from_json(b'{"a":1}')

if __name__ == "__main__": unittest.main()
