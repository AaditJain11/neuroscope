"""Format detection from file CONTENT (not just the extension). Never loads/unpickles anything."""
import io, zipfile
from dataclasses import dataclass, field

MAX_UPLOAD_BYTES = 200 * 1024 * 1024

@dataclass
class Detection:
    format: str                 # e.g. "keras", "onnx", "unknown"
    confidence: str             # "certain" | "likely" | "none"
    note: str = ""

def detect_format(data: bytes, filename: str = "") -> Detection:
    if len(data) == 0:
        raise ValueError("Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("File exceeds the upload size limit.")
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if data[:8] == b"\x89HDF\r\n\x1a\n":
        return Detection("keras_h5", "certain", "HDF5 container (legacy Keras .h5 if it holds a model).")
    if data[4:8] == b"TFL3":
        return Detection("tflite", "certain", "TensorFlow Lite flatbuffer.")
    if data[:4] == b"PK\x03\x04":
        try:
            names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
        except zipfile.BadZipFile:
            return Detection("unknown", "none", "Corrupt ZIP archive.")
        if {"config.json", "metadata.json"} <= names:
            return Detection("keras", "certain", "Keras v3 archive.")
        if any(n.endswith("data.pkl") for n in names):
            if any(n.endswith("constants.pkl") or n.endswith("code/__torch__.py") or "/code/" in n for n in names):
                return Detection("torchscript", "likely", "TorchScript archive.")
            return Detection("torch_pickle", "certain",
                             "PyTorch pickle archive. Not loaded: pickles can run arbitrary code.")
        return Detection("unknown", "none", "ZIP archive with unrecognised contents.")
    if data[:1] == b"{" and ext == "json":
        return Detection("json_tree", "likely", "JSON; checked by the tree adapter.")
    if ext == "onnx" and data[:1] in (b"\x08", b"\x3a", b"\x12"):
        return Detection("onnx", "likely", "ONNX protobuf (no magic number; validated on load).")
    if data[:2] in (b"\x80\x04", b"\x80\x05", b"\x80\x03", b"\x80\x02"):
        return Detection("pickle", "certain", "Python pickle (sklearn/joblib). Refused: unsafe to load.")
    return Detection("unknown", "none", "Unrecognised format.")
