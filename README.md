# NeuroScope (work in progress)

Status: backend core only. Verified: format detection, tree adapter (14 unit tests).
Not built yet: FastAPI layer, Keras/ONNX adapters, React frontend.

## Run tests (PowerShell, from neuroscope\backend)
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install numpy scikit-learn
    python -m unittest discover -s tests -t .
