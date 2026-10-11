import os
import zipfile


def sniff_format(file_path: str) -> str:
    """Determine file model format by checking magic bytes and content structure."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        header = f.read(16)

    # HDF5 format magic number: \x89HDF\r\n\x1a\n
    if header.startswith(b"\x89HDF\r\n\x1a\n"):
        return "h5"

    # Zip format magic number: PK\x03\x04
    if header.startswith(b"PK\x03\x04"):
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, "r") as z:
                names = z.namelist()
                if "config.json" in names or "metadata.json" in names:
                    return "keras_zip"
        return "keras_zip"

    # JSON text detection
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read(100).strip()
            if content.startswith("{") or content.startswith("["):
                return "json"
    except Exception:
        pass

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".keras":
        return "keras_zip"
    if ext == ".h5":
        return "h5"
    if ext == ".json":
        return "json"

    return "unknown"