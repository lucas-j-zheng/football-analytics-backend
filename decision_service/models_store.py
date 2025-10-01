from typing import Any, Dict, Tuple
from joblib import dump, load
import os


MODELS_DIR = os.getenv("DECISION_MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_model(name: str, version: str, model: Any, calibrator: Any, metadata: Dict) -> str:
    ensure_dir(MODELS_DIR)
    fname = f"{name}__{version}.joblib"
    fpath = os.path.join(MODELS_DIR, fname)
    dump({"model": model, "calibrator": calibrator, "metadata": metadata}, fpath)
    return fpath


def load_model(name: str, version: str) -> Tuple[Any, Any, Dict]:
    fname = f"{name}__{version}.joblib"
    fpath = os.path.join(MODELS_DIR, fname)
    obj = load(fpath)
    return obj["model"], obj["calibrator"], obj["metadata"]


