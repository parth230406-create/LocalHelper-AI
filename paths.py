from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "storage"
BASE_DIR.mkdir(parents=True, exist_ok=True)

def data_dir() -> Path:
    p = BASE_DIR / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p

def index_dir() -> Path:
    p = data_dir() / "indexes"
    p.mkdir(parents=True, exist_ok=True)
    return p

def uploads_dir() -> Path:
    p = data_dir() / "uploads"
    p.mkdir(parents=True, exist_ok=True)
    return p