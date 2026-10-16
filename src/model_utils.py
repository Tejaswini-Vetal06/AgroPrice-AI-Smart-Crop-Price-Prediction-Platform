"""
Model utility functions for AgroPrice AI.
Provides safe model loading, metadata caching, and verification.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
from sklearn.pipeline import Pipeline
import streamlit as st

from src.config import DEFAULT_MODEL_PATH, METADATA_PATH


@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: Optional[Path] = None) -> Pipeline:
    """Loads and caches the trained scikit-learn pipeline."""
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Model file not found at {target_path}. Please run train.py first.")
    
    pipeline = joblib.load(target_path)
    return pipeline


@st.cache_data(show_spinner=False)
def load_model_metadata(metadata_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads and caches model metadata and evaluation metrics."""
    target_path = Path(metadata_path) if metadata_path else METADATA_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {target_path}. Please run train.py first.")
    
    with open(target_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def is_model_ready() -> bool:
    """Checks if the trained pipeline and metadata are ready on disk."""
    return DEFAULT_MODEL_PATH.exists() and METADATA_PATH.exists()
