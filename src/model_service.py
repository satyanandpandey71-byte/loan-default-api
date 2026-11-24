"""Loads the trained artifact and turns validated records into risk scores."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    METADATA_FILENAME,
    MODEL_FILENAME,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RAW_FEATURES,
)


def categorize_risk(probability: float, high_cutoff: float, low_cutoff: float) -> str:
    """Map a default probability to a risk band."""
    if probability >= high_cutoff:
        return "high"
    if probability >= low_cutoff:
        return "medium"
    return "low"


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the model input frame with the exact training column order.

    Omitted optional fields become NaN, which the pipeline imputes.
    """
    df = pd.DataFrame(records, columns=RAW_FEATURES)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # None -> NaN
    for col in CATEGORICAL_FEATURES:
        # Imputers only recognise NaN (not None) as "missing".
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df


class RiskModel:
    def __init__(self, model, metadata: dict[str, Any]):
        self.model = model
        self.metadata = metadata
        self.version: str = metadata.get("model_version", "unknown")
        bands = metadata["risk_bands"]
        self.high_cutoff: float = float(bands["high_cutoff"])
        self.low_cutoff: float = float(bands["low_cutoff"])

    @classmethod
    def load(
        cls,
        model_path: Path | str | None = None,
        metadata_path: Path | str | None = None,
    ) -> "RiskModel":
        model_path = Path(model_path or MODELS_DIR / MODEL_FILENAME)
        metadata_path = Path(metadata_path or model_path.with_name(METADATA_FILENAME))
        model = joblib.load(model_path)
        with open(metadata_path) as fh:
            metadata = json.load(fh)
        return cls(model, metadata)

    def predict_default_probability(self, records: list[dict[str, Any]]) -> np.ndarray:
        frame = records_to_frame(records)
        return self.model.predict_proba(frame)[:, 1]

    def score(self, record: dict[str, Any]) -> dict[str, Any]:
        probability = float(self.predict_default_probability([record])[0])
        return {
            "default_probability": round(probability, 4),
            "risk_category": categorize_risk(probability, self.high_cutoff, self.low_cutoff),
            "model_version": self.version,
        }
