"""Feature engineering that lives *inside* the sklearn Pipeline.

Because this is a pipeline step, the exact same transformation runs at
training time and at serving time.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio features. Missing/zero income yields NaN (imputed later)."""
    out = df.copy()
    income = out["annual_income"].astype(float)
    income = income.where(income > 0, np.nan)
    out["loan_to_income"] = out["loan_amount"].astype(float) / income
    return out


def make_feature_engineer() -> FunctionTransformer:
    return FunctionTransformer(add_engineered_features, validate=False)
