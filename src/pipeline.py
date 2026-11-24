"""Model pipelines and hyper-parameter grids.

Every candidate is a single sklearn Pipeline:

    feature engineering -> (impute, encode, scale) -> classifier

Persisting the whole Pipeline guarantees that serving applies exactly the same
preprocessing that was fitted on the training data.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)
from src.features import make_feature_engineer

MODEL_NAMES = ["logistic_regression", "random_forest", "gradient_boosting"]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def _classifier(name: str):
    if name == "logistic_regression":
        # class_weight="balanced" up-weights the rare default class.
        return LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        )
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=1,  # parallelism comes from GridSearchCV; also best for 1-row serving
            random_state=RANDOM_STATE,
        )
    if name == "gradient_boosting":
        # GradientBoostingClassifier has no class_weight argument, so imbalance
        # is handled with balanced sample weights passed at fit time
        # (see `fit_params_for` below).
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")


def build_pipeline(name: str) -> Pipeline:
    return Pipeline(
        [
            ("features", make_feature_engineer()),
            ("prep", build_preprocessor()),
            ("clf", _classifier(name)),
        ]
    )


def param_grid(name: str, quick: bool = False) -> dict[str, list]:
    """Hyper-parameter grid for GridSearchCV. `quick=True` shrinks it for tests/demos."""
    if name == "logistic_regression":
        return {"clf__C": [0.1, 1.0] if quick else [0.01, 0.1, 1.0, 10.0]}
    if name == "random_forest":
        if quick:
            return {"clf__n_estimators": [60], "clf__max_depth": [8], "clf__min_samples_leaf": [20]}
        return {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [6, 10],
            "clf__min_samples_leaf": [10, 30],
        }
    if name == "gradient_boosting":
        if quick:
            return {"clf__n_estimators": [60], "clf__learning_rate": [0.1], "clf__max_depth": [3]}
        return {
            "clf__n_estimators": [150, 300],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3],
        }
    raise ValueError(f"Unknown model '{name}'.")


def fit_params_for(name: str, y) -> dict:
    """Extra fit() arguments. Only gradient boosting needs sample weights."""
    if name == "gradient_boosting":
        from sklearn.utils.class_weight import compute_sample_weight

        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}
