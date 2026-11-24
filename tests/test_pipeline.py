"""Tests for the ML pipeline, data generator and training script."""
import numpy as np
import pandas as pd
import pytest

from src.config import RAW_FEATURES, TARGET
from src.data import generate_synthetic_loans
from src.model_service import RiskModel, categorize_risk, records_to_frame
from src.pipeline import MODEL_NAMES, build_pipeline, fit_params_for, param_grid
from src.train import best_threshold, run_training, threshold_table
from tests.conftest import VALID_PAYLOAD


# ------------------------------------------------------------ data generator
def test_generator_is_deterministic_and_has_expected_shape():
    a = generate_synthetic_loans(n=500, seed=1)
    b = generate_synthetic_loans(n=500, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == RAW_FEATURES + [TARGET] or set(a.columns) == set(RAW_FEATURES + [TARGET])
    assert 0.08 < a[TARGET].mean() < 0.20


def test_generator_injects_missing_values():
    df = generate_synthetic_loans(n=2000, seed=3)
    assert df["credit_utilization"].isna().any()
    assert generate_synthetic_loans(n=500, seed=3, missing_rate=0).isna().sum().sum() == 0


# ------------------------------------------------------------------ pipeline
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts_with_missing_values(name, synthetic_df):
    df = synthetic_df.head(600)
    pipe = build_pipeline(name)
    pipe.set_params(**{k: v[0] for k, v in param_grid(name, quick=True).items()})
    pipe.fit(df[RAW_FEATURES], df[TARGET], **fit_params_for(name, df[TARGET]))
    proba = pipe.predict_proba(df[RAW_FEATURES])[:, 1]
    assert proba.shape == (600,) and np.all((proba >= 0) & (proba <= 1))


def test_pipeline_handles_unseen_category_and_all_missing_row(synthetic_df):
    pipe = build_pipeline("logistic_regression").fit(
        synthetic_df[RAW_FEATURES], synthetic_df[TARGET]
    )
    row = {c: np.nan for c in RAW_FEATURES}
    row["home_ownership"] = "SPACESHIP"  # never seen in training
    proba = pipe.predict_proba(pd.DataFrame([row]))[:, 1]
    assert np.isfinite(proba).all()


def test_serving_matches_training_pipeline(model_dir, synthetic_df):
    """The API path must give the same numbers as calling the pipeline directly."""
    model = RiskModel.load(model_dir / "model.joblib", model_dir / "metadata.json")
    direct = model.model.predict_proba(synthetic_df[RAW_FEATURES].head(1))[:, 1][0]
    row = synthetic_df[RAW_FEATURES].head(1).iloc[0]
    record = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    served = model.predict_default_probability([record])[0]
    assert served == pytest.approx(direct)


def test_records_to_frame_uses_training_column_order_and_nan():
    frame = records_to_frame([{**VALID_PAYLOAD, "credit_score": None}])
    assert list(frame.columns) == RAW_FEATURES
    assert np.isnan(frame.loc[0, "credit_score"])


# -------------------------------------------------------------- risk bands
@pytest.mark.parametrize(
    "p,expected",
    [(0.0, "low"), (0.19, "low"), (0.2, "medium"), (0.49, "medium"), (0.5, "high"), (1.0, "high")],
)
def test_categorize_risk_boundaries(p, expected):
    assert categorize_risk(p, high_cutoff=0.5, low_cutoff=0.2) == expected


# ------------------------------------------------------- threshold analysis
def test_threshold_table_monotonic_and_consistent():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.15, 5000)
    p = np.clip(0.15 + 0.5 * (y - 0.15) + rng.normal(0, 0.15, 5000), 0, 1)
    t = threshold_table(y, p)
    assert t["recall"].is_monotonic_decreasing
    assert t["approval_rate"].is_monotonic_increasing
    assert 0 < best_threshold(t) < 0.8
    # A useful model must beat "approve everybody".
    assert t["expected_cost"].min() < y.mean() * 5.0


# ------------------------------------------------------ end-to-end training
def test_run_training_quick_end_to_end(tmp_path):
    meta = run_training(
        n_samples=2500, cv_folds=3, quick=True,
        models=["logistic_regression", "random_forest"],
        output_dir=tmp_path / "m", reports_dir=tmp_path / "r",
        make_plots=False, verbose=False,
    )
    assert (tmp_path / "m" / "model.joblib").exists()
    assert (tmp_path / "m" / "metadata.json").exists()
    assert (tmp_path / "r" / "threshold_analysis.csv").exists()
    assert meta["test_metrics"]["roc_auc"] > 0.65
    bands = meta["risk_bands"]
    assert 0 < bands["low_cutoff"] < bands["high_cutoff"] < 1

    # The saved artifact must be loadable and servable.
    model = RiskModel.load(tmp_path / "m" / "model.joblib")
    result = model.score(VALID_PAYLOAD)
    assert result["risk_category"] in {"low", "medium", "high"}
