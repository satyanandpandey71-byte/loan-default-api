"""Shared fixtures: a small, fast model trained on synthetic data in a temp dir."""
import json
import sys
from pathlib import Path

import joblib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app import create_app  # noqa: E402
from src.config import RAW_FEATURES, TARGET  # noqa: E402
from src.data import generate_synthetic_loans  # noqa: E402
from src.pipeline import build_pipeline  # noqa: E402

VALID_PAYLOAD = {
    "age": 38,
    "annual_income": 62000,
    "employment_years": 7.5,
    "home_ownership": "MORTGAGE",
    "loan_amount": 12000,
    "loan_term_months": 36,
    "loan_purpose": "debt_consolidation",
    "interest_rate": 11.5,
    "credit_score": 690,
    "credit_history_years": 12,
    "num_credit_lines": 8,
    "delinquencies_2y": 0,
    "credit_utilization": 35.0,
    "debt_to_income": 17.5,
}

LOW_RISK_PAYLOAD = {
    **VALID_PAYLOAD,
    "annual_income": 140000,
    "loan_amount": 6000,
    "credit_score": 800,
    "delinquencies_2y": 0,
    "credit_utilization": 8.0,
    "debt_to_income": 6.0,
    "interest_rate": 6.5,
    "home_ownership": "OWN",
}

HIGH_RISK_PAYLOAD = {
    **VALID_PAYLOAD,
    "annual_income": 24000,
    "loan_amount": 22000,
    "loan_term_months": 60,
    "credit_score": 520,
    "delinquencies_2y": 4,
    "credit_utilization": 95.0,
    "debt_to_income": 42.0,
    "interest_rate": 24.0,
    "home_ownership": "RENT",
    "loan_purpose": "small_business",
}


@pytest.fixture(scope="session")
def synthetic_df():
    return generate_synthetic_loans(n=3000, seed=7)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, synthetic_df):
    """Fit a logistic-regression pipeline and write model + metadata artifacts."""
    out = tmp_path_factory.mktemp("model")
    pipe = build_pipeline("logistic_regression")
    pipe.fit(synthetic_df[RAW_FEATURES], synthetic_df[TARGET])
    joblib.dump(pipe, out / "model.joblib")
    metadata = {
        "model_version": "test-1.0",
        "risk_bands": {"low_cutoff": 0.2, "high_cutoff": 0.5},
    }
    (out / "metadata.json").write_text(json.dumps(metadata))
    return out


@pytest.fixture()
def app(model_dir):
    return create_app(model_dir / "model.joblib", model_dir / "metadata.json")


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def client_without_model(tmp_path):
    return create_app(tmp_path / "missing.joblib", tmp_path / "missing.json").test_client()
