"""Central configuration: paths, feature lists and business assumptions.

Keeping the feature lists here means training, validation and serving all agree
on the same schema.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"

RANDOM_STATE = 42
TARGET = "default"

# ---- Raw input features (what a client sends to /predict) -------------------
NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "credit_score",
    "credit_history_years",
    "num_credit_lines",
    "delinquencies_2y",
    "credit_utilization",
    "debt_to_income",
]
CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose"]

# Derived inside the pipeline (see src/features.py), never sent by the client.
ENGINEERED_NUMERIC_FEATURES = ["loan_to_income"]

RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

HOME_OWNERSHIP_VALUES = ["RENT", "MORTGAGE", "OWN", "OTHER"]
LOAN_PURPOSE_VALUES = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "car",
    "small_business",
    "medical",
    "other",
]

# ---- Business assumptions for the threshold analysis ------------------------
# Relative costs, in units of "profit on one good loan":
#   approving a loan that later defaults loses far more than the profit
#   forgone by declining a good applicant.
COST_FALSE_NEGATIVE = 5.0  # approve an applicant who defaults
COST_FALSE_POSITIVE = 1.0  # decline an applicant who would have repaid

# Risk bands are derived from the cost-optimal decision threshold t*:
#   low    : p <  t* * LOW_BAND_FRACTION
#   medium : t* * LOW_BAND_FRACTION <= p < t*
#   high   : p >= t*
LOW_BAND_FRACTION = 0.5

MODEL_VERSION = "1.0.0"
