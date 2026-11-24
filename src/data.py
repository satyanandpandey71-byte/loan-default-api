"""Synthetic loan-application data generator.

No real banking data is bundled, so this module simulates a realistic portfolio:
default risk rises with high debt-to-income, low credit score, recent
delinquencies, high utilisation, long terms, etc. Missing values are injected at
random so the imputation step has real work to do.

To use real data (e.g. a Lending Club export), provide a CSV with the same
columns (see src/config.py) plus a binary `default` column and pass it to
`python -m src.train --data path/to/file.csv`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    HOME_OWNERSHIP_VALUES,
    LOAN_PURPOSE_VALUES,
    RANDOM_STATE,
    TARGET,
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _solve_intercept(z: np.ndarray, target_rate: float) -> float:
    """Bisection search for the intercept that gives the desired mean default rate."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate_synthetic_loans(
    n: int = 15_000,
    default_rate: float = 0.13,
    missing_rate: float = 1.0,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate `n` synthetic applicants with a binary `default` target.

    `missing_rate` scales the amount of injected missingness (0 disables it).
    """
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(40, 11, n), 21, 75).round()
    annual_income = np.clip(rng.lognormal(np.log(55_000), 0.5, n), 12_000, 400_000).round(-2)
    employment_years = np.minimum(age - 18, rng.exponential(6.5, n)).clip(0, 45).round(1)

    home_ownership = rng.choice(HOME_OWNERSHIP_VALUES, n, p=[0.42, 0.40, 0.15, 0.03])
    loan_purpose = rng.choice(
        LOAN_PURPOSE_VALUES, n, p=[0.40, 0.20, 0.10, 0.10, 0.06, 0.06, 0.08]
    )
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n, p=[0.05, 0.10, 0.50, 0.10, 0.25])

    # Loan size loosely tracks income.
    loan_amount = np.clip(
        rng.lognormal(np.log(11_000), 0.55, n) * (annual_income / 55_000) ** 0.35,
        1_000,
        50_000,
    ).round(-2)

    credit_score = np.clip(rng.normal(685, 70, n), 450, 850).round()
    credit_history_years = np.minimum(age - 18, rng.gamma(3.0, 3.5, n)).clip(0, 50).round(1)
    num_credit_lines = np.clip(rng.poisson(8, n) + 1, 1, 40)
    delinquencies_2y = rng.poisson(np.clip(0.9 - (credit_score - 450) / 500, 0.03, None))
    credit_utilization = np.clip(rng.beta(2, 3, n) * 100 + (700 - credit_score) / 15, 0, 120).round(1)
    debt_to_income = np.clip(rng.normal(18, 8, n) + loan_amount / annual_income * 10, 0, 70).round(1)
    interest_rate = np.clip(
        6.0 + (720 - credit_score) / 28 + (loan_term_months - 36) / 24 + rng.normal(0, 1.3, n),
        5,
        30,
    ).round(2)

    loan_to_income = loan_amount / annual_income

    # ---- True data-generating process for default risk --------------------
    z = (
        -0.010 * (credit_score - 680)
        + 0.035 * (debt_to_income - 18)
        + 0.012 * (credit_utilization - 35)
        + 0.45 * delinquencies_2y
        + 0.07 * (interest_rate - 12)
        - 0.50 * (np.log(annual_income) - np.log(55_000))
        + 1.50 * (loan_to_income - 0.22)
        + 0.35 * (loan_term_months >= 60)
        + 0.50 * (loan_purpose == "small_business")
        + 0.25 * (loan_purpose == "medical")
        - 0.30 * (home_ownership == "OWN")
        + 0.15 * (home_ownership == "RENT")
        - 0.04 * np.minimum(employment_years, 15)
        - 0.02 * np.minimum(credit_history_years, 20)
        + rng.normal(0, 0.5, n)  # irreducible noise
    )
    z = z + _solve_intercept(z, default_rate)
    default = (rng.random(n) < _sigmoid(z)).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "employment_years": employment_years,
            "home_ownership": home_ownership,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "loan_purpose": loan_purpose,
            "interest_rate": interest_rate,
            "credit_score": credit_score,
            "credit_history_years": credit_history_years,
            "num_credit_lines": num_credit_lines,
            "delinquencies_2y": delinquencies_2y,
            "credit_utilization": credit_utilization,
            "debt_to_income": debt_to_income,
            TARGET: default,
        }
    )

    # ---- Inject missing-at-random values -----------------------------------
    missing_probs = {
        "employment_years": 0.04,
        "credit_score": 0.03,
        "credit_history_years": 0.03,
        "credit_utilization": 0.05,
        "debt_to_income": 0.02,
        "home_ownership": 0.01,
    }
    for col, p in missing_probs.items():
        mask = rng.random(n) < p * missing_rate
        df[col] = df[col].where(~mask, np.nan)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic loan dataset CSV.")
    parser.add_argument("--n", type=int, default=15_000)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "loans.csv")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    df = generate_synthetic_loans(n=args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out} (default rate {df[TARGET].mean():.1%})")


if __name__ == "__main__":
    main()
