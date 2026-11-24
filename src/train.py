"""Train, tune, evaluate and persist the loan-default model.

    python -m src.train                 # full run on synthetic data
    python -m src.train --quick         # small grids, fast (demo / CI)
    python -m src.train --data my.csv   # real data with the same columns

Data is split three ways:
    train (fit)   - model selection with stratified CV + GridSearchCV
    calibration   - Platt scaling so outputs behave like probabilities, and
                    the cost-optimal decision threshold is chosen here
    test          - touched once, for the final report
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.config import (
    CATEGORICAL_FEATURES,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    LOW_BAND_FRACTION,
    METADATA_FILENAME,
    MODEL_FILENAME,
    MODEL_VERSION,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RAW_FEATURES,
    REPORTS_DIR,
    TARGET,
)
from src.data import generate_synthetic_loans
from src.model_service import categorize_risk
from src.pipeline import MODEL_NAMES, build_pipeline, fit_params_for, param_grid


# --------------------------------------------------------------------------- #
# Threshold analysis
# --------------------------------------------------------------------------- #
def threshold_table(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray | None = None,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> pd.DataFrame:
    """Business trade-off for every candidate cut-off.

    Policy: decline an applicant when P(default) >= threshold, else approve.
      recall        - share of true defaulters that we decline (default class)
      precision     - share of declined applicants who really would default
      approval_rate - share of applicants approved (volume / revenue side)
      bad_rate_approved - default rate inside the approved book (risk side)
      expected_cost - average cost per applicant, using the relative costs
                      of approving a defaulter vs declining a good customer
    """
    y_true = np.asarray(y_true)
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.80, 0.01), 2)
    n = len(y_true)
    rows = []
    for t in thresholds:
        declined = proba >= t
        approved = ~declined
        tp = int(((declined) & (y_true == 1)).sum())
        fp = int(((declined) & (y_true == 0)).sum())  # good customers turned away
        fn = int(((approved) & (y_true == 1)).sum())  # defaulters approved
        rows.append(
            {
                "threshold": float(t),
                "approval_rate": approved.mean(),
                "recall": tp / max(tp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "bad_rate_approved": fn / max(approved.sum(), 1),
                "false_negatives": fn,
                "false_positives": fp,
                "expected_cost": (fn * cost_fn + fp * cost_fp) / n,
            }
        )
    return pd.DataFrame(rows)


def best_threshold(table: pd.DataFrame) -> float:
    return float(table.loc[table["expected_cost"].idxmin(), "threshold"])


# --------------------------------------------------------------------------- #
# Plots (optional - skipped if matplotlib is missing)
# --------------------------------------------------------------------------- #
def save_plots(table: pd.DataFrame, chosen: float, y_test, p_test, out_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    files = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(table.threshold, table.recall, label="Recall (default class)")
    ax.plot(table.threshold, table.precision, label="Precision")
    ax.plot(table.threshold, table.approval_rate, label="Approval rate")
    ax.axvline(chosen, color="k", ls="--", lw=1, label=f"Chosen t={chosen:.2f}")
    ax.set_xlabel("Decline if P(default) >= threshold")
    ax.set_ylabel("Share")
    ax.set_title("Approval volume vs. risk caught")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(table.threshold, table.expected_cost, color="tab:red")
    ax.axvline(chosen, color="k", ls="--", lw=1)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Expected cost per applicant")
    ax.set_title(f"Cost curve (FN cost {COST_FALSE_NEGATIVE:g} : FP cost {COST_FALSE_POSITIVE:g})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "threshold_analysis.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    files.append(path.name)

    fpr, tpr, _ = roc_curve(y_test, p_test)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, p_test):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (held-out test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "roc_curve.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    files.append(path.name)
    return files


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def load_data(data_path: Path | None, n_samples: int, seed: int) -> pd.DataFrame:
    if data_path is None:
        return generate_synthetic_loans(n=n_samples, seed=seed)
    df = pd.read_csv(data_path)
    missing = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{data_path} is missing required columns: {missing}")
    return df


def run_training(
    data_path: Path | None = None,
    n_samples: int = 15_000,
    cv_folds: int = 5,
    quick: bool = False,
    models: list[str] | None = None,
    output_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    seed: int = RANDOM_STATE,
    make_plots: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    log = print if verbose else (lambda *a, **k: None)
    models = models or MODEL_NAMES

    df = load_data(data_path, n_samples, seed)
    X, y = df[RAW_FEATURES], df[TARGET].astype(int)
    log(f"Data: {len(df):,} rows, default rate {y.mean():.2%}")

    # 1) hold out the test set, 2) carve a calibration set from what remains
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_dev, y_dev, test_size=0.15, stratify=y_dev, random_state=seed
    )
    X_fit, X_cal, X_test = (d.reset_index(drop=True) for d in (X_fit, X_cal, X_test))
    y_fit, y_cal, y_test = (s.reset_index(drop=True) for s in (y_fit, y_cal, y_test))
    log(f"Split: fit={len(X_fit):,}  calibration={len(X_cal):,}  test={len(X_test):,}")

    # ---- model comparison + tuning with stratified CV -----------------------
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scoring = {"roc_auc": "roc_auc", "recall": "recall", "pr_auc": "average_precision"}
    searches: dict[str, GridSearchCV] = {}
    comparison_rows, all_cv_rows = [], []

    for name in models:
        t0 = time.time()
        search = GridSearchCV(
            build_pipeline(name),
            param_grid(name, quick=quick),
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X_fit, y_fit, **fit_params_for(name, y_fit))
        searches[name] = search
        i = search.best_index_
        r = search.cv_results_
        comparison_rows.append(
            {
                "model": name,
                "cv_roc_auc": r["mean_test_roc_auc"][i],
                "cv_roc_auc_std": r["std_test_roc_auc"][i],
                "cv_recall_default": r["mean_test_recall"][i],
                "cv_pr_auc": r["mean_test_pr_auc"][i],
                "best_params": json.dumps(
                    {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
                ),
                "seconds": round(time.time() - t0, 1),
            }
        )
        cvr = pd.DataFrame(r)[
            ["params", "mean_test_roc_auc", "std_test_roc_auc", "mean_test_recall", "mean_test_pr_auc"]
        ].copy()
        cvr.insert(0, "model", name)
        cvr["params"] = cvr["params"].astype(str)
        all_cv_rows.append(cvr)
        log(
            f"  {name:<20} CV AUC {r['mean_test_roc_auc'][i]:.4f} "
            f"(+/- {r['std_test_roc_auc'][i]:.4f})  recall {r['mean_test_recall'][i]:.3f}  "
            f"[{time.time() - t0:.0f}s]"
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("cv_roc_auc", ascending=False)
    best_name = comparison.iloc[0]["model"]
    best_pipeline = searches[best_name].best_estimator_
    log(f"Selected model: {best_name}")

    # ---- probability calibration -------------------------------------------
    # Class weights inflate scores for the rare class; Platt scaling on a
    # separate split maps them back to real default frequencies.
    calibrated = CalibratedClassifierCV(FrozenEstimator(best_pipeline), method="sigmoid")
    calibrated.fit(X_cal, y_cal)

    # ---- decision threshold chosen on the calibration set -------------------
    p_cal = calibrated.predict_proba(X_cal)[:, 1]
    threshold = best_threshold(threshold_table(y_cal, p_cal))
    low_cutoff = round(threshold * LOW_BAND_FRACTION, 4)

    # ---- final, one-shot evaluation on the test set -------------------------
    p_test = calibrated.predict_proba(X_test)[:, 1]
    p_test_raw = best_pipeline.predict_proba(X_test)[:, 1]
    table = threshold_table(y_test, p_test)
    row = threshold_table(y_test, p_test, thresholds=np.array([threshold])).iloc[0]
    approve_all_cost = float(y_test.mean() * COST_FALSE_NEGATIVE)

    test_metrics = {
        "roc_auc": roc_auc_score(y_test, p_test),
        "pr_auc": average_precision_score(y_test, p_test),
        "brier_calibrated": brier_score_loss(y_test, p_test),
        "brier_uncalibrated": brier_score_loss(y_test, p_test_raw),
        "mean_predicted_probability": float(p_test.mean()),
        "observed_default_rate": float(y_test.mean()),
        "decision_threshold": threshold,
        "recall_default_at_threshold": row["recall"],
        "precision_default_at_threshold": row["precision"],
        "approval_rate_at_threshold": row["approval_rate"],
        "bad_rate_among_approved": row["bad_rate_approved"],
        "expected_cost_at_threshold": row["expected_cost"],
        "expected_cost_approve_everyone": approve_all_cost,
    }

    bands = pd.Series(
        [categorize_risk(p, threshold, low_cutoff) for p in p_test], name="band"
    )
    band_summary = {
        band: {
            "applicants": int((bands == band).sum()),
            "observed_default_rate": float(y_test[bands == band].mean())
            if (bands == band).any()
            else None,
        }
        for band in ("low", "medium", "high")
    }

    log("\nHeld-out test results")
    log(f"  ROC-AUC                      {test_metrics['roc_auc']:.4f}")
    log(f"  PR-AUC                       {test_metrics['pr_auc']:.4f}")
    log(f"  Decision threshold           {threshold:.2f}")
    log(f"  Recall (default class)       {row['recall']:.3f}")
    log(f"  Precision (default class)    {row['precision']:.3f}")
    log(f"  Approval rate                {row['approval_rate']:.3f}")
    log(f"  Bad rate among approved      {row['bad_rate_approved']:.3f}  (portfolio: {y_test.mean():.3f})")
    log(f"  Cost/applicant: model {row['expected_cost']:.3f} vs approve-all {approve_all_cost:.3f}")
    log(f"  Risk bands: {json.dumps(band_summary)}")

    # ---- persist artifacts --------------------------------------------------
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(calibrated, output_dir / MODEL_FILENAME, compress=3)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "selected_model": best_name,
        "best_params": json.loads(comparison.iloc[0]["best_params"]),
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "data": {
            "source": str(data_path) if data_path else "synthetic",
            "rows": int(len(df)),
            "default_rate": float(y.mean()),
            "fit_rows": int(len(X_fit)),
            "calibration_rows": int(len(X_cal)),
            "test_rows": int(len(X_test)),
        },
        "cross_validation": {"folds": cv_folds, "results": comparison.drop(columns="best_params").to_dict("records")},
        "test_metrics": test_metrics,
        "cost_assumptions": {
            "false_negative": COST_FALSE_NEGATIVE,
            "false_positive": COST_FALSE_POSITIVE,
        },
        "risk_bands": {
            "low_cutoff": low_cutoff,
            "high_cutoff": threshold,
            "test_summary": band_summary,
        },
    }
    with open(output_dir / METADATA_FILENAME, "w") as fh:
        json.dump(metadata, fh, indent=2, default=float)

    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    pd.concat(all_cv_rows).to_csv(reports_dir / "cv_grid_results.csv", index=False)
    table.to_csv(reports_dir / "threshold_analysis.csv", index=False)
    if make_plots:
        save_plots(table, threshold, y_test, p_test, reports_dir)

    log(f"\nSaved model to {output_dir / MODEL_FILENAME}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", type=Path, help="CSV with RAW_FEATURES + 'default' (default: synthetic)")
    parser.add_argument("--n-samples", type=int, default=15_000, help="synthetic rows to generate")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="tiny grids for a fast demo run")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, help="subset of models to compare")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    run_training(
        data_path=args.data,
        n_samples=args.n_samples,
        cv_folds=args.cv_folds,
        quick=args.quick,
        models=args.models,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
