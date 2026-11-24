"""Request validation for the /predict endpoint.

Hand-rolled (no extra dependency) and deliberately strict:
  * unknown fields are rejected, so typos never silently become "missing"
  * booleans and numeric strings are not accepted as numbers
  * NaN / Infinity are rejected
  * optional fields may be omitted or null; the pipeline imputes them
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.config import HOME_OWNERSHIP_VALUES, LOAN_PURPOSE_VALUES


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # "float" | "int" | "category"
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None
    case: str | None = None  # "upper" | "lower" normalisation for categories


FIELDS: tuple[Field, ...] = (
    # --- required --------------------------------------------------------
    Field("age", "int", required=True, minimum=18, maximum=100),
    Field("annual_income", "float", required=True, minimum=1, maximum=10_000_000),
    Field("loan_amount", "float", required=True, minimum=500, maximum=100_000),
    Field("loan_term_months", "int", required=True, minimum=12, maximum=60),
    Field("interest_rate", "float", required=True, minimum=0, maximum=40),
    Field(
        "home_ownership", "category", required=True,
        choices=tuple(HOME_OWNERSHIP_VALUES), case="upper",
    ),
    Field(
        "loan_purpose", "category", required=True,
        choices=tuple(LOAN_PURPOSE_VALUES), case="lower",
    ),
    # --- optional (imputed by the pipeline when absent) -------------------
    Field("employment_years", "float", minimum=0, maximum=60),
    Field("credit_score", "int", minimum=300, maximum=850),
    Field("credit_history_years", "float", minimum=0, maximum=80),
    Field("num_credit_lines", "int", minimum=0, maximum=60),
    Field("delinquencies_2y", "int", minimum=0, maximum=50),
    Field("credit_utilization", "float", minimum=0, maximum=150),
    Field("debt_to_income", "float", minimum=0, maximum=100),
)

FIELD_BY_NAME = {f.name: f for f in FIELDS}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_field(field: Field, value: Any) -> tuple[Any, str | None]:
    """Return (clean_value, error_message). Exactly one of them is None."""
    if field.kind == "category":
        if not isinstance(value, str):
            return None, "must be a string"
        cleaned = value.strip()
        cleaned = cleaned.upper() if field.case == "upper" else cleaned.lower()
        if cleaned not in field.choices:
            return None, f"must be one of {list(field.choices)}"
        return cleaned, None

    if not _is_number(value):
        return None, "must be a number"
    if isinstance(value, float) and not math.isfinite(value):
        return None, "must be a finite number"
    if field.kind == "int":
        if isinstance(value, float):
            if not value.is_integer():
                return None, "must be a whole number"
            value = int(value)
    else:
        value = float(value)

    if field.minimum is not None and value < field.minimum:
        return None, f"must be >= {field.minimum:g}"
    if field.maximum is not None and value > field.maximum:
        return None, f"must be <= {field.maximum:g}"
    return value, None


def validate_payload(payload: Any) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate one applicant record.

    Returns (clean_record, errors). `clean_record` has an entry for every known
    field (None for omitted optional fields) and is only meaningful when
    `errors` is empty.
    """
    errors: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        return {}, [{"field": "<body>", "message": "must be a JSON object"}]

    for key in payload:
        if key not in FIELD_BY_NAME:
            errors.append({"field": str(key), "message": "unknown field"})

    clean: dict[str, Any] = {}
    for field in FIELDS:
        value = payload.get(field.name)
        if value is None:
            if field.required:
                errors.append({"field": field.name, "message": "is required"})
            clean[field.name] = None
            continue
        cleaned, err = _validate_field(field, value)
        if err:
            errors.append({"field": field.name, "message": err})
        clean[field.name] = cleaned

    # Cross-field consistency checks (only when both sides are valid numbers).
    age = clean.get("age")
    if age is not None:
        emp = clean.get("employment_years")
        if emp is not None and emp > age - 14:
            errors.append(
                {"field": "employment_years", "message": "is inconsistent with age"}
            )
        hist = clean.get("credit_history_years")
        if hist is not None and hist > age - 16:
            errors.append(
                {"field": "credit_history_years", "message": "is inconsistent with age"}
            )

    return clean, errors
