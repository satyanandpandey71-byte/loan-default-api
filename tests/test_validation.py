"""Unit tests for request validation (src/schemas.py)."""
import copy

import pytest

from src.schemas import validate_payload
from tests.conftest import VALID_PAYLOAD


def errors_for(payload):
    _, errors = validate_payload(payload)
    return {e["field"]: e["message"] for e in errors}


def test_valid_payload_has_no_errors():
    clean, errors = validate_payload(VALID_PAYLOAD)
    assert errors == []
    assert clean["age"] == 38 and clean["home_ownership"] == "MORTGAGE"


def test_optional_fields_can_be_omitted_or_null():
    required_only = {
        k: v
        for k, v in VALID_PAYLOAD.items()
        if k
        not in {
            "employment_years", "credit_score", "credit_history_years",
            "num_credit_lines", "delinquencies_2y", "credit_utilization", "debt_to_income",
        }
    }
    clean, errors = validate_payload(required_only)
    assert errors == []
    assert clean["credit_score"] is None

    with_null = {**VALID_PAYLOAD, "credit_score": None}
    assert validate_payload(with_null)[1] == []


@pytest.mark.parametrize(
    "field",
    ["age", "annual_income", "loan_amount", "loan_term_months",
     "interest_rate", "home_ownership", "loan_purpose"],
)
def test_missing_required_field(field):
    payload = copy.deepcopy(VALID_PAYLOAD)
    del payload[field]
    assert errors_for(payload)[field] == "is required"


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", "38"),            # numeric string
        ("age", True),            # bool is not a number
        ("annual_income", [1]),   # wrong container
        ("credit_score", {"x": 1}),
        ("interest_rate", float("nan")),
        ("interest_rate", float("inf")),
        ("age", 38.5),            # not a whole number
    ],
)
def test_wrong_types_rejected(field, value):
    assert field in errors_for({**VALID_PAYLOAD, field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", 17),
        ("age", 101),
        ("annual_income", 0),
        ("loan_amount", 100),
        ("loan_amount", 5_000_000),
        ("credit_score", 200),
        ("credit_score", 900),
        ("debt_to_income", -1),
        ("credit_utilization", 500),
        ("loan_term_months", 6),
    ],
)
def test_out_of_range_rejected(field, value):
    assert field in errors_for({**VALID_PAYLOAD, field: value})


def test_invalid_category_rejected():
    errs = errors_for({**VALID_PAYLOAD, "loan_purpose": "yacht", "home_ownership": "CASTLE"})
    assert "loan_purpose" in errs and "home_ownership" in errs


def test_category_is_normalised():
    clean, errors = validate_payload(
        {**VALID_PAYLOAD, "home_ownership": " rent ", "loan_purpose": "CAR"}
    )
    assert errors == []
    assert clean["home_ownership"] == "RENT" and clean["loan_purpose"] == "car"


def test_unknown_field_rejected():
    assert errors_for({**VALID_PAYLOAD, "creditscore": 700})["creditscore"] == "unknown field"


def test_cross_field_consistency():
    errs = errors_for({**VALID_PAYLOAD, "age": 25, "employment_years": 20})
    assert "employment_years" in errs
    errs = errors_for({**VALID_PAYLOAD, "age": 25, "credit_history_years": 20})
    assert "credit_history_years" in errs


@pytest.mark.parametrize("body", [[], "text", 5, None])
def test_non_object_body_rejected(body):
    _, errors = validate_payload(body)
    assert errors and errors[0]["field"] == "<body>"


def test_multiple_errors_reported_together():
    errs = errors_for({"age": 10})
    assert "age" in errs and "annual_income" in errs and "loan_purpose" in errs
