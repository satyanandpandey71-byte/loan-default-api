"""Integration tests for the Flask API (/health and /predict)."""
import json

from tests.conftest import HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD, VALID_PAYLOAD


# ---------------------------------------------------------------------- UI
def test_ui_page_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.content_type.startswith("text/html")
    html = resp.get_data(as_text=True)
    assert "Loan Default Risk Scorer" in html and "/predict" in html


# ----------------------------------------------------------------- /health
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-1.0"


def test_health_reports_unavailable_when_model_missing(client_without_model):
    resp = client_without_model.get("/health")
    assert resp.status_code == 503
    assert resp.get_json()["model_loaded"] is False


# ---------------------------------------------------------------- /predict
def test_predict_returns_probability_and_category(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body) == {"default_probability", "risk_category", "model_version"}
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_category"] in {"low", "medium", "high"}
    assert resp.content_type == "application/json"


def test_predict_works_with_only_required_fields(client):
    optional = {
        "employment_years", "credit_score", "credit_history_years",
        "num_credit_lines", "delinquencies_2y", "credit_utilization", "debt_to_income",
    }
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k not in optional}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert 0.0 <= resp.get_json()["default_probability"] <= 1.0


def test_predict_is_deterministic(client):
    a = client.post("/predict", json=VALID_PAYLOAD).get_json()
    b = client.post("/predict", json=VALID_PAYLOAD).get_json()
    assert a == b


def test_riskier_applicant_scores_higher(client):
    low = client.post("/predict", json=LOW_RISK_PAYLOAD).get_json()
    high = client.post("/predict", json=HIGH_RISK_PAYLOAD).get_json()
    assert high["default_probability"] > low["default_probability"]
    assert low["risk_category"] == "low"
    assert high["risk_category"] == "high"


def test_predict_rejects_invalid_values_with_details(client):
    resp = client.post("/predict", json={**VALID_PAYLOAD, "credit_score": 999, "age": "x"})
    assert resp.status_code == 422
    body = resp.get_json()
    assert body["error"] == "validation_failed"
    assert {d["field"] for d in body["details"]} == {"credit_score", "age"}


def test_predict_rejects_missing_required_field(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "loan_amount"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    assert resp.get_json()["details"][0] == {"field": "loan_amount", "message": "is required"}


def test_predict_rejects_unknown_field(client):
    resp = client.post("/predict", json={**VALID_PAYLOAD, "ssn": "123"})
    assert resp.status_code == 422


def test_predict_rejects_malformed_json(client):
    resp = client.post("/predict", data="{not json", content_type="application/json")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "invalid_json"


def test_predict_rejects_missing_content_type(client):
    resp = client.post("/predict", data=json.dumps(VALID_PAYLOAD))
    assert resp.status_code == 400


def test_predict_rejects_non_object_json(client):
    resp = client.post("/predict", json=[VALID_PAYLOAD])
    assert resp.status_code == 422
    assert resp.get_json()["details"][0]["field"] == "<body>"


def test_predict_returns_503_without_model(client_without_model):
    resp = client_without_model.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "model_unavailable"


def test_oversized_body_rejected(client):
    resp = client.post("/predict", json={**VALID_PAYLOAD, "pad": "x" * 100_000})
    assert resp.status_code == 413
    assert resp.get_json()["error"] == "request_entity_too_large"


# --------------------------------------------------------- generic errors
def test_unknown_route_returns_json_404(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "not_found"


def test_wrong_method_returns_json_405(client):
    resp = client.get("/predict")
    assert resp.status_code == 405
    assert resp.get_json()["error"] == "method_not_allowed"
