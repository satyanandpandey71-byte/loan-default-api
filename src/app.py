"""Flask REST API: /health and /predict."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from src.model_service import RiskModel
from src.schemas import validate_payload

logger = logging.getLogger("loan_api")


def create_app(
    model_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
) -> Flask:
    """Application factory (makes the app easy to test with a temporary model)."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # a single applicant is tiny
    app.json.sort_keys = False

    model_path = model_path or os.environ.get("MODEL_PATH")
    metadata_path = metadata_path or os.environ.get("METADATA_PATH")

    try:
        app.extensions["risk_model"] = RiskModel.load(model_path, metadata_path)
        logger.info("Loaded model version %s", app.extensions["risk_model"].version)
    except Exception:  # noqa: BLE001 - keep the process up so /health can report it
        logger.exception("Could not load model artifact")
        app.extensions["risk_model"] = None

    def get_model() -> RiskModel | None:
        return app.extensions["risk_model"]

    # ------------------------------------------------------------------ routes
    @app.get("/")
    def index():
        """Simple web UI that calls /predict."""
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/health")
    def health():
        model = get_model()
        if model is None:
            return jsonify(status="unavailable", model_loaded=False), 503
        return jsonify(status="ok", model_loaded=True, model_version=model.version)

    @app.post("/predict")
    def predict():
        model = get_model()
        if model is None:
            return jsonify(error="model_unavailable", message="Model is not loaded."), 503

        payload = request.get_json(silent=True)
        if payload is None:
            return (
                jsonify(
                    error="invalid_json",
                    message="Body must be valid JSON sent with Content-Type: application/json.",
                ),
                400,
            )

        record, errors = validate_payload(payload)
        if errors:
            return jsonify(error="validation_failed", details=errors), 422

        result = model.score(record)
        logger.info(
            "scored risk=%s p=%.4f", result["risk_category"], result["default_probability"]
        )  # never log applicant fields (PII)
        return jsonify(result)

    # --------------------------------------------------------- error handlers
    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        return (
            jsonify(error=exc.name.lower().replace(" ", "_"), message=exc.description),
            exc.code,
        )

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        logger.exception("Unhandled error")
        return jsonify(error="internal_server_error", message="Unexpected error."), 500

    return app
