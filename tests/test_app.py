"""tests/test_app.py
Unit tests for the Multilingual Profanity & Sentiment Filter API.

The tests cover the main public endpoints:
* ``GET /health`` – basic health‑check.
* ``POST /filter`` – profanity detection and text sanitisation.
* ``POST /sentiment`` – multilingual sentiment analysis.

The test suite uses ``TestClient`` from FastAPI and ``pytest`` for assertions.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

# --------------------------------------------------------------------------- #
# Import the FastAPI application.
# The project may expose the app as ``app`` from ``app.main`` or ``main``.
# --------------------------------------------------------------------------- #
try:
    # Common layout: src/app/main.py -> app variable
    from app.main import app  # type: ignore
except ImportError:  # pragma: no cover
    try:
        # Alternative layout: main.py at the repo root
        from main import app  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Unable to import the FastAPI app. Ensure the project exposes "
            "`app` from either `app.main` or `main`."
        ) from exc

client = TestClient(app)


def _post_json(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to POST JSON data and return the parsed response.

    Args:
        endpoint: API path (e.g., ``/filter``).
        payload: JSON‑serialisable request body.

    Returns:
        The decoded JSON response as a Python ``dict``.
    """
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200, f"Unexpected status {response.status_code}: {response.text}"
    return response.json()


# --------------------------------------------------------------------------- #
# Health‑check
# --------------------------------------------------------------------------- #
def test_health_check() -> None:
    """The health endpoint must return a simple JSON payload with status OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    # The exact key may vary; we accept any truthy ``status`` field.
    assert data.get("status") in {"ok", "OK", True, "healthy"}


# --------------------------------------------------------------------------- #
# Profanity filter – English
# --------------------------------------------------------------------------- #
def test_profanity_filter_english() -> None:
    """Detect profanity in an English sentence and verify sanitisation."""
    payload = {"text": "This is a damn test.", "language": "en"}
    result = _post_json("/filter", payload)

    # Expected keys
    assert "clean_text" in result
    assert "profanities" in result

    # The profanity list should contain the offending word (case‑insensitive)
    profanities = result["profanities"]
    assert isinstance(profanities, list)
    assert any(p.lower() == "damn" for p in profanities)

    # The cleaned text should mask or remove the profanity (implementation‑defined)
    clean = result["clean_text"]
    assert isinstance(clean, str)
    assert "damn" not in clean.lower()


# --------------------------------------------------------------------------- #
# Profanity filter – Spanish
# --------------------------------------------------------------------------- #
def test_profanity_filter_spanish() -> None:
    """Detect Spanish profanity and ensure it is reported correctly."""
    payload = {"text": "¡Qué carajo!", "language": "es"}
    result = _post_json("/filter", payload)

    assert "profanities" in result and isinstance(result["profanities"], list)
    # The word "carajo" is a common Spanish profanity.
    assert any(p.lower() == "carajo" for p in result["profanities"])
    assert "clean_text" in result
    assert "carajo" not in result["clean_text"].lower()


# --------------------------------------------------------------------------- #
# Profanity filter – Chinese (Simplified)
# --------------------------------------------------------------------------- #
def test_profanity_filter_chinese() -> None:
    """Detect profanity in Chinese text."""
    # "他妈的" is a strong profanity in Mandarin.
    payload = {"text": "这真是他妈的糟糕！", "language": "zh"}
    result = _post_json("/filter", payload)

    assert "profanities" in result and isinstance(result["profanities"], list)
    assert any(p == "他妈的" for p in result["profanities"])
    assert "clean_text" in result
    assert "他妈的" not in result["clean_text"]


# --------------------------------------------------------------------------- #
# Profanity filter – Unsupported language
# --------------------------------------------------------------------------- #
def test_profanity_filter_unsupported_language() -> None:
    """The API should reject requests for languages it does not support."""
    payload = {"text": "Some text", "language": "xx"}  # 'xx' is a dummy code
    response = client.post("/filter", json=payload)
    assert response.status_code == 400
    data = response.json()
    # The error response should contain a helpful message.
    assert "detail" in data
    assert "unsupported language" in data["detail"].lower()


# --------------------------------------------------------------------------- #
# Sentiment analysis – English (positive)
# --------------------------------------------------------------------------- #
def test_sentiment_analysis_english_positive() -> None:
    """Positive English sentiment should be classified as positive."""
    payload = {"text": "I absolutely love this product!", "language": "en"}
    result = _post_json("/sentiment", payload)

    assert "sentiment" in result
    assert result["sentiment"].lower() in {"positive", "pos"}
    assert "score" in result
    # Score should be a float between -1 and 1 (or 0‑1 depending on implementation)
    score = result["score"]
    assert isinstance(score, (float, int))
    assert -1.0 <= float(score) <= 1.0


# --------------------------------------------------------------------------- #
# Sentiment analysis – English (negative)
# --------------------------------------------------------------------------- #
def test_sentiment_analysis_english_negative() -> None:
    """Negative English sentiment should be classified as negative."""
    payload = {"text": "I hate this terrible service.", "language": "en"}
    result = _post_json("/sentiment", payload)

    assert "sentiment" in result
    assert result["sentiment"].lower() in {"negative", "neg"}
    assert "score" in result
    score = result["score"]
    assert isinstance(score, (float, int))
    assert -1.0 <= float(score) <= 1.0


# --------------------------------------------------------------------------- #
# Sentiment analysis – Chinese (neutral)
# --------------------------------------------------------------------------- #
def test_sentiment_analysis_chinese_neutral() -> None:
    """A neutral Chinese sentence should not be strongly positive/negative."""
    payload = {"text": "今天的天气还可以。", "language": "zh"}
    result = _post_json("/sentiment", payload)

    assert "sentiment" in result
    # Accept any of the three standard labels
    assert result["sentiment"].lower() in {"neutral", "pos", "neg", "positive", "negative"}
    assert "score" in result
    score = result["score"]
    assert isinstance(score, (float, int))
    # For a neutral sentence the absolute score is expected to be low.
    assert abs(float(score)) < 0.3


# --------------------------------------------------------------------------- #
# Sentiment analysis – Unsupported language
# --------------------------------------------------------------------------- #
def test_sentiment_analysis_unsupported_language() -> None:
    """The sentiment endpoint must reject unsupported language codes."""
    payload = {"text": "Some text", "language": "zz"}
    response = client.post("/sentiment", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "unsupported language" in data["detail"].lower()


# --------------------------------------------------------------------------- #
# Edge case – Empty text
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "endpoint",
    ["/filter", "/sentiment"],
)
def test_empty_text_returns_error(endpoint: str) -> None:
    """Both endpoints should return a 422 validation error for empty strings."""
    payload = {"text": "", "language": "en"}
    response = client.post(endpoint, json=payload)
    # FastAPI's Pydantic validation typically yields 422 for empty fields.
    assert response.status_code in {400, 422}
    data = response.json()
    assert "detail" in data


# --------------------------------------------------------------------------- #
# Edge case – Missing fields
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "endpoint,missing_key",
    [
        ("/filter", "language"),
        ("/filter", "text"),
        ("/sentiment", "language"),
        ("/sentiment", "text"),
    ],
)
def test_missing_fields(endpoint: str, missing_key: str) -> None:
    """Requests missing required fields must be rejected with a clear error."""
    payload = {"text": "sample", "language": "en"}
    payload.pop(missing_key)
    response = client.post(endpoint, json=payload)
    assert response.status_code in {400, 422}
    data = response.json()
    assert "detail" in data
    # The error message should mention the missing field.
    assert missing_key in json.dumps(data).lower()
