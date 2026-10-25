import os
from typing import List, Dict, Any

import spacy
import jieba
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from transformers import pipeline, Pipeline

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Default spaCy language models (must be installed via pip, e.g. `python -m spacy download en_core_web_sm`)
_SPACY_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "zh": "zh_core_web_sm",
}
# Minimal profanity word lists per language (real projects would load comprehensive lists)
_PROFANITY_LEXICON: Dict[str, set] = {
    "en": {"shit", "damn", "fuck", "bitch"},
    "es": {"mierda", "joder", "puta"},
    "de": {"scheiße", "verdammt", "fick"},
    "fr": {"merde", "putain"},
    "zh": {"他妈的", "屌", "操"},
}
# Sentiment model identifier (multilingual)
_SENTIMENT_MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment"
)

# ----------------------------------------------------------------------
# Pydantic request/response models
# ----------------------------------------------------------------------


class ProfanityRequest(BaseModel):
    """Request payload for profanity detection."""

    text: str = Field(..., min_length=1, description="Text to be inspected.")
    language: str = Field(
        ...,
        description="ISO‑639‑1 language code (e.g., 'en', 'es', 'zh').",
        regex=r"^[a-z]{2}$",
    )

    @validator("language")
    def _ensure_supported(cls, v: str) -> str:
        if v not in _PROFANITY_LEXICON:
            raise ValueError(f"Unsupported language code: {v}")
        return v


class ProfanityResponse(BaseModel):
    """Response payload for profanity detection."""

    profane_words: List[str] = Field(
        ..., description="List of detected profanity tokens (empty if none)."
    )
    language: str = Field(..., description="Language code of the processed text.")


class SentimentRequest(BaseModel):
    """Request payload for sentiment analysis."""

    text: str = Field(..., min_length=1, description="Text to be analysed.")


class SentimentResponse(BaseModel):
    """Response payload for sentiment analysis."""

    label: str = Field(..., description="Sentiment label (e.g., '1 star', '5 stars').")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the label.")


class FilterResponse(BaseModel):
    """Combined response for profanity and sentiment."""

    profanity: ProfanityResponse
    sentiment: SentimentResponse


# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------


class LanguageResources:
    """
    Container for language‑specific resources such as spaCy pipelines.
    It lazily loads spaCy models on first use to keep startup time low.
    """

    def __init__(self) -> None:
        self._nlp_cache: Dict[str, Any] = {}

    def get_spacy_nlp(self, lang: str):
        """Return a spaCy language pipeline for the given language."""
        if lang in self._nlp_cache:
            return self._nlp_cache[lang]

        model_name = _SPACY_MODELS.get(lang)
        if not model_name:
            raise ValueError(f"No spaCy model configured for language '{lang}'")

        try:
            nlp = spacy.load(model_name, disable=["parser", "ner", "textcat"])
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' not found. Install it with "
                f"`python -m spacy download {model_name}`."
            ) from exc

        self._nlp_cache[lang] = nlp
        return nlp


def _tokenize(text: str, language: str, resources: LanguageResources) -> List[str]:
    """
    Tokenize `text` according to the language.
    - For Chinese (`zh`) we use jieba.
    - For other languages we use spaCy's tokenizer.
    """
    if language == "zh":
        # jieba returns a generator; we convert to list and strip whitespace
        return [tok.strip() for tok in jieba.cut(text) if tok.strip()]
    else:
        nlp = resources.get_spacy_nlp(language)
        return [token.text for token in nlp(text)]


def detect_profanity(
    text: str, language: str, resources: LanguageResources
) -> List[str]:
    """
    Return a list of profanity tokens found in `text` for the given `language`.
    Matching is case‑insensitive for alphabetic languages.
    """
    tokens = _tokenize(text, language, resources)

    # Normalise tokens for case‑insensitive matching where appropriate
    if language != "zh":
        normalized = [t.lower() for t in tokens]
    else:
        normalized = tokens

    profanity_set = _PROFANITY_LEXICON[language]
    found = [original for original, norm in zip(tokens, normalized) if norm in profanity_set]
    # Preserve order of appearance and remove duplicates while keeping first occurrence
    seen = set()
    unique_found = []
    for word in found:
        if word not in seen:
            seen.add(word)
            unique_found.append(word)
    return unique_found


def load_sentiment_pipeline() -> Pipeline:
    """
    Initialise the HuggingFace sentiment analysis pipeline.
    The chosen model is multilingual and returns a star rating (1‑5).
    """
    try:
        return pipeline("sentiment-analysis", model=_SENTIMENT_MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load sentiment model '{_SENTIMENT_MODEL_NAME}'. "
            "Ensure the model name is correct and the internet is reachable."
        ) from exc


def analyze_sentiment(text: str, sentiment_pipe: Pipeline) -> Dict[str, Any]:
    """
    Run sentiment analysis on `text` using the provided pipeline.
    Returns a dict with `label` and `score`.
    """
    result = sentiment_pipe(text)
    if not result:
        raise RuntimeError("Sentiment pipeline returned an empty result.")
    # The pipeline returns a list of dicts; we take the first entry
    entry = result[0]
    # Normalise the label to a string and score to a float
    label = str(entry.get("label", "")).strip()
    score = float(entry.get("score", 0.0))
    return {"label": label, "score": score}


# ----------------------------------------------------------------------
# FastAPI application
# ----------------------------------------------------------------------


app = FastAPI(
    title="Multilingual Profanity & Sentiment Filter API",
    description=(
        "Detect profanity and analyse sentiment for multiple languages. "
        "Powered by spaCy, jieba, and HuggingFace Transformers."
    ),
    version="1.0.0",
)

# Resources are created once at startup and shared across requests
_resources = LanguageResources()
_sentiment_pipe: Pipeline | None = None


@app.on_event("startup")
def _load_models() -> None:
    """Load heavy models once when the service starts."""
    global _sentiment_pipe
    _sentiment_pipe = load_sentiment_pipeline()


@app.post(
    "/profanity",
    response_model=ProfanityResponse,
    summary="Detect profanity in a given text.",
    tags=["Profanity"],
)
def profanity_endpoint(payload: ProfanityRequest) -> ProfanityResponse:
    """
    Detect profanity tokens in `payload.text` for the specified language.
    Returns the list of detected profanity words (empty if none).
    """
    try:
        profane = detect_profanity(payload.text, payload.language, _resources)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ProfanityResponse(profane_words=profane, language=payload.language)


@app.post(
    "/sentiment",
    response_model=SentimentResponse,
    summary="Analyse sentiment of a given text.",
    tags=["Sentiment"],
)
def sentiment_endpoint(payload: SentimentRequest) -> SentimentResponse:
    """
    Analyse the sentiment of `payload.text` using a multilingual transformer model.
    Returns a label (e.g., '1 star' … '5 stars') and a confidence score.
    """
    if _sentiment_pipe is None:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded.")
    try:
        result = analyze_sentiment(payload.text, _sentiment_pipe)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SentimentResponse(label=result["label"], score=result["score"])


@app.post(
    "/filter",
    response_model=FilterResponse,
    summary="Combined profanity detection and sentiment analysis.",
    tags=["Combined"],
)
def filter_endpoint(payload: ProfanityRequest) -> FilterResponse:
    """
    Run both profanity detection and sentiment analysis on the same text.
    The request payload re‑uses the profanity schema (text + language).
    """
    profanity_res = profanity_endpoint(ProfanityRequest(**payload.dict()))
    sentiment_res = sentiment_endpoint(SentimentRequest(text=payload.text))
    return FilterResponse(profanity=profanity_res, sentiment=sentiment_res)


# ----------------------------------------------------------------------
# Optional: entry point for local development
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)