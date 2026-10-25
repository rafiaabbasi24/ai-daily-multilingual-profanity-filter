import re
import logging
from functools import lru_cache
from typing import Dict, Any, Tuple

import jieba  # Chinese tokenization
import spacy
from spacy.language import Language
from transformers import pipeline as hf_pipeline, Pipeline
from fastapi import HTTPException

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _contains_chinese(text: str) -> bool:
    """Return True if the string contains any CJK Unified Ideographs."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


@lru_cache(maxsize=1)
def _load_spacy_model(lang: str) -> Language:
    """
    Load a spaCy language model.

    For English we use ``en_core_web_sm``.
    For other languages we fall back to the multilingual small model ``xx_ent_wiki_sm``.
    """
    try:
        if lang == "en":
            return spacy.load("en_core_web_sm")
        else:
            return spacy.load("xx_ent_wiki_sm")
    except OSError as exc:
        logger.error("spaCy model loading failed: %s", exc)
        raise HTTPException(status_code=500, detail="spaCy model could not be loaded.") from exc


@lru_cache(maxsize=1)
def _load_profanity_pipeline() -> Pipeline:
    """
    Load a HuggingFace text‑classification pipeline for profanity detection.

    The model ``unitary/toxic-bert`` is a multilingual toxicity classifier.
    """
    try:
        return hf_pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            tokenizer="unitary/toxic-bert",
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        logger.error("Failed to load profanity model: %s", exc)
        raise HTTPException(status_code=500, detail="Profanity model could not be loaded.") from exc


@lru_cache(maxsize=1)
def _load_sentiment_pipeline() -> Pipeline:
    """
    Load a HuggingFace text‑classification pipeline for sentiment analysis.

    The model ``nlptown/bert-base-multilingual-uncased-sentiment`` provides
    a 1‑5 star rating for many languages.
    """
    try:
        return hf_pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        logger.error("Failed to load sentiment model: %s", exc)
        raise HTTPException(status_code=500, detail="Sentiment model could not be loaded.") from exc


class LanguageDetector:
    """
    Very lightweight language detector.

    - If Chinese characters are present, returns ``'zh'``.
    - Otherwise defaults to English (``'en'``) and lets spaCy handle tokenisation.
    """

    @staticmethod
    def detect(text: str) -> str:
        """
        Detect the language of ``text``.

        Returns:
            str: ``'zh'`` for Chinese, ``'en'`` for everything else.
        """
        if _contains_chinese(text):
            return "zh"
        return "en"


class ProfanityFilter:
    """
    Wrapper around a HuggingFace text‑classification model that predicts
    whether a piece of text is profane/toxic.
    """

    def __init__(self):
        self.pipeline = _load_profanity_pipeline()

    def is_profane(self, text: str, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Evaluate ``text`` for profanity.

        Args:
            text: Input string.
            threshold: Score above which the text is considered profane.

        Returns:
            Tuple[bool, float]: ``(is_profane, confidence)`` where ``confidence``
            is the probability of the toxic class.
        """
        if not text.strip():
            return False, 0.0

        try:
            result = self.pipeline(text)[0]  # type: ignore[index]
            # The model returns labels like "toxic" / "non‑toxic"
            label = result["label"].lower()
            score = float(result["score"])
            is_toxic = label in {"toxic", "hate", "offensive", "insult"} or score >= threshold
            return is_toxic, score
        except Exception as exc:
            logger.exception("Profanity detection failed: %s", exc)
            raise HTTPException(status_code=500, detail="Profanity detection error.") from exc


class SentimentAnalyzer:
    """
    Wrapper around a multilingual sentiment analysis model.
    Returns a star rating (1‑5) and a confidence score.
    """

    def __init__(self):
        self.pipeline = _load_sentiment_pipeline()

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyse sentiment of ``text``.

        Returns:
            dict: ``{'label': '1 star', 'score': 0.73, 'rating': 1}``
        """
        if not text.strip():
            return {"label": "neutral", "score": 0.0, "rating": 3}

        try:
            result = self.pipeline(text)[0]  # type: ignore[index]
            label = result["label"]
            score = float(result["score"])
            # Extract numeric rating from label, e.g. "4 stars"
            rating_match = re.search(r"(\d)", label)
            rating = int(rating_match.group(1)) if rating_match else 3
            return {"label": label, "score": score, "rating": rating}
        except Exception as exc:
            logger.exception("Sentiment analysis failed: %s", exc)
            raise HTTPException(status_code=500, detail="Sentiment analysis error.") from exc


class TextProcessor:
    """
    High‑level pipeline that combines language detection, profanity checking
    and sentiment analysis.
    """

    def __init__(self):
        self.lang_detector = LanguageDetector()
        self.profanity_filter = ProfanityFilter()
        self.sentiment_analyzer = SentimentAnalyzer()

    def process(self, text: str) -> Dict[str, Any]:
        """
        Run the full pipeline on ``text``.

        Returns a dictionary with:
            - ``language``: ISO‑639‑1 code (``'en'`` or ``'zh'``)
            - ``tokens``: list of tokens (spaCy for English, jieba for Chinese)
            - ``is_profane``: bool
            - ``profanity_score``: float
            - ``sentiment``: dict from :class:`SentimentAnalyzer`
        """
        language = self.lang_detector.detect(text)

        # Tokenisation
        if language == "zh":
            tokens = list(jieba.cut_for_search(text))
        else:
            nlp = _load_spacy_model(language)
            doc = nlp(text)
            tokens = [token.text for token in doc]

        is_profane, profanity_score = self.profanity_filter.is_profane(text)
        sentiment = self.sentiment_analyzer.analyze(text)

        return {
            "language": language,
            "tokens": tokens,
            "is_profane": is_profane,
            "profanity_score": profanity_score,
            "sentiment": sentiment,
        }


# Singleton instance for FastAPI dependency injection
_processor_instance: TextProcessor | None = None


def get_text_processor() -> TextProcessor:
    """
    Return a cached :class:`TextProcessor` instance.

    FastAPI can use this as a dependency:
        ``processor: TextProcessor = Depends(get_text_processor)``

    Returns:
        TextProcessor: The shared pipeline object.
    """
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = TextProcessor()
        logger.info("TextProcessor instance created.")
    return _processor_instance