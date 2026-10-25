# Multilingual Profanity & Sentiment Filter API

## Project Overview
The **Multilingual Profanity & Sentiment Filter API** provides a fast, scalable HTTP service for detecting profanity and evaluating sentiment across multiple languages. It combines rule‑based tokenization (jieba for Chinese), spaCy pipelines (English, German, French, Spanish, etc.), and transformer‑based sentiment models to deliver high‑accuracy results in a single endpoint.

Key capabilities:
- Detect profanity in 20+ languages with customizable blacklists.
- Return sentiment polarity (positive, neutral, negative) with confidence scores.
- Batch processing for high‑throughput use‑cases.
- Docker‑ready deployment and easy integration into existing services.

## Tech Stack
| Layer | Technology |
|-------|------------|
| **API framework** | FastAPI |
| **Language processing** | spaCy, jieba |
| **Sentiment analysis** | Hugging Face `transformers` (distilbert‑base‑uncased‑finetuned‑sst‑2) |
| **Profanity detection** | Custom regex + scikit‑learn TF‑IDF classifier (fallback) |
| **Testing** | pytest |
| **Containerisation** | Docker |
| **Python** | ≥3.9 |

## Quick Start

### Prerequisites
- Python 3.9 or newer
- `git`
- (Optional) Docker Engine 20.10+

### Clone the repository
```bash
git clone https://github.com/your-org/multilingual-profanity-sentiment-filter.git
cd multilingual-profanity-sentiment-filter
```

### Install dependencies
```bash
# Using a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install the package and all required libraries
pip install -r requirements.txt
```

### Download language models
```bash
# spaCy models
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download es_core_news_sm

# jieba does not require a download step
```

### Run the API locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The service will be reachable at `http://localhost:8000`.

### Docker (optional)
```bash
docker build -t profanity-sentiment-api .
docker run -p 8000:8000 profanity-sentiment-api
```

## API Specification

### Base URL
`http://<host>:<port>/api/v1`

### Endpoints

| Method | Path | Description | Request Body | Response |
|--------|------|-------------|--------------|----------|
| `POST` | `/filter` | Analyze a single text string. | `{ "text": "string", "lang": "en" }` | `{ "profanity": true, "profane_words": ["badword"], "sentiment": { "label": "negative", "score": 0.92 } }` |
| `POST` | `/batch` | Analyze a list of texts (max 100). | `{ "texts": ["..."], "lang": "en" }` | `{ "results": [ {...}, {...} ] }` |
| `GET` | `/health` | Health check. | – | `{ "status": "ok" }` |

#### Example request (single text)

```bash
curl -X POST http://localhost:8000/api/v1/filter \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this product!", "lang":"en"}'
```

#### Example response

```json
{
  "profanity": false,
  "profane_words": [],
  "sentiment": {
    "label": "positive",
    "score": 0.987
  }
}
```

## Configuration

All configurable options live in `app/config.py`. You can override them with environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PROFANITY_BLACKLIST_PATH` | Path to a JSON file containing language‑specific profanity lists. | `config/blacklist.json` |
| `SENTIMENT_MODEL_NAME` | Hugging Face model identifier. | `distilbert-base-uncased-finetuned-sst-2-english` |
| `MAX_BATCH_SIZE` | Maximum number of texts per batch request. | `100` |
| `LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | `INFO` |

### Adding a new language
1. Add a spaCy model name (or jieba for Chinese) to `SUPPORTED_LANGUAGES` in `app/config.py`.
2. Provide a profanity list for the language in `config/blacklist.json`.
3. (Optional) Add a custom tokenizer if needed.

## Testing

Run the full test suite with:

```bash
pytest -vv
```

The tests cover:
- Profanity detection edge cases.
- Sentiment prediction consistency.
- API contract validation (FastAPI TestClient).

## Development Workflow

1. **Branching** – Use `feature/<name>` branches for new functionality.
2. **Linting** – Run `ruff .` and `black .` before committing.
3. **Static typing** – `mypy app/` should pass without errors.
4. **CI** – GitHub Actions automatically run lint, type‑check, and tests on PRs.

## License
This project is licensed under the **MIT License**. See `LICENSE` for details.

## Contact & Support
- **Maintainer:** Jane Doe <jane.doe@example.com>
- **Issues:** Please open a GitHub Issue with a clear description and reproducible steps.