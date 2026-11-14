# Research Assistant

Small CLI that answers research questions with Groq.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt
copy .env.example .env          # then put your GROQ_API_KEY in .env
```

## Usage

```bash
python research_assistant.py "What are the tradeoffs of RAG vs fine-tuning?"
```

The answer goes to stdout. Logs go to stderr as JSON lines, for example:

```json
{"ts": "2026-11-14T10:00:00", "level": "INFO", "logger": "research_assistant", "msg": "request_ok", "attempt": 1, "model": "openai/gpt-oss-120b", "latency_ms": 812, "total_tokens": 431}
```

Connection errors, timeouts, 429s and 5xx errors are retried up to 3 times with backoff (1s, 2s, 4s). Other errors fail right away.

## Tests

```bash
pytest
```
