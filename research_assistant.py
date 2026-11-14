"""Simple research assistant CLI backed by the Groq SDK."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

import groq
from dotenv import load_dotenv

MAX_RETRIES_CAP = 3

SYSTEM_PROMPT = (
    "You are a careful research assistant. Give a concise, well-structured answer: "
    "a short summary, key points as bullets, and open questions or caveats. "
    "Say clearly when you are unsure; do not invent sources."
)

RETRYABLE_ERRORS = (
    groq.APIConnectionError,  # includes APITimeoutError
    groq.RateLimitError,
    groq.InternalServerError,
)

logger = logging.getLogger("research_assistant")


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.handlers[:] = [handler]
    logger.setLevel(level.upper())
    logger.propagate = False


def log(level: int, msg: str, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str = "openai/gpt-oss-120b"
    max_retries: int = MAX_RETRIES_CAP
    timeout: float = 30.0
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set (add it to .env)")
        max_retries = int(os.getenv("MAX_RETRIES", MAX_RETRIES_CAP))
        return cls(
            api_key=api_key,
            model=os.getenv("GROQ_MODEL", cls.model),
            max_retries=max(0, min(max_retries, MAX_RETRIES_CAP)),
            timeout=float(os.getenv("REQUEST_TIMEOUT", cls.timeout)),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
        )


class ResearchAssistant:
    def __init__(
        self,
        config: Config,
        client: groq.Groq | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        # SDK retries disabled so our own retry loop is the only one.
        self.client = client or groq.Groq(
            api_key=config.api_key, timeout=config.timeout, max_retries=0
        )
        self._sleep = sleep

    def ask(self, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            start = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                )
            except RETRYABLE_ERRORS as exc:
                if attempt == attempts:
                    log(logging.ERROR, "request_failed", attempt=attempt, error=type(exc).__name__)
                    raise
                delay = 2 ** (attempt - 1)
                log(
                    logging.WARNING,
                    "request_retry",
                    attempt=attempt,
                    error=type(exc).__name__,
                    retry_in_s=delay,
                )
                self._sleep(delay)
                continue

            usage = getattr(response, "usage", None)
            log(
                logging.INFO,
                "request_ok",
                attempt=attempt,
                model=self.config.model,
                latency_ms=round((time.perf_counter() - start) * 1000),
                total_tokens=getattr(usage, "total_tokens", None),
            )
            return response.choices[0].message.content or ""

        raise RuntimeError("unreachable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the research assistant a question.")
    parser.add_argument("question", nargs="+", help="the research question")
    args = parser.parse_args(argv)

    try:
        config = Config.from_env()
    except ValueError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    # Windows consoles default to cp1252, which can't print many model outputs.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    setup_logging(config.log_level)
    assistant = ResearchAssistant(config)
    try:
        print(assistant.ask(" ".join(args.question)))
    except groq.APIError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
