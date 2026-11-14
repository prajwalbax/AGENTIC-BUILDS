import json
import logging
from types import SimpleNamespace

import groq
import httpx
import pytest

import research_assistant as ra


def make_response(text="answer", tokens=42):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(total_tokens=tokens),
    )


def connection_error():
    return groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))


def status_error(cls, code):
    request = httpx.Request("POST", "https://api.groq.com")
    return cls("boom", response=httpx.Response(code, request=request), body=None)


class FakeClient:
    """Mimics client.chat.completions.create, replaying a script of results."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def config():
    return ra.Config(api_key="test-key", model="test-model", max_retries=3)


def make_assistant(config, client):
    sleeps = []
    return ra.ResearchAssistant(config, client=client, sleep=sleeps.append), sleeps


def test_ask_returns_answer_and_sends_prompt(config):
    client = FakeClient(make_response("hello"))
    assistant, sleeps = make_assistant(config, client)

    assert assistant.ask("  What is RAG?  ") == "hello"
    call = client.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1] == {"role": "user", "content": "What is RAG?"}
    assert sleeps == []


def test_retries_transient_errors_then_succeeds(config):
    client = FakeClient(
        connection_error(),
        status_error(groq.RateLimitError, 429),
        make_response("ok"),
    )
    assistant, sleeps = make_assistant(config, client)

    assert assistant.ask("q") == "ok"
    assert len(client.calls) == 3
    assert sleeps == [1, 2]


def test_gives_up_after_three_retries(config):
    errors = [status_error(groq.InternalServerError, 500) for _ in range(4)]
    client = FakeClient(*errors)
    assistant, sleeps = make_assistant(config, client)

    with pytest.raises(groq.InternalServerError):
        assistant.ask("q")
    assert len(client.calls) == 4  # 1 attempt + 3 retries
    assert sleeps == [1, 2, 4]


def test_non_retryable_error_is_raised_immediately(config):
    client = FakeClient(status_error(groq.AuthenticationError, 401))
    assistant, sleeps = make_assistant(config, client)

    with pytest.raises(groq.AuthenticationError):
        assistant.ask("q")
    assert len(client.calls) == 1
    assert sleeps == []


def test_empty_question_rejected(config):
    assistant, _ = make_assistant(config, FakeClient())
    with pytest.raises(ValueError):
        assistant.ask("   ")


def test_config_from_env_caps_retries(monkeypatch):
    monkeypatch.setattr(ra, "load_dotenv", lambda: None)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("MAX_RETRIES", "10")
    monkeypatch.setenv("GROQ_MODEL", "m")

    config = ra.Config.from_env()
    assert config.max_retries == 3
    assert config.model == "m"


def test_config_requires_api_key(monkeypatch):
    monkeypatch.setattr(ra, "load_dotenv", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        ra.Config.from_env()


def test_logs_are_json_with_fields():
    formatter = ra.JsonFormatter()
    record = logging.LogRecord("research_assistant", logging.INFO, __file__, 1, "request_ok", None, None)
    record.fields = {"attempt": 1, "latency_ms": 12}

    entry = json.loads(formatter.format(record))
    assert entry["msg"] == "request_ok"
    assert entry["level"] == "INFO"
    assert entry["attempt"] == 1
    assert entry["latency_ms"] == 12


def test_main_reports_missing_key(monkeypatch, capsys):
    monkeypatch.setattr(ra, "load_dotenv", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert ra.main(["hello"]) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err
