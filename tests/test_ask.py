"""The Ask chat, with a fake Groq client (no network, no API key needed)."""
import json
import sys
from types import SimpleNamespace

import pytest

from conftest import _load_app


class FakeStatusError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def reply(content="", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroq:
    """Replays scripted replies; models listed in ``retired`` fail like a decommissioned Groq model."""

    def __init__(self, replies, retired=(), blocked=(), available=()):
        self.replies = list(replies)
        self.retired = set(retired)
        self.blocked = set(blocked)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(
            data=[SimpleNamespace(id=m, active=True) for m in available]))

    def create(self, model, **kwargs):
        self.calls.append({"model": model, **kwargs})
        if model in self.retired:
            raise FakeStatusError(f"Error code: 400 - The model `{model}` has been decommissioned", 400)
        if model in self.blocked:
            raise FakeStatusError(
                f"Error code: 404 - The model `{model}` does not exist or you do not have access to it.", 404)
        return self.replies.pop(0)


@pytest.fixture
def ask_client(tmp_path, monkeypatch, fake_github):
    monkeypatch.setenv("ENABLE_ASK_FEATURE", "true")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    return _load_app(tmp_path, monkeypatch, fake_github, token=False)


def use_fake(monkeypatch, fake):
    monkeypatch.setattr(sys.modules["llm"], "Groq", lambda **kwargs: fake)


def test_answer_with_tools_and_page_links(ask_client, monkeypatch):
    fake = FakeGroq([
        reply(tool_calls=[tool_call("search_cpus", {"q": "9654"}),
                          tool_call("get_generation_info", {"codename": "Genoa"}, "call-2")]),
        reply("The [AMD EPYC 9654](/cpu/amd-epyc-9654) has 96 cores and uses socket SP5."),
    ])
    use_fake(monkeypatch, fake)
    response = ask_client.post("/api/ask", json={"question": "How many cores does the EPYC 9654 have?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "/cpu/amd-epyc-9654" in body["answer"]
    assert any(s.get("url") == "/cpu/amd-epyc-9654" for s in body["sources"])
    assert fake.calls[0]["model"] == "openai/gpt-oss-120b"

    system_prompt = fake.calls[0]["messages"][0]["content"]
    assert "/compare/" in system_prompt and "Report an error" in system_prompt and "'Genoa'" in system_prompt
    tool_messages = [m for m in fake.calls[1]["messages"] if m["role"] == "tool"]
    assert '"Socket": "SP5"' in tool_messages[1]["content"]


def test_retired_model_falls_back_to_the_next_one(ask_client, monkeypatch):
    fake = FakeGroq([reply("There are many CPUs.")], retired={"openai/gpt-oss-120b"})
    use_fake(monkeypatch, fake)
    response = ask_client.post("/api/ask", json={"question": "How many CPUs are there?"})
    assert response.status_code == 200, response.text
    assert [c["model"] for c in fake.calls] == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def test_uses_a_model_the_key_can_access_when_defaults_are_blocked(ask_client, monkeypatch):
    llm = sys.modules["llm"]
    fake = FakeGroq(
        [reply("There are many CPUs.")],
        blocked=set(llm.DEFAULT_GROQ_MODELS),
        available=["whisper-large-v3", "meta-llama/llama-prompt-guard-2-22m", "some-vendor/new-chat-model"],
    )
    use_fake(monkeypatch, fake)
    response = ask_client.post("/api/ask", json={"question": "How many CPUs are there?"})
    assert response.status_code == 200, response.text
    assert [c["model"] for c in fake.calls] == [*llm.DEFAULT_GROQ_MODELS, "some-vendor/new-chat-model"]
    assert llm.GROQ_MODEL == "some-vendor/new-chat-model"  # remembered for the next question


def test_provider_errors_use_a_status_cloudflare_passes_through(ask_client, monkeypatch):
    """Cloudflare swaps 502/504 bodies for its own page, so the chat would show nothing useful."""
    class Broken(FakeGroq):
        def create(self, model, **kwargs):
            raise FakeStatusError("Error code: 401 - Invalid API Key", 401)

    use_fake(monkeypatch, Broken([]))
    response = ask_client.post("/api/ask", json={"question": "Which GPU has the most memory?"})
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]
