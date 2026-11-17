"""Audit fixes: start/sit model verdict, shared output cap, safe errors, provider helpers, fmt_as_of."""

import sys
import types

from src import agent
from src.agent import FantasyAgent, PROVIDER_LABELS, available_providers
from src.formatting import fmt_age, fmt_as_of

STATS = {"found": True, "season": {"ppg": 12.3, "games": 6}, "recent": {"ppg": 15.0, "weeks": 4},
         "weekly": [10, 12, 14]}


class Recorder(FantasyAgent):
    def analyze(self, prompt):
        self.prompt = prompt
        return "ok"


def test_start_sit_with_model_summary_asks_to_explain_app_pick():
    a = Recorder()
    a.analyze_start_sit("P1", "P2", STATS, STATS, model_summary={
        "Projection (this week)": "P1 14.2 vs P2 11.8", "P(P1 outscores P2)": "63%",
        "Style": "Balanced", "App's pick": "P1"})
    p = a.prompt
    assert "My model's assessment" in p and "App's pick: P1" in p and "P(P1 outscores P2): 63%" in p
    assert "Explain the app's pick" in p and "disagree" in p and "season PPG" in p
    assert p.rstrip().endswith("START: [Player Name]")


def test_start_sit_without_model_summary_keeps_plain_prompt():
    a = Recorder()
    a.analyze_start_sit("P1", "P2", STATS, STATS)
    assert "My model's assessment" not in a.prompt and "Give a concise comparison" in a.prompt


def _install_fake_genai(monkeypatch, generate):
    client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate))
    genai = types.SimpleNamespace(Client=lambda api_key: client,
                                  types=types.SimpleNamespace(GenerateContentConfig=lambda **kw: kw))
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai.types)


def test_gemini_gets_shared_output_cap(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    seen = {}

    def generate(**kw):
        seen.update(kw)
        return types.SimpleNamespace(text="fine")
    _install_fake_genai(monkeypatch, generate)
    assert FantasyAgent("gemini").analyze("Q") == "fine"
    assert seen["config"]["max_output_tokens"] == agent.MAX_OUTPUT_TOKENS == 1024
    assert seen["config"]["system_instruction"] == agent.SYSTEM_PROMPT


def test_claude_uses_shared_output_cap(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    seen = {}

    def create(**kw):
        seen.update(kw)
        return types.SimpleNamespace(content=[])
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        Anthropic=lambda api_key: types.SimpleNamespace(messages=types.SimpleNamespace(create=create))))
    FantasyAgent("claude").analyze("Q")
    assert seen["max_tokens"] == agent.MAX_OUTPUT_TOKENS


def test_errors_are_redacted_and_truncated(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")

    def generate(**kw):
        raise RuntimeError("bad key AIzaSyABC123_-x and sk-ant-api03-SECRET " + "z" * 1000)
    _install_fake_genai(monkeypatch, generate)
    out = FantasyAgent("gemini").analyze("Q")
    assert "AI analysis failed" in out and "AIza" not in out and "sk-ant" not in out
    assert "[redacted]" in out and out.endswith("…")
    assert len(out) < agent.MAX_ERROR_CHARS + 60


def test_short_error_is_not_truncated():
    assert agent._safe_error(RuntimeError("overloaded")) == "overloaded"


def test_missing_key_mentions_streamlit_secrets(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "Streamlit Secrets" in FantasyAgent("claude").analyze("hi")


def test_available_providers_and_labels(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert available_providers() == []
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    assert available_providers() == ["gemini"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert available_providers() == ["claude", "gemini"]
    assert PROVIDER_LABELS == {"claude": "Claude", "gemini": "Gemini"}


def test_fmt_age_edge_cases_and_as_of():
    assert fmt_age(None) == "just now" and fmt_age(-5) == "just now"
    assert fmt_age(45) == "1 minute ago"  # round(0.75)=1; also guards the 30-89s window
    assert fmt_as_of("Injuries", 3 * 3600) == "Injuries as of 3 hours ago"
    assert fmt_as_of("Injuries", 0) == "Injuries as of just now"
