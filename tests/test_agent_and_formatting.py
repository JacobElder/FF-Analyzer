"""AI agent prompt building and error handling (mocked clients); UI formatting helpers."""

import sys
import types

import pytest

from src import agent
from src.agent import FantasyAgent, _fmt_context, _fmt_stats
from src.formatting import fmt_age, grade_color, lineup_summary, verdict_color

STATS = {"found": True, "season": {"ppg": 12.3, "games": 6}, "recent": {"ppg": 15.0, "weeks": 4},
         "weekly": [10, 12, 14, 9, 20, 9]}


class Recorder(FantasyAgent):
    """Captures the prompt instead of calling an API."""
    def analyze(self, prompt):
        self.prompt = prompt
        return "ok"


def test_missing_key_returns_warning_without_calling_api(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # importing would raise
    assert "ANTHROPIC_API_KEY" in FantasyAgent("claude").analyze("hi")
    assert "GOOGLE_API_KEY" in FantasyAgent("gemini").analyze("hi")


def test_unknown_provider_is_reported():
    assert "Unknown AI provider" in FantasyAgent("gpt").analyze("hi")


def _fake_anthropic(monkeypatch, create):
    mod = types.SimpleNamespace(Anthropic=lambda api_key: types.SimpleNamespace(
        messages=types.SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "anthropic", mod)


def test_claude_joins_only_text_blocks_and_passes_system_prompt(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    seen = {}

    def create(**kw):
        seen.update(kw)
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="thinking", text="hidden"),
                                              types.SimpleNamespace(type="text", text="Start "),
                                              types.SimpleNamespace(type="text", text="him.")])
    _fake_anthropic(monkeypatch, create)
    assert FantasyAgent("claude").analyze("Q") == "Start him."
    assert seen["system"] == agent.SYSTEM_PROMPT and seen["model"] == agent.CLAUDE_MODEL
    assert seen["messages"] == [{"role": "user", "content": "Q"}]


def test_api_errors_become_messages(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    def create(**kw):
        raise RuntimeError("overloaded")
    _fake_anthropic(monkeypatch, create)
    assert "AI analysis failed" in FantasyAgent("claude").analyze("Q")


def test_gemini_empty_response_has_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    client = types.SimpleNamespace(models=types.SimpleNamespace(
        generate_content=lambda **kw: types.SimpleNamespace(text=None)))
    genai = types.SimpleNamespace(Client=lambda api_key: client,
                                  types=types.SimpleNamespace(GenerateContentConfig=lambda **kw: kw))
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai.types)
    assert "empty response" in FantasyAgent("gemini").analyze("Q")


def test_fmt_helpers():
    assert _fmt_context({"a": None, "b": "", "c": []}) == "  (no additional context)"
    assert _fmt_context({"Matchup": "vs BUF", "x": None}) == "  Matchup: vs BUF"
    assert _fmt_stats({}) == "  No stats found for this season."
    assert _fmt_stats({"found": False}) == "  No stats found for this season."
    assert "did not play" in _fmt_stats({**STATS, "recent": {"ppg": None}})
    assert "9, 20, 9" in _fmt_stats(STATS, last_n=3) and "10," not in _fmt_stats(STATS, last_n=3)


def test_prompts_include_key_facts_and_closing_line():
    a = Recorder()
    a.analyze_waiver("Joe", STATS, [{"name": "Me1"}], {"rec": 1}, budget=37, context={"Injury": "Q"})
    assert all(s in a.prompt for s in ("Joe", "PPR league", "$37", "Me1", "Injury: Q", "RECOMMENDATION:"))
    a.analyze_start_sit("P1", "P2", STATS, {}, {"rec": 0})
    assert "Standard league" in a.prompt and "START:" in a.prompt and "No stats found" in a.prompt
    a.analyze_trade([], ["Star"], {"Star": STATS}, trade_summary={"Verdict": "ACCEPT"})
    assert "(nothing)" in a.prompt and "Star" in a.prompt and "VERDICT:" in a.prompt


@pytest.mark.parametrize("seconds,text", [(10, "1 min ago"), (300, "5 min ago"), (3599, "1h ago"),
                                          (7200, "2h ago"), (86000, "1 days ago"), (3 * 86400, "3 days ago")])
def test_fmt_age(seconds, text):
    assert fmt_age(seconds) == text


def test_lineup_summary_and_colors():
    positions = ["QB", "RB", "RB", "FLEX", "BN", "BN", "IR"]
    assert lineup_summary(positions) == "QB, 2 RB, FLEX (+ 2 bench, 1 IR)"
    assert lineup_summary(["QB"]) == "QB"
    assert verdict_color("ACCEPT") == "green" and verdict_color("??") == "gray"
    assert grade_color("B") == "blue" and grade_color(None) == "gray"
