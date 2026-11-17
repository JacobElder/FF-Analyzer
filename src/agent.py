"""
Optional LLM narrative analysis (Claude or Gemini).

Prompts carry the app's own computed context (matchup, injury, Vegas, usage,
model scores) and the system prompt forbids inventing news, because model
training data is older than this week's injury report.

Model IDs are configurable via CLAUDE_MODEL / GEMINI_MODEL in .env.
"""

import os
import re

from dotenv import load_dotenv

from src.scoring import DEFAULT_SCORING, scoring_label

load_dotenv()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MAX_OUTPUT_TOKENS = 1024  # shared output cap for both providers
MAX_ERROR_CHARS = 300     # provider error text shown to the user is truncated to this

_KEY_ENV = {"claude": "ANTHROPIC_API_KEY", "gemini": "GOOGLE_API_KEY"}
PROVIDER_LABELS = {"claude": "Claude", "gemini": "Gemini"}

# Anything shaped like an Anthropic (sk-ant-...) or Google (AIza...) key is redacted from errors.
_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]+|AIza[A-Za-z0-9_\-]+")


def available_providers() -> list[str]:
    """Provider IDs (in PROVIDER_LABELS order) whose API key is set in the environment."""
    return [p for p in PROVIDER_LABELS if os.getenv(_KEY_ENV[p])]


def _safe_error(e: Exception) -> str:
    """str(e) with key-like substrings redacted, truncated to MAX_ERROR_CHARS."""
    text = _KEY_PATTERN.sub("[redacted]", str(e))
    return text if len(text) <= MAX_ERROR_CHARS else text[:MAX_ERROR_CHARS].rstrip() + "…"

SYSTEM_PROMPT = (
    "You are a concise fantasy football analyst. Base your analysis ONLY on the data in the "
    "user's message. Your training data is out of date: do not cite injuries, depth-chart "
    "changes, trades, suspensions, or news that are not in the provided data. If something "
    "important is missing (e.g. no stats found), say so plainly instead of guessing."
)


def _fmt_context(context: dict | None) -> str:
    lines = [f"  {k}: {v}" for k, v in (context or {}).items() if v not in (None, "", [])]
    return "\n".join(lines) if lines else "  (no additional context)"


def _fmt_stats(stats: dict, last_n: int = 6) -> str:
    if not stats or (not stats.get("found", True) and not stats.get("weekly")):
        return "  No stats found for this season."
    s = stats.get("season") or {}
    r = stats.get("recent") or {}
    weekly = stats.get("weekly") or []
    weekly_str = ", ".join(str(round(w, 1)) for w in weekly[-last_n:]) if weekly else "no games"
    recent = r.get("ppg")
    return (f"  Season PPG: {s.get('ppg', 'N/A')} over {s.get('games', 0)} games\n"
            f"  Last {r.get('weeks', 4)} weeks PPG: {recent if recent is not None else 'did not play'}\n"
            f"  Most recent game scores: {weekly_str}")


class FantasyAgent:
    def __init__(self, provider: str = "claude"):
        self.provider = provider

    def analyze(self, prompt: str) -> str:
        key_name = _KEY_ENV.get(self.provider)
        if key_name is None:
            return f"⚠️ **Unknown AI provider:** {self.provider!r}. Use 'claude' or 'gemini'."
        api_key = os.getenv(key_name)
        if not api_key:
            return f"⚠️ **API key not configured.** Add `{key_name}` to your `.env` file or Streamlit Secrets."
        try:
            if self.provider == "claude":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT, max_output_tokens=MAX_OUTPUT_TOKENS),
            )
            return resp.text or "⚠️ The model returned an empty response."
        except ImportError as e:
            return f"⚠️ **AI package not installed:** {e}. Run `pip install -r requirements.txt`."
        except Exception as e:
            return f"⚠️ **AI analysis failed:** {_safe_error(e)}"

    def analyze_waiver(self, player_name, stats, roster, scoring=None, budget=100, context=None) -> str:
        scoring = scoring or DEFAULT_SCORING
        roster_names = ", ".join(p["name"] for p in roster) if roster else "Unknown"
        prompt = f"""Evaluate this waiver wire pickup in a {scoring_label(scoring)} league.

Player: {player_name}
{_fmt_stats(stats)}
Context from my data:
{_fmt_context(context)}
Available FAAB budget: ${budget}
My current roster: {roster_names}

Give a concise analysis (3-5 sentences): is this a worthwhile pickup, and why or why not?
End with: RECOMMENDATION: [ADD / STREAM / AVOID]"""
        return self.analyze(prompt)

    def analyze_start_sit(self, p1_name, p2_name, p1_stats, p2_stats, scoring=None, contexts=None,
                          model_summary=None) -> str:
        """model_summary: optional dict of the app's own verdict, e.g. {"Projection (this week)":
        "P1 14.2 vs P2 11.8", "Floor / ceiling": ..., "P(P1 outscores P2)": "63%",
        "Style": "Balanced", "App's pick": "P1"}. When given, the model explains that pick."""
        scoring = scoring or DEFAULT_SCORING
        contexts = contexts or {}
        if model_summary:
            verdict = f"""
My model's assessment (this week's projections, not season averages):
{_fmt_context(model_summary)}
"""
            task = ("Explain the app's pick above in a concise comparison (3-4 sentences) covering this "
                    "week's projection, floor/ceiling, recent form, matchup, and injury status. Do not "
                    "substitute a different pick based on season PPG alone. If the data above gives you a "
                    "concrete reason to disagree with the app's pick, say so explicitly and explain why.")
        else:
            verdict = ""
            task = "Give a concise comparison (3-4 sentences) covering recent form, matchup, and injury status."
        prompt = f"""Help me decide who to start this week in a {scoring_label(scoring)} league.

Player 1: {p1_name}
{_fmt_stats(p1_stats, 4)}
{_fmt_context(contexts.get(p1_name))}

Player 2: {p2_name}
{_fmt_stats(p2_stats, 4)}
{_fmt_context(contexts.get(p2_name))}
{verdict}
{task}
End with: START: [Player Name]"""
        return self.analyze(prompt)

    def analyze_trade(self, giving, receiving, stats_map, scoring=None, contexts=None, trade_summary=None) -> str:
        scoring = scoring or DEFAULT_SCORING
        contexts = contexts or {}

        def summarize(names):
            if not names:
                return "  (nothing)"
            return "\n".join(f"- {n}\n{_fmt_stats(stats_map.get(n, {}), 4)}\n{_fmt_context(contexts.get(n))}"
                             for n in names)

        prompt = f"""Evaluate this trade in a {scoring_label(scoring)} league.

YOU GIVE:
{summarize(giving)}

YOU RECEIVE:
{summarize(receiving)}

My model's assessment:
{_fmt_context(trade_summary)}

Analyze the value of each side based on the stats and context above (4-5 sentences).
End with: VERDICT: [ACCEPT / DECLINE / NEGOTIATE]"""
        return self.analyze(prompt)
