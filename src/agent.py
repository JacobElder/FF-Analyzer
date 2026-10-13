"""
Optional LLM narrative analysis (Claude or Gemini).

Prompts carry the app's own computed context (matchup, injury, Vegas, usage,
model scores) and the system prompt forbids inventing news, because model
training data is older than this week's injury report.

Model IDs are configurable via CLAUDE_MODEL / GEMINI_MODEL in .env.
"""

import os

from dotenv import load_dotenv

from src.scoring import DEFAULT_SCORING, scoring_label

load_dotenv()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_KEY_ENV = {"claude": "ANTHROPIC_API_KEY", "gemini": "GOOGLE_API_KEY"}

SYSTEM_PROMPT = (
    "You are a concise fantasy football analyst. Base your analysis ONLY on the data in the "
    "user's message. Your training data is out of date: do not cite injuries, depth-chart "
    "changes, trades, suspensions, or news that are not in the provided data. If something "
    "important is missing (e.g. no stats found), say so plainly instead of guessing."
)


def _fmt_context(context: dict | None) -> str:
    if not context:
        return "  (no additional context)"
    return "\n".join(f"  {k}: {v}" for k, v in context.items() if v not in (None, "", []))


def _fmt_stats(stats: dict, last_n: int = 6) -> str:
    if not stats or not stats.get("found", True) and not stats.get("weekly"):
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
        key_name = _KEY_ENV.get(self.provider, "ANTHROPIC_API_KEY")
        api_key = os.getenv(key_name)
        if not api_key:
            return f"⚠️ **API key not configured.** Add `{key_name}` to your `.env` file."
        try:
            if self.provider == "claude":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=1024,
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
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            return resp.text or "⚠️ The model returned an empty response."
        except ImportError as e:
            return f"⚠️ **AI package not installed:** {e}. Run `pip install -r requirements.txt`."
        except Exception as e:
            return f"⚠️ **AI analysis failed:** {e}"

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

    def analyze_start_sit(self, p1_name, p2_name, p1_stats, p2_stats, scoring=None, contexts=None) -> str:
        scoring = scoring or DEFAULT_SCORING
        contexts = contexts or {}
        prompt = f"""Help me decide who to start this week in a {scoring_label(scoring)} league.

Player 1: {p1_name}
{_fmt_stats(p1_stats, 4)}
{_fmt_context(contexts.get(p1_name))}

Player 2: {p2_name}
{_fmt_stats(p2_stats, 4)}
{_fmt_context(contexts.get(p2_name))}

Give a concise comparison (3-4 sentences) covering recent form, matchup, and injury status.
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
