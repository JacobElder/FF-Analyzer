import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SCORING = {
    "rec": 0.5,
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_int": -2.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "fum_lost": -2.0,
}


class FantasyAgent:
    def __init__(self, provider="claude"):
        self.provider = provider
        if provider == "claude":
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self._model = genai.GenerativeModel("gemini-2.0-flash")

    def analyze(self, prompt: str) -> str:
        try:
            if self.provider == "claude":
                msg = self._client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text
            else:
                response = self._model.generate_content(prompt)
                return response.text
        except TypeError as e:
            key = "ANTHROPIC_API_KEY" if self.provider == "claude" else "GOOGLE_API_KEY"
            return f"⚠️ **API key not configured.** Add `{key}` to your `.env` file.\n\nError: {e}"
        except Exception as e:
            return f"⚠️ **AI analysis failed:** {e}"

    def analyze_waiver(self, player_name, stats, roster, scoring=None, budget=100) -> str:
        scoring = scoring or DEFAULT_SCORING
        rec_str = "PPR" if scoring.get("rec", 0) >= 1 else ("Half-PPR" if scoring.get("rec", 0) >= 0.5 else "Standard")
        roster_names = ", ".join(p["name"] for p in roster) if roster else "Unknown"
        season_ppg = stats.get("season", {}).get("ppg", "N/A")
        recent_ppg = stats.get("recent", {}).get("ppg", "N/A")
        weekly = stats.get("weekly", [])
        weekly_str = ", ".join(str(round(w, 1)) for w in weekly[-6:]) if weekly else "No data"

        prompt = f"""You are a fantasy football analyst. Evaluate this waiver wire pickup in a {rec_str} league.

Player: {player_name}
Season PPG: {season_ppg}
Last 4 weeks PPG: {recent_ppg}
Recent weekly scores: {weekly_str}
Available FAAB budget: ${budget}
My current roster: {roster_names}

Provide a concise analysis (3-5 sentences): Is this a worthwhile pickup? Why or why not?
Mention matchup trends, role, and any injury/usage context you know about.
End with: RECOMMENDATION: [ADD / STREAM / AVOID]"""
        return self.analyze(prompt)

    def analyze_start_sit(self, p1_name, p2_name, p1_stats, p2_stats, scoring=None) -> str:
        scoring = scoring or DEFAULT_SCORING
        rec_str = "PPR" if scoring.get("rec", 0) >= 1 else ("Half-PPR" if scoring.get("rec", 0) >= 0.5 else "Standard")

        def fmt(stats):
            s = stats.get("season", {})
            r = stats.get("recent", {})
            w = stats.get("weekly", [])
            weekly_str = ", ".join(str(round(x, 1)) for x in w[-4:]) if w else "No data"
            return f"Season PPG: {s.get('ppg', 'N/A')}, Recent PPG: {r.get('ppg', 'N/A')}, Last 4 weeks: {weekly_str}"

        prompt = f"""You are a fantasy football analyst. Help me decide who to start in a {rec_str} league.

Player 1: {p1_name}
{fmt(p1_stats)}

Player 2: {p2_name}
{fmt(p2_stats)}

Give a concise comparison (3-4 sentences) covering recent form, role, and any matchup factors.
End with: START: [Player Name]"""
        return self.analyze(prompt)

    def analyze_trade(self, giving, receiving, stats_map, scoring=None) -> str:
        scoring = scoring or DEFAULT_SCORING
        rec_str = "PPR" if scoring.get("rec", 0) >= 1 else ("Half-PPR" if scoring.get("rec", 0) >= 0.5 else "Standard")

        def summarize(names):
            lines = []
            for name in names:
                s = stats_map.get(name, {})
                ppg = s.get("season", {}).get("ppg", "N/A")
                recent = s.get("recent", {}).get("ppg", "N/A")
                lines.append(f"  {name}: Season PPG {ppg}, Recent PPG {recent}")
            return "\n".join(lines) if lines else "  (no stats available)"

        prompt = f"""You are a fantasy football analyst. Evaluate this trade in a {rec_str} league.

YOU GIVE:
{summarize(giving)}

YOU RECEIVE:
{summarize(receiving)}

Analyze the value of each side based on stats, age, role, and trajectory (4-5 sentences).
End with: VERDICT: [ACCEPT / DECLINE / NEGOTIATE]"""
        return self.analyze(prompt)
