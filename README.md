# 🏈 FF Analyzer

A Streamlit dashboard for Sleeper fantasy football leagues: waiver targets with FAAB bids, start/sit
comparisons, trade analysis, and roster-construction grades. It uses nflverse weekly stats scored with
your league's own settings, plus opponent defense, injuries, weather and (optionally) Vegas totals.
Optional Claude or Gemini narratives summarize the numbers.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # optional: AI keys, Odds API key, Sleeper defaults
streamlit run app.py
```

In the sidebar, open **➕ Sync from Sleeper** and enter your Sleeper username and league **name** (or the current
season's league ID). The name always finds this season's league; Sleeper issues a new ID every season.

## What each tab does

| Tab | What you get |
|---|---|
| 📋 Roster | Season/recent PPG, games, injury status and next matchup for every rostered player. Players with no stats this season are flagged instead of silently showing 0. |
| 🔄 Waivers | The full unrostered pool (sortable, filterable, injured players hidden by default). The deep dive gives a 0–100 value score, a FAAB bid from your remaining budget, a drop candidate, matchup/weather/usage/Vegas context and a Monte Carlo outcome distribution. |
| ⚖️ Who to Start | Compare 2–4 roster players on projected points: value × this week's opponent-defense multiplier × injury factor (an Out player is never recommended). |
| 🤝 Trade | Pick players to give from your roster and players to receive from any NFL team. The verdict comes from your **best starting lineup before vs. after**, so a 2-for-1 is judged by who would actually start. Per-player value over replacement and playoff-week schedule strength are also shown. |
| 🏗️ Roster Construction | Position grades against league-specific replacement levels, plus waiver targets boosted where you're weak. |

**🔄 Refresh data** in the sidebar re-downloads everything (stats, schedule, Sleeper players/injuries, odds).
The sidebar warns when your sync is more than 2 days old: the waiver pool is a snapshot and won't drop players other managers have claimed until you re-sync.

## How the numbers are built

- **Stats**: nflverse weekly player stats, regular season only. Before the current season has data, the app
  falls back to last season and says so.
- **Scoring**: fantasy points are computed from raw box-score columns using your league's Sleeper scoring
  (passing/rushing/receiving, 2-pt conversions, fumbles, kicking, position reception bonuses such as TE premium).
  Yardage-milestone bonuses are not modeled.
- **Player matching**: Sleeper `gsis_id` ↔ nflverse `player_id`, falling back to an exact normalized name
  (suffixes and accents ignored) disambiguated by position and team. There is no fuzzy or last-name matching.
- **Games and form**: every game a player recorded stats in counts, including 0-point games. "Recent" means the
  last 4 NFL weeks.
- **Value**: expected points = 65% recent + 35% season, adjusted ±30% for week-to-week consistency.
- **Replacement level**: the PPG of the first player past the league's rostered depth at each position. Depth is
  derived from team count and lineup slots (FLEX and SUPER_FLEX included).
- **Monte Carlo**: a zero-inflated Gamma fitted to each player's weekly scores (10,000 draws). Samples are seeded
  per player, so every chart and probability for that player uses the same draws.

## Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | Enables AI narratives (either one). |
| `CLAUDE_MODEL` / `GEMINI_MODEL` | Override model IDs (defaults: `claude-sonnet-5-5`, `gemini-2.5-flash`). |
| `ODDS_API_KEY` | Vegas implied team totals ([the-odds-api.com](https://the-odds-api.com), free tier). |
| `SLEEPER_USERNAME` / `SLEEPER_LEAGUE` | Pre-fill the sync form. |

## Project layout

```
app.py                     Streamlit UI
src/
  scoring.py               League scoring → fantasy points (vectorized)
  wrangler.py              nflverse stats download + per-player StatsIndex
  players.py               Name/team normalization, Sleeper ↔ nflverse ID linking
  sleeper_integration.py   League sync, FAAB budget, waiver pool, shared player DB
  evaluator.py             Value model, waiver scores, start/sit, lineup-based trades, roster grades
  simulator.py             Monte Carlo simulations
  defense.py               Points allowed by defense × position
  schedule.py              Next games, playoff schedule strength, weather
  injuries.py              Injury statuses from the Sleeper player DB
  vegas.py                 Implied team totals
  state_manager.py         League files (leagues/<league_id>.json)
  storage.py               Cache paths and atomic file writes
tests/                     pytest suite (python -m pytest)
```

Caches live in `data/` and league files in `leagues/`. Both are plain files, and you can delete `data/` at any time.

## Not yet implemented

Screenshot roster parsing, Yahoo/ESPN sync, and natural-language roster edits from the original PRD
(`FF-Analyzer (PRD).md`) are not built. Sleeper is the only supported platform.
