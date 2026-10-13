# 🏈 FF Analyzer

A Streamlit dashboard for Sleeper fantasy football leagues: waiver targets with FAAB bids, start/sit
comparisons, trade analysis, and roster-construction grades. It uses nflverse weekly stats scored with
your league's own settings, plus opponent defense, injuries, weather and (optionally) Vegas totals.
Optional Claude or Gemini narratives summarize the numbers.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # optional: AI keys, Odds API key
streamlit run app.py
```

Your leagues are listed in `config.toml`:

```toml
[sleeper]
username = "JElder52"
league_ids = ["1391496495336808448", "1389377870526177280"]
auto_sync_minutes = 30
```

On load, the app syncs each listed league from Sleeper whenever its saved copy is older than
`auto_sync_minutes`. A sync pulls your roster and current lineup slots, team name, record, this week's matchup,
remaining FAAB, scoring, lineup settings and the waiver pool. If Sleeper is unreachable, the last saved copy is used
and the sidebar says so. **🔄 Sync this league now** forces an update. To track another league once, use
**➕ Add another Sleeper league**; to keep it synced, add its ID to `config.toml`. Sleeper issues new league IDs each
season, so update the IDs when a new season starts.

## What each tab does

| Tab | What you get |
|---|---|
| 📋 Roster | This week's matchup score, then every player in your Sleeper lineup order (slot, season/recent PPG, injury, next opponent). It flags injured starters and empty lineup slots, and marks players with no stats this season instead of showing 0. |
| 🔄 Waivers | The full unrostered pool (sortable, filterable, injured players hidden by default). The deep dive gives a 0–100 value score, a FAAB bid from your remaining budget, a drop candidate, matchup/weather/usage/Vegas context and a Monte Carlo outcome distribution. |
| ⚖️ Who to Start | Compare 2–4 roster players on projected points: value × this week's opponent-defense multiplier × injury factor (an Out player is never recommended). |
| 🤝 Trade | Pick players to give from your roster and players to receive from any NFL team. The verdict comes from your **best starting lineup before vs. after**, so a 2-for-1 is judged by who would actually start. Per-player value over replacement and playoff-week schedule strength are also shown. |
| 🏗️ Roster Construction | Grades for the positions your league's lineup actually uses, measured against league-specific replacement levels, plus waiver targets boosted where you're weak. |

The sidebar shows your team, record and a **📜 League Settings** summary (teams, lineup, scoring, waivers, playoff
weeks). **🔄 Refresh data** re-downloads the NFL stats, schedule, Sleeper player/injury data and odds.

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
| `SLEEPER_USERNAME` / `SLEEPER_LEAGUE_IDS` / `SLEEPER_AUTO_SYNC_MINUTES` | Override `config.toml` (IDs comma-separated). |

## Project layout

```
app.py                     Streamlit UI
src/
  scoring.py               League scoring → fantasy points (vectorized)
  wrangler.py              nflverse stats download + per-player StatsIndex
  players.py               Name/team normalization, Sleeper ↔ nflverse ID linking
  config.py                config.toml + .env settings
  league_sync.py           Auto-sync of the configured leagues
  sleeper_integration.py   League sync (lineup slots, record, matchup, FAAB, waiver pool), shared player DB
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

Caches live in `data/` and synced league files in `leagues/`. Neither is committed to git, and both are rebuilt
automatically if you delete them.

## Not yet implemented

Screenshot roster parsing, Yahoo/ESPN sync, and natural-language roster edits from the original PRD
(`FF-Analyzer (PRD).md`) are not built. Sleeper is the only supported platform.
