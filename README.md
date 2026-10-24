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
| 📋 Roster | Projected score for both lineups this week and your win chance, **lineup fixes** that name who to start in place of an Out, bye-week or weaker starter (and the win chance after), every player with this week's and rest-of-season projections, and league standings with your playoff position. |
| 🔄 Waivers | **Bid recommendations**: every add that would improve your best starting lineup, with its gain, the player to drop, what he's worth to you, the suggested bid and its chance to win. Below that, the browsable pool and a deep dive with the lineup gain by week, the win-probability curve by bid, which rivals also want him, plus matchup/weather/usage/Vegas context. |
| 🧾 FAAB Planner | This week's claims in priority order, each with its own drop and bid, within your budget; the waiver deadline; what this league pays for adds; every team's remaining FAAB and bidding style; and your FAAB log (cost vs. points scored since). |
| ⚖️ Who to Start | Compare 2–4 roster players on this week's projection. Choose **Safe** (rank by floor), **Balanced** (projection) or **Upside** (ceiling); it defaults from your matchup win chance — favorites play it safe, underdogs chase upside. |
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
  last 6 NFL weeks.
- **History value**: expected points = 35% recent + 65% season. Backtested on 2025 (`python -m src.backtest`), this
  predicted the next 4 weeks better than heavier recent weighting.
- **Weekly projection**: 80% Sleeper's weekly projection (scored with your league's settings) + 20% history value
  × opponent-defense multiplier — the most accurate blend in the backtest. It is 0 on bye weeks and while injured,
  using estimated absences (Out 1 week, suspension 2, IR/PUP 4; Sleeper gives no return dates). Rookies with no
  history use the projection alone. Defense multipliers are pulled toward average early in the season.
  The projections endpoint is unofficial; if it stops working the app says so and uses history only.
- **Replacement level**: the PPG of the first player past the league's rostered depth at each position. Depth is
  derived from team count and lineup slots (FLEX and SUPER_FLEX included).
- **Monte Carlo**: a zero-inflated Gamma fitted to each player's weekly scores (10,000 draws). Samples are seeded
  per player, so every chart and probability for that player uses the same draws. Start/sit centers each
  player's distribution on this week's projection.

## How waiver bids are built

1. **Lineup gain**: for each of the top 60 available players, the app adds him, drops whichever player costs your
   lineup the least, and sums the improvement in your best starting lineup over every week through the fantasy
   championship. DEF and K count only the next 2 weeks, since they can be streamed for $0. Backup RBs get a small
   insurance bonus when you roster their starter, and a "promoted" note when a player ahead of them on the depth
   chart is out.
2. **Worth to you**: that gain compared with a typical good add this week, times your budget spread over the
   claims you can still make (about one every two weeks). Late in the season each claim is worth more.
3. **Competition**: the same lineup-gain math runs on every rival roster. A rival "wants" a player who adds ≥0.5
   pts/week to their lineup. The competing bid is priced from this league's own winning bids for an add of that
   size, adjusted for the most aggressive interested manager and capped by their FAAB.
4. **Bid**: the cheapest bid with a 75% chance to win, never above what he's worth to you; the league minimum when
   nobody else wants him.
5. **Plan**: claims are chosen greedily by expected gain. Each later claim is re-valued as if the earlier ones went
   through and repriced at that value, so the plan won't stack two players for one spot.

The rival-behavior constants in `src/market.py` (bid probability, noise, aggression weight) are judgment calls
tuned against league history, not backtested. Treat bids as a well-informed starting point.

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
  evaluator.py             History value model, generic waiver score, lineup optimizer, trades, roster grades
  projections.py           Sleeper weekly projections, scored with league settings
  valuation.py             Week-by-week player values, lineup gains, lineup fixes
  market.py                League bid history, rival interest, win curves, bids, claim planner, standings
  waiver_analysis.py       Runs the above over the waiver pool and builds the claim plan
  backtest.py              2025 backtest of the projection/history blend (python -m src.backtest 2025)
  simulator.py             Monte Carlo simulations
  defense.py               Points allowed by defense × position
  schedule.py              Next games, playoff schedule strength, weather
  injuries.py              Injury statuses from the Sleeper player DB
  vegas.py                 Implied team totals
  state_manager.py         League files (leagues/<league_id>.json)
  storage.py               Cache paths and atomic file writes
tests/                     pytest suite (see Testing below)
```

Caches live in `data/` and synced league files in `leagues/`. Neither is committed to git, and both are rebuilt
automatically if you delete them.

## Testing

```
pip install -r requirements-dev.txt
python -m pytest                                   # whole suite, offline
python -m coverage run -m pytest && python -m coverage report   # fails under 80% branch coverage
```

Tests never touch the network or your real `data/`, `leagues/` or `.env`: `tests/conftest.py` points the data
directories at a temp folder, clears API-key and Sleeper env vars, and blocks sockets for every test.

## Not yet implemented

Screenshot roster parsing, Yahoo/ESPN sync, and natural-language roster edits from the original PRD
(`FF-Analyzer (PRD).md`) are not built. Sleeper is the only supported platform.
