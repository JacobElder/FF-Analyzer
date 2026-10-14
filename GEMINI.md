# FF-Analyzer: agent notes

Streamlit dashboard for Sleeper fantasy football leagues. See README.md for features and setup.

## Run / test
- App: `streamlit run app.py`
- Tests: `python -m pytest`

## Architecture
- `app.py`: UI only. Heavy objects are built once per scoring configuration in `load_analytics()`
  (`st.cache_resource`), and per-player lookups are O(1) through `StatsIndex`.
- `src/scoring.py`: the single place fantasy points are computed. Don't reintroduce `fantasy_points` /
  `fantasy_points_ppr` column shortcuts; they ignore league settings.
- `src/players.py`: identity. Always look players up by player dict (`player_id`, `gsis_id`, `name`,
  `position`, `team`), not bare names. Matching must stay strict: no last-name or substring fallbacks.
- `src/sleeper_integration.py`: the only module that downloads `/players/nfl` (shared cache, ≤ 2×/day).
- League files: `leagues/<league_id>.json` (git-ignored), written only through `state_manager.save_league()`.
  Default leagues come from `config.toml` and are refreshed by `src/league_sync.py` when older than
  `auto_sync_minutes`.

- Waiver/FAAB engine: `projections.py` → `valuation.py` (ValuationContext.week_values is the single source of
  weekly player values) → `market.py` → `waiver_analysis.py`. In the app, `load_context` / `load_waiver_engine`
  cache it per league sync and budget.

## Conventions
- Team abbreviations are Sleeper style (`LAR`, `JAX`, `WAS`); normalize external data with `norm_team()`.
- Stats are regular season only; a player's games include 0-point games.
- File writes go through `src/storage.py` atomic helpers.
- AI prompts must pass the app's computed context; the system prompt forbids inventing news.
