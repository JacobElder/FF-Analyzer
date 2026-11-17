import hmac
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.agent import PROVIDER_LABELS, FantasyAgent, available_providers
from src.config import load_config
from src.defense import EMPTY_MATCHUP, build_matchup_table, compute_defense_rankings, get_matchup_info
from src.evaluator import (
    DEFAULT_ROSTER_POSITIONS, analyze_roster_construction, compute_positional_scarcity,
    injury_factor, player_value, recommend_drop, score_trade, score_waiver, starter_counts,
)
from src.formatting import fmt_age, fmt_as_of, grade_color, lineup_summary, verdict_color
from src.injuries import build_injury_map, injury_status, is_inactive
from src.league_sync import PARTIAL_RETRY_S, auto_sync, leagues_due, sync_age_seconds
from src.live import game_started
from src.market import (
    TARGET_WIN_PROB, faab_log, playoff_position, risk_mode, standings, waiver_day,
)
from src.players import RELEVANT_POSITIONS, build_sleeper_name_index, enrich_with_ids, player_label, sleeper_to_player
from src.schedule import (
    LAST_NFL_WEEK, fetch_schedules, get_playoff_schedule_strength, get_weather_for_game, next_game_map,
    playoff_weeks_for,
)
from src.season import (
    all_week_weights, combined_risk_prob, find_trades, league_dists, median_win_prob, simulate_season, team_week_dist,
    week_weights, weeks_until_deadline,
)
from src.scoring import compute_fantasy_points, effective_scoring, scoring_key, scoring_label
from src.simulator import (
    SCORE_CV, _summary, simulate_lineup_change, simulate_player, simulate_trade, win_shares,
)
from src.sleeper_integration import (
    GAMEDAY_PLAYERS_TTL, fetch_all_players, players_db_age_seconds, players_db_due, sync_sleeper_league,
)
from src.state_manager import list_leagues, load_league, save_league
from src.storage import DATA_DIR, DataUnavailable
from src.projections import Projections
from src.valuation import (
    NON_LINEUP_SLOTS, ValuationContext, fantasy_weeks, lineup_suggestions, lineup_totals, team_week_opponents,
)
from src.vegas import fetch_implied_totals, vegas_status
from src.waiver_analysis import analyze_waiver_pool, build_claim_plan, faab_enabled as league_uses_faab
from src.wrangler import StatsIndex, current_nfl_season, fetch_weekly_stats

load_dotenv()

st.set_page_config(page_title="FF Analyzer", layout="wide")
st.title("🏈 FF Analyzer")

STALE_SYNC_DAYS = 2
MAX_PASSWORD_ATTEMPTS = 5     # wrong AI passwords per session before a lockout
PASSWORD_LOCKOUT_S = 30       # first lockout; doubles with every further wrong password
MAX_AI_CALLS = 30             # AI requests per browser session
PLAYERS_RETRY_S = 10 * 60     # while the player DB can't be refreshed, retry the download this often


# ── Cached loaders ────────────────────────────────────────────────────────────
# Large read-only objects use cache_resource (no per-rerun copy); the file
# caches underneath keep data across app restarts.

@st.cache_resource(ttl=6 * 3600, show_spinner="Loading NFL stats…")
def load_weekly_stats():
    return fetch_weekly_stats()

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_schedules():
    return fetch_schedules()

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_next_games():
    return next_game_map(load_schedules())

@st.cache_data(ttl=1800, show_spinner=False)  # the file cache underneath holds odds for 4 hours
def load_vegas():
    return fetch_implied_totals()

@st.cache_resource(ttl=GAMEDAY_PLAYERS_TTL, show_spinner="Loading Sleeper player database…")
def load_players_db():
    db = fetch_all_players()
    name_index = build_sleeper_name_index(db)
    injuries = build_injury_map(db)
    trade_pool = sorted(
        (sleeper_to_player(pid, info) for pid, info in db.items()
         if info.get("active") and info.get("team") and info.get("position") in RELEVANT_POSITIONS),
        key=lambda p: p["name"],
    )
    return db, name_index, injuries, trade_pool

@st.cache_resource(ttl=6 * 3600, show_spinner="Scoring players for this league…")
def load_analytics(scoring_items: tuple, num_teams: int, roster_positions: tuple):
    df = load_weekly_stats()  # never empty here: the app stops earlier without stats
    df = df.assign(fp=compute_fantasy_points(df, dict(scoring_items)))
    index = StatsIndex(df)
    matchups = build_matchup_table(compute_defense_rankings(df))
    repl = compute_positional_scarcity(index, starter_counts(num_teams, list(roster_positions)))
    return index, matchups, repl

@st.cache_resource(ttl=3 * 3600, max_entries=6, show_spinner="Projecting the rest of the season…")
def load_context(key: tuple, _index, _matchups, _vegas, _players_db, _injury_map, _league):
    """
    Week-by-week valuation context for one league sync. `key` =
    (league_key, synced_at, scoring_key, season, week, playoff_week_start); the
    underscored arguments are derived from it and skipped by Streamlit's hashing.
    """
    _, _, scoring_items, season, week, playoff_start = key
    weeks = fantasy_weeks(week, playoff_start, settings=_league)
    # Only weeks before the current one are final. nflverse has stats for the current week as soon as
    # Thursday's game is played, so its last week would freeze this week's projections mid-week.
    proj = Projections(int(season), weeks, dict(scoring_items), completed_through=int(week) - 1) if weeks else None
    return ValuationContext(_index, proj, team_week_opponents(load_schedules()), _matchups, _injury_map, weeks,
                            _vegas, _players_db)

@st.cache_resource(ttl=3 * 3600, max_entries=6, show_spinner="Simulating the rest of the season…")
def load_season(key: tuple, _ctx, _league, _players_db):
    """Every team's weekly lineup projection, and the playoff-odds simulation, for one league sync."""
    positions = _league.get("roster_positions") or DEFAULT_ROSTER_POSITIONS
    dists = league_dists(_league, _ctx, _players_db, positions)
    return dists, simulate_season(_league, _ctx, _players_db, positions, dists=dists)

@st.cache_resource(ttl=3 * 3600, max_entries=12, show_spinner="Pricing waiver claims against your league…")
def load_waiver_engine(key: tuple, budget: int, _ctx, _roster, _waivers, _league, _players_db, _weights,
                       _rival_weights):
    """Waiver-pool analysis and this week's claim plan for one league sync and budget."""
    positions = _league.get("roster_positions") or DEFAULT_ROSTER_POSITIONS
    analysis = analyze_waiver_pool(_roster, _waivers, _ctx, _league, _players_db, budget, positions,
                                   weights=_weights, rival_weights=_rival_weights)
    analysis["plan"] = build_claim_plan(analysis, _roster, _ctx, budget, positions, weights=_weights)
    return analysis

@st.cache_resource(ttl=3 * 3600, max_entries=6, show_spinner="Searching every rival roster for trades…")
def load_trade_ideas(key: tuple, _roster, _league, _ctx, _players_db, _weights, _rival_weights):
    """1-for-1 trades that help both teams, for one league sync."""
    positions = _league.get("roster_positions") or DEFAULT_ROSTER_POSITIONS
    rivals = [r for r in _league.get("league_rosters") or [] if r["roster_id"] != _league.get("roster_id")]
    return find_trades(_roster, rivals, _ctx, _players_db, positions, _weights, _rival_weights)

@st.cache_data(ttl=3 * 3600, show_spinner=False)
def cached_weather(home_team: str, gameday: str):
    return get_weather_for_game(home_team, gameday)

@st.cache_data(ttl=3 * 3600, max_entries=256, show_spinner=False)
def sim_samples(player_key: str, name: str, stats_hash: str, center, _stats) -> np.ndarray:
    """Monte Carlo draws for one player, cached by player ID + stats + this week's center (projection)."""
    return simulate_player(_stats, name, center=center)

def stats_hash(s: dict) -> str:
    return repr((s.get("weekly"), (s.get("recent") or {}).get("ppg"), (s.get("season") or {}).get("ppg")))

def samples_for(p: dict, s: dict, center=None) -> np.ndarray:
    return sim_samples(pkey(p), p["name"], stats_hash(s), center, s)

def pkey(p: dict) -> str:
    """Stable per-player key: the Sleeper ID, or the name for players without one."""
    return p.get("player_id") or p["name"]


def refresh_all_data():
    fetch_weekly_stats(force=True)
    fetch_schedules(force=True)
    fetch_all_players(force=True)
    fetch_implied_totals()  # not forced: odds requests count against a monthly quota, so keep the 4-hour cache
    for path in DATA_DIR.glob("projections_*.json"):  # re-fetched on demand
        path.unlink(missing_ok=True)
    st.cache_data.clear()
    st.cache_resource.clear()


# ── Sidebar: settings, data, league ───────────────────────────────────────────

st.sidebar.header("⚙️ Settings")
# Only providers with an API key configured are offered
AI_PROVIDERS = available_providers()
if AI_PROVIDERS:
    if st.session_state.get("ai_provider") not in AI_PROVIDERS:
        st.session_state.ai_provider = AI_PROVIDERS[0]
    st.sidebar.radio("AI Provider", AI_PROVIDERS, key="ai_provider", format_func=PROVIDER_LABELS.get)
else:
    st.sidebar.caption("AI narratives are off: add `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY` to .env or "
                       "Streamlit Secrets.")

# Optional password for the AI buttons (AI_PASSWORD in .env or Streamlit secrets), so a shared
# copy of the app can hold an API key without strangers spending it. Unset = AI open, as locally.
# After MAX_PASSWORD_ATTEMPTS wrong tries the box locks for PASSWORD_LOCKOUT_S, doubling each time.
AI_PASSWORD = os.getenv("AI_PASSWORD", "")


def _check_ai_password():
    entered = st.session_state.get("ai_pw", "")
    st.session_state.ai_pw = ""
    if hmac.compare_digest(entered.encode(), AI_PASSWORD.encode()):
        st.session_state.ai_unlocked = True
        return
    fails = st.session_state.get("ai_pw_fails", 0) + 1
    st.session_state.ai_pw_fails = fails
    st.session_state.ai_pw_error = True
    if fails >= MAX_PASSWORD_ATTEMPTS:
        st.session_state.ai_locked_until = time.time() + PASSWORD_LOCKOUT_S * 2 ** (fails - MAX_PASSWORD_ATTEMPTS)


if AI_PASSWORD and AI_PROVIDERS and not st.session_state.get("ai_unlocked"):
    locked_for = st.session_state.get("ai_locked_until", 0) - time.time()
    if locked_for > 0:
        st.sidebar.error(f"Too many wrong passwords — try again in {math.ceil(locked_for)} s.")
    else:
        st.sidebar.text_input("🔒 AI password", type="password", key="ai_pw", on_change=_check_ai_password,
                              help="Unlocks the AI analysis buttons")
        if st.session_state.pop("ai_pw_error", False):
            st.sidebar.error("Wrong password.")

if st.sidebar.button("🔄 Refresh data", help="Re-download stats, schedule, Sleeper players/injuries, projections and odds"):
    with st.spinner("Refreshing all data sources…"):
        refresh_all_data()
    st.rerun()

st.sidebar.divider()
st.sidebar.header("📁 League")

CONFIG = load_config()
SYNC_MAX_AGE = CONFIG["auto_sync_minutes"] * 60


@st.cache_resource
def _sync_state():
    """Process-wide auto-sync bookkeeping: last attempt time and last error per league."""
    return {"attempts": {}, "errors": {}, "players_cleared": 0.0}


# Load the Sleeper player database first: the league syncs below (run in parallel) all need it,
# and this way it's downloaded once instead of once per league
_sync = _sync_state()
# Injury news moves fast on game days: drop the in-memory copy once the file cache is due for a
# refresh (throttled, so a Sleeper outage doesn't mean a download attempt on every rerun)
if players_db_due() and time.time() - _sync["players_cleared"] > PLAYERS_RETRY_S:
    _sync["players_cleared"] = time.time()
    load_players_db.clear()
try:
    _players_db, _name_index, _injury_map, _trade_pool = load_players_db()
except DataUnavailable as e:  # not cached: the next rerun tries again
    st.error(f"{e}. Check your connection and click **🔄 Refresh data** to retry.")
    _players_db, _name_index, _injury_map, _trade_pool = {}, {}, {}, []

# Keep the default leagues current: re-sync any whose saved data is older than auto_sync_minutes
if CONFIG["username"] and leagues_due(CONFIG["league_ids"], SYNC_MAX_AGE, _sync["attempts"]):
    with st.spinner("Updating your leagues from Sleeper…"):
        due = leagues_due(CONFIG["league_ids"], SYNC_MAX_AGE, _sync["attempts"])
        errors = auto_sync(CONFIG["username"], CONFIG["league_ids"], SYNC_MAX_AGE, _sync["attempts"])
    for lid in due:
        if lid in errors:
            _sync["errors"][lid] = errors[lid]
        else:
            _sync["errors"].pop(lid, None)

if flash := st.session_state.pop("flash", None):
    st.sidebar.success(flash)

_default_order = {lid: i for i, lid in enumerate(CONFIG["league_ids"])}
leagues = sorted(list_leagues(), key=lambda kv: (_default_order.get(kv[0], len(_default_order)), kv[1].lower()))
league_labels = dict(leagues)
if leagues:
    if st.session_state.get("league_key") not in league_labels:
        st.session_state.league_key = leagues[0][0]
    selected_league = st.sidebar.selectbox("Select League", list(league_labels), key="league_key",
                                           format_func=league_labels.get)
    current_data = load_league(selected_league)
else:
    selected_league = None
    current_data = {}
    st.sidebar.info("No leagues yet — add one below, or list your league IDs in config.toml.")

with st.sidebar.expander("➕ Add another Sleeper league", expanded=not leagues):
    sleeper_user = st.text_input("Sleeper Username", value=CONFIG["username"])
    target_league = st.text_input("League ID or Name",
                                  help="Sleeper issues a new league ID every season — the name always finds the current one. "
                                       "Add the ID to config.toml to keep it synced automatically.")
    if st.button("Sync"):
        with st.spinner("Syncing from Sleeper..."):
            data, err = sync_sleeper_league(sleeper_user.strip(), target_league.strip())
        if data:
            st.session_state.league_key = save_league(data)
            st.session_state.flash = f"Synced: {data['league_name']} ({data['season']})"
            st.rerun()
        else:
            st.error(err)


# ── Data loading ──────────────────────────────────────────────────────────────

try:
    _weekly_df = load_weekly_stats()
except DataUnavailable as e:  # not cached: the next rerun tries again
    st.error(f"{e}. Check your connection and click **🔄 Refresh data** to retry.")
    st.stop()
if _weekly_df.empty:
    st.error("Couldn't load NFL stats from nflverse and no cached copy exists. "
             "Check your connection and click **🔄 Refresh data**.")
    st.stop()

_schedules_df = load_schedules()
_next_games = load_next_games()
_vegas_totals = load_vegas()

scoring = effective_scoring(current_data.get("scoring_settings"))
roster_positions = current_data.get("roster_positions") or DEFAULT_ROSTER_POSITIONS
num_teams = int(current_data.get("total_rosters") or 12)
_index, _matchups, _repl_levels = load_analytics(scoring_key(scoring), num_teams, tuple(roster_positions))

roster = enrich_with_ids(current_data.get("roster", []), _players_db, _name_index)
waivers = enrich_with_ids(current_data.get("waivers", []), _players_db, _name_index)
playoff_weeks = playoff_weeks_for(current_data.get("playoff_week_start"), current_data)

_season_year = _index.season
_current_season = current_nfl_season()
status_bits = [f"Stats: {_season_year} regular season through week {_index.last_week}",
               f"Scoring: {scoring_label(scoring)}"]
if _schedules_df.empty:
    status_bits.append("Schedule unavailable")
db_age = players_db_age_seconds()
if db_age is not None:
    status_bits.append(fmt_as_of("Injuries", db_age))
odds = vegas_status()
if odds["fetched_at"] is not None:
    status_bits.append(fmt_as_of("Odds", time.time() - odds["fetched_at"])
                       + (f" ({odds['remaining']} API credits left)" if odds["remaining"] is not None else ""))
st.caption(" · ".join(status_bits))
if _season_year and _season_year < _current_season:
    st.info(f"No {_current_season} stats yet — values use the {_season_year} season. "
            f"Matchups use the {_current_season} schedule with {_season_year} defensive data.")
if not _players_db:
    st.warning("Sleeper player database unavailable — injury statuses and player lookups are disabled.")

# ── Sidebar: team, sync status, league settings, budget ─────────────────────

_NOTABLE_SCORING = [("rec", "Reception"), ("pass_td", "Pass TD"), ("pass_int", "Interception"),
                    ("fum_lost", "Fumble lost"), ("bonus_rec_te", "TE reception bonus"),
                    ("bonus_rec_rb", "RB reception bonus"), ("bonus_rec_wr", "WR reception bonus")]

if current_data:
    record = current_data.get("record")
    if record:
        rec = f"{record['wins']}-{record['losses']}" + (f"-{record['ties']}" if record.get("ties") else "")
        st.sidebar.markdown(f"**{current_data.get('team_name', '')}** · {rec} · {record['points_for']:g} PF")

    is_default = selected_league in _default_order
    age = sync_age_seconds(current_data)
    if age is not None:
        auto = f" · auto-updates every {CONFIG['auto_sync_minutes']:g} min" if is_default else ""
        st.sidebar.caption(f"Synced {fmt_age(age)}{auto}")
        if not is_default and age > STALE_SYNC_DAYS * 86400:
            st.sidebar.warning("Roster and waiver pool may be stale — sync to drop players already claimed.")
    elif roster:
        st.sidebar.warning("Sync time unknown (older file). Sync to refresh the roster and waiver pool.")
    if err := _sync["errors"].get(selected_league):
        st.sidebar.warning(f"Auto-sync failed: {err.rstrip('.')}. Showing the last saved data.")
    # A sync whose secondary requests failed carries those parts forward from the previous one
    missing_parts = sorted(set(current_data.get("partial") or [])
                           | ({"schedule"} if current_data.get("schedule_complete") is False else set()))
    if missing_parts:
        st.warning(f"The last Sleeper sync couldn't load: {', '.join(missing_parts)}. Those parts show the "
                   "previous sync's data"
                   + (f"; it retries automatically within {PARTIAL_RETRY_S // 60} minutes." if is_default
                      else " — click **🔄 Sync this league now** to retry."))

    if current_data.get("league_id") and st.sidebar.button("🔄 Sync this league now"):
        with st.spinner("Syncing from Sleeper..."):
            data, err = sync_sleeper_league(current_data.get("username") or CONFIG["username"],
                                            current_data["league_id"])
        if data:
            save_league(data)
            _sync["errors"].pop(selected_league, None)
            st.session_state.flash = f"Synced: {data['league_name']}"
            st.rerun()
        else:
            st.sidebar.error(err)

    league_season = current_data.get("season")
    if league_season and str(league_season).isdigit() and int(league_season) < _current_season:
        st.sidebar.warning(f"This is the {league_season} league. Sleeper creates a new league ID each season — "
                           "sync by league name to load the current one.")

    with st.sidebar.expander("📜 League Settings"):
        settings_rows = [f"**Teams:** {num_teams}",
                         f"**Lineup:** {lineup_summary(current_data.get('roster_positions') or DEFAULT_ROSTER_POSITIONS)}",
                         f"**Scoring:** {scoring_label(scoring)}"]
        settings_rows += [f"- {label}: {scoring[key]:+g}" for key, label in _NOTABLE_SCORING if scoring.get(key)]
        if league_uses_faab(current_data):
            total = current_data.get("waiver_budget_total")
            settings_rows.append("**Waivers:** FAAB" + (f", ${total} season budget" if total else ""))
        else:
            settings_rows.append("**Waivers:** priority order (no FAAB)")
        if record and record.get("waiver_position"):
            settings_rows.append(f"**Your waiver position:** {record['waiver_position']}")
        if current_data.get("playoff_week_start"):
            settings_rows.append(f"**Playoffs:** weeks {playoff_weeks[0]}–{playoff_weeks[-1]}")
        st.markdown("\n\n".join(settings_rows))

synced_budget = int(current_data.get("budget", 100))
faab_on = league_uses_faab(current_data)
my_waiver_pos = (current_data.get("record") or {}).get("waiver_position")
if faab_on:
    with st.sidebar.expander("💰 FAAB Budget", expanded=True):
        budget = st.number_input("Remaining Budget ($)", min_value=0, max_value=10000, value=synced_budget, step=1,
                                 key=f"budget_{selected_league}",
                                 help="Synced from Sleeper. Change it to test bids; it isn't saved.")
else:
    budget = synced_budget
    with st.sidebar.expander("🔢 Waiver priority", expanded=True):
        st.caption("This league doesn't use FAAB — claims go by waiver priority"
                   + (f"; you're #{my_waiver_pos}." if my_waiver_pos else "."))


# ── Per-player helpers ────────────────────────────────────────────────────────

def stats_for(p) -> dict:
    return _index.get_player_stats(p)

def inj_for(p) -> str | None:
    return injury_status(p, _injury_map)

def matchup_for(p) -> tuple[dict | None, dict]:
    next_g = _next_games.get(p.get("team", "")) if p.get("team") else None
    if not next_g:
        return None, dict(EMPTY_MATCHUP)
    return next_g, get_matchup_info(next_g["opponent"], p.get("position", ""), _matchups)

def stats_note(p, s) -> str:
    if s["found"]:
        return ""
    return "team D" if p.get("position") == "DEF" else "⚠️ not found"

def player_context(p, s=None) -> dict:
    """Facts handed to the AI so it doesn't rely on stale training data."""
    s = s or stats_for(p)
    next_g, m = matchup_for(p)
    usage = _index.get_usage_trends(p)
    ctx = {
        "Position": p.get("position"), "Team": p.get("team"),
        "Injury status": inj_for(p) or "None reported",
        "Stats found": "yes" if s["found"] else "NO — no stats this season",
        "Next game": (f"Week {next_g['week']} {'vs' if next_g['is_home'] else '@'} {next_g['opponent']} "
                      f"(defense rank {m['rank'] or '?'}, {m['rank_label']} vs {p.get('position')})") if next_g else "None scheduled",
        "Team implied total (Vegas)": _vegas_totals.get(p.get("team")),
    }
    for key, label in (("target_share", "Target share"), ("wopr", "WOPR")):
        if key in usage:
            ctx[f"{label} (recent / season)"] = f"{usage[key]['recent_avg']:.1%} / {usage[key]['season_avg']:.1%}"
    if "recent_carries_pg" in usage:
        ctx["Recent carries per game"] = usage["recent_carries_pg"]
    return ctx


roster_by_id = {pkey(p): p for p in roster}
roster_stats = {pkey(p): stats_for(p) for p in roster}
# Evaluator functions (drops, trades, roster grades) take stats keyed by name
stats_by_name = {p["name"]: roster_stats[pkey(p)] for p in roster}


# ── Rest-of-season engine (projections, lineup gains, FAAB market) ───────────
# Needs an in-season league synced from Sleeper (a current week); otherwise the
# tabs fall back to the history-only views.

_engine_key = (selected_league, current_data.get("synced_at"), scoring_key(scoring), current_data.get("season"),
               current_data.get("week"), current_data.get("playoff_week_start"))
ctx = None
if roster and current_data.get("week") and str(current_data.get("season", "")).isdigit():
    ctx = load_context(_engine_key, _index, _matchups, _vegas_totals, _players_db, _injury_map, current_data)
    if not ctx.weeks:
        ctx = None
# Playoff odds weight every rest-of-season value: a playoff week counts by the chance of being there
team_dists, season_sim = load_season(_engine_key, ctx, current_data, _players_db) if ctx else ({}, None)
my_p_playoffs = season_sim["p_playoffs"].get(current_data.get("roster_id")) if season_sim else None
my_p_bye = season_sim["p_bye"].get(current_data.get("roster_id")) if season_sim else None
playoff_start = current_data.get("playoff_week_start")
my_weights = week_weights(ctx.weeks, playoff_start, my_p_playoffs, current_data, my_p_bye) if ctx else None
rival_weights = all_week_weights(season_sim, ctx.weeks, playoff_start, current_data) if ctx else {}
# Rest-of-season gains count playoff weeks by the chance of being there (see season.week_weights)
WEIGHTED = " (playoff-weighted)" if my_p_playoffs is not None else ""
engine = load_waiver_engine(_engine_key, int(budget), ctx, roster, waivers, current_data, _players_db,
                            my_weights, rival_weights) if ctx and waivers else None
candidate_by_id = {pkey(c): c for c in (engine or {}).get("candidates", [])}
faab_on = engine["faab"] if engine else faab_on
week_now = ctx.current_week if ctx else None
_now = datetime.now(timezone.utc)
rival_rosters = [r for r in current_data.get("league_rosters") or [] if r["roster_id"] != current_data.get("roster_id")]
if ctx and not (ctx.projections and ctx.projections.available):
    st.warning("Sleeper projections are unavailable right now — values use past production only.")


# This week's outlook, shared by the Roster and Who to Start tabs: the head-to-head game and,
# in leagues that use one, the league-median game (a win for finishing in the top half)
def locked(p) -> bool:
    """His game this week has kicked off, so he can't be moved in or out of the lineup."""
    return bool(p.get("team")) and game_started(p["team"], _now, _schedules_df, week_now)


def live_projection(my_ids: list, opp_ids: list, my_points: dict, opp_points: dict) -> dict | None:
    """
    This week's projected final score for both lineups: points already scored by starters whose
    game has kicked off, plus the projection for those still to play. Win chance from a normal
    approximation over the players still to play (weekly scores vary by about SCORE_CV of the mean).
    """
    if not my_ids or not opp_ids:
        return None
    live = False

    def side(pids, points):
        nonlocal live
        total, var = 0.0, 0.0
        for pid in pids:
            p = sleeper_to_player(pid, _players_db.get(pid, {}))
            if locked(p):
                live = True
                total += points.get(pid) or 0.0
            else:
                v = ctx.week_values(p)[week_now]
                total += v
                var += (SCORE_CV * v) ** 2
        return total, var

    mine, var_m = side(my_ids, my_points)
    theirs, var_t = side(opp_ids, opp_points)
    sd = math.sqrt(var_m + var_t)
    p_win = 0.5 * (1 + math.erf((mine - theirs) / (sd * math.sqrt(2)))) if sd else float(mine > theirs) + 0.5 * (mine == theirs)
    return {"mine": round(mine, 1), "theirs": round(theirs, 1), "p_win": round(p_win, 3), "live": live}


mu = current_data.get("matchup") or {}
mu_proj = live_projection(mu.get("starters") or [], mu.get("opponent_starters") or [],
                          mu.get("players_points") or {}, mu.get("opponent_players_points") or {}) \
    if ctx and mu.get("opponent") else None
p_median = None
if ctx and current_data.get("median_game") and mu.get("starters"):
    p_median = median_win_prob(team_week_dist(mu["starters"], ctx, _players_db, week_now),
                               [d[week_now] for rid, d in team_dists.items() if rid != current_data.get("roster_id")],
                               seed=f"{selected_league}|{week_now}")
risk_p = combined_risk_prob(mu_proj["p_win"] if mu_proj else None, p_median)


def proj_now(p) -> float | None:
    """This week's projected points (bye/injury aware), or None without the engine."""
    return ctx.week_values(p)[week_now] if ctx else None



# ── UI Helpers ────────────────────────────────────────────────────────────────


def _score_gauge(score: float, label: str = "Value Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text": label},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "steelblue"},
               "steps": [{"range": [0, 40], "color": "firebrick"},
                         {"range": [40, 55], "color": "orange"},
                         {"range": [55, 72], "color": "yellowgreen"},
                         {"range": [72, 100], "color": "seagreen"}]},
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=0, l=20, r=20))
    return fig

_RAIN_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

def _raincloud_fig(names_samples: dict, title: str = "") -> go.Figure:
    """
    Half-violin + box + jittered points. 2 series → mirrored butterfly; 3-4 → grouped.
    names_samples: {label: np.ndarray of MC samples}
    """
    rng = np.random.default_rng(42)
    fig = go.Figure()
    mirrored = len(names_samples) == 2
    for i, (name, samples) in enumerate(names_samples.items()):
        samples = np.asarray(samples, dtype=float)
        pts = rng.choice(samples, size=min(300, len(samples)), replace=False)
        color = _RAIN_COLORS[i % len(_RAIN_COLORS)]
        kwargs = dict(y=pts, name=name, fillcolor=color, line_color=color, opacity=0.65,
                      box_visible=True, meanline_visible=True, points="all",
                      marker=dict(color=color, size=2.5, opacity=0.35), spanmode="hard")
        if mirrored:
            side = "negative" if i == 0 else "positive"
            kwargs.update(side=side, jitter=0.05, pointpos=1.6 if side == "negative" else -1.6)
        else:
            kwargs.update(x=[name] * len(pts), jitter=0.3, pointpos=-0.6)
        fig.add_trace(go.Violin(**kwargs))
    fig.update_layout(violinmode="overlay" if mirrored else "group", title=title,
                      yaxis_title="Simulated Fantasy Points", height=400,
                      margin=dict(t=50, b=20, l=20, r=20))
    return fig

def candidate_notes(c: dict) -> str:
    notes = []
    if c.get("promotion"):
        notes.append(f"⬆️ {c['promotion']}")
    if c.get("return_week"):
        notes.append(f"🏥 back ~wk {c['return_week']} (est.)")
    if c.get("handcuff_for"):
        notes.append(f"🔗 handcuff for {c['handcuff_for']}")
    if c.get("streamer"):
        notes.append("🔄 streamer: next 2 weeks only")
    if c.get("ros_proj") is not None and not c.get("p_win_known") and c["gain_total"] > 0:
        notes.append("rivals not modeled")
    return " · ".join(notes)


def _ai_block(state_key: str, request_id, button_label: str, run):
    """
    AI button whose output survives reruns. The answer is stored with its request, so the same
    request is never sent twice; each session gets at most MAX_AI_CALLS requests.
    """
    if not AI_PROVIDERS:
        st.caption("🔑 Add `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY` to .env or Streamlit Secrets to use this.")
        return
    if AI_PASSWORD and not st.session_state.get("ai_unlocked"):
        st.caption("🔒 Enter the AI password in the sidebar to use this.")
        return
    stored = st.session_state.get(state_key)
    if not (stored and stored[0] == request_id) and st.button(button_label, key=f"{state_key}_btn"):
        calls = st.session_state.get("ai_calls", 0)
        if calls >= MAX_AI_CALLS:
            st.warning(f"AI limit reached for this session ({MAX_AI_CALLS} requests).")
        else:
            st.session_state.ai_calls = calls + 1
            with st.spinner(f"Analyzing with {PROVIDER_LABELS[st.session_state.ai_provider]}…"):
                stored = st.session_state[state_key] = (request_id, run())
    if stored and stored[0] == request_id:
        st.markdown(stored[1])


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_roster, tab_waivers, tab_planner, tab_start_sit, tab_trade, tab_roster_mgmt = st.tabs(
    ["📋 Roster", "🔄 Waivers", "🧾 FAAB Planner", "⚖️ Who to Start", "🤝 Trade", "🏗️ Roster Construction"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ROSTER
# ════════════════════════════════════════════════════════════════════════════
_INJURY_COL = {"Injury": st.column_config.TextColumn(width=110)}  # fits "Questionable"
_SHORT_SLOT = {"SUPER_FLEX": "SFLEX", "WRRB_FLEX": "W/R", "REC_FLEX": "W/T", "IDP_FLEX": "IDP"}
_SLOT_ORDER = {slot: i for i, slot in enumerate(dict.fromkeys(
    [s for s in roster_positions if s not in NON_LINEUP_SLOTS] + ["BN", "IR", "TAXI"]))}

with tab_roster:
    st.subheader(current_data.get("team_name") or "Your Roster")
    if not roster:
        st.info("Sync a league from the sidebar.")
    else:
        lineup = lineup_suggestions(roster, ctx, roster_positions) if ctx else None
        # Players whose game has kicked off are locked in Sleeper: only suggest swaps that can still be made
        by_name = {p["name"]: p for p in roster}
        swaps = [s_ for s_ in (lineup or {}).get("swaps", [])
                 if not locked(by_name[s_["in"]]) and not (s_["out"] and locked(by_name[s_["out"]]))]
        skipped = len((lineup or {}).get("swaps", [])) - len(swaps)
        if mu and mu.get("bye"):
            st.caption(f"Week {mu['week']}: bye")
        elif mu and mu.get("opponent"):
            opp_rec = mu.get("opponent_record") or {}
            opp_rec_str = f" ({opp_rec.get('wins', 0)}-{opp_rec.get('losses', 0)})" if opp_rec else ""
            proj_mu = mu_proj
            mc1, mc2, mc3, mc4 = st.columns(4)
            if proj_mu:
                live = proj_mu["live"]
                how = "live + projected" if live else "projected"
                mc1.metric(f"Week {mu['week']} — You ({how})", f"{proj_mu['mine']:g}",
                           f"{mu['points']:g} scored so far" if live else None, delta_color="off")
                mc2.metric(f"{mu['opponent']}{opp_rec_str}", f"{proj_mu['theirs']:g}",
                           f"{mu['opponent_points']:g} scored so far" if live else None, delta_color="off")
                mc3.metric("Win chance (current lineup)", f"{proj_mu['p_win']:.0%}")
                if swaps:
                    outs, ins = {s_["out"] for s_ in swaps}, {s_["in"] for s_ in swaps}
                    fixed_ids = [p.get("player_id") for p in roster
                                 if (p.get("slot") and p.get("slot") not in NON_LINEUP_SLOTS and p["name"] not in outs)
                                 or p["name"] in ins]
                    # Always computable here: both starter lists are non-empty once fixes exist
                    fixed = live_projection(fixed_ids, mu.get("opponent_starters") or [],
                                            mu.get("players_points") or {}, mu.get("opponent_players_points") or {})
                    mc4.metric("Win chance with fixes below", f"{fixed['p_win']:.0%}",
                               f"{fixed['p_win'] - proj_mu['p_win']:+.0%}")
                if p_median is not None:
                    md1, md2, _, _ = st.columns(4)
                    md1.metric("Median game: top-half finish", f"{p_median:.0%}",
                               help="This league also scores every team against the league median each week.")
                    md2.metric("Expected wins this week", f"{proj_mu['p_win'] + p_median:.1f} of 2")
                st.caption(("Points already scored by starters whose game has kicked off, plus projections for "
                            "the rest, " if live else "Projections for ")
                           + "both starting lineups as set in Sleeper at the last sync. "
                           f"Suggested start/sit style this week: **{risk_mode(risk_p)}** "
                           + ("based on both games, head-to-head and the median " if p_median is not None else "")
                           + "(favorites should cut variance, underdogs should chase upside).")
            else:
                mc1.metric(f"Week {mu['week']} — You", f"{mu['points']:g}")
                mc2.metric(f"{mu['opponent']}{opp_rec_str}", f"{mu['opponent_points']:g}")
                mc3.caption("Live points from Sleeper as of the last sync.")

        if swaps:
            st.markdown(f"#### 🔁 Lineup fixes for week {week_now} "
                        f"(+{sum(s_['gain'] for s_ in swaps):.1f} projected pts)")
            st.dataframe(pd.DataFrame([{"Slot": s_["slot"], "Bench": s_["out"] or "(empty)", "Start": s_["in"],
                                        "+Pts": s_["gain"], "Why": s_["reason"]} for s_ in swaps]),
                         hide_index=True, width="stretch")
            st.caption("Make these in the Sleeper app — Sleeper's public API is read-only, so this app can't "
                       "change your lineup. Re-sync afterwards to clear this list.")
        else:
            has_slots = any(p.get("slot") for p in roster)
            flagged = [p["name"] for p in roster
                       if p.get("slot") not in (None, *NON_LINEUP_SLOTS) and is_inactive(inj_for(p))]
            if flagged:
                st.error(f"🚨 Starting an inactive player: {', '.join(flagged)} — swap them out in Sleeper.")
            n_starting = len([s_ for s_ in roster_positions if s_ not in NON_LINEUP_SLOTS])
            n_started = len([p for p in roster if p.get("slot") not in (None, *NON_LINEUP_SLOTS)])
            if has_slots and n_started < n_starting:
                st.warning(f"⚠️ {n_starting - n_started} empty starting slot(s) in your Sleeper lineup.")
            elif lineup and not skipped:
                st.success(f"✅ Your week {week_now} lineup is already the best by projection.")
        if skipped:
            st.caption(f"🔒 {skipped} better lineup move(s) left out: those players' games have already kicked off.")

        has_slots = any(p.get("slot") for p in roster)
        rows = []
        for p in sorted(roster, key=lambda p: _SLOT_ORDER.get(p.get("slot"), len(_SLOT_ORDER))):
            s = roster_stats[pkey(p)]
            next_g, m = matchup_for(p)
            rows.append({
                **({"Slot": _SHORT_SLOT.get(p.get("slot"), p.get("slot", ""))} if has_slots else {}),
                "Player": p["name"], "Pos": p.get("position", ""), "Team": p.get("team", ""),
                "Season PPG": s["season"]["ppg"] or 0, "Recent PPG": s["recent"]["ppg"] or 0,
                "Games": s["season"]["games"],
                **({f"Proj Wk {week_now}": proj_now(p), "ROS Proj": ctx.ros_total(p)} if ctx else {}),
                "Injury": inj_for(p) or "",
                # Drop the "(top 25%)" detail so the column doesn't get cut off
                "Next Matchup": f"vs {next_g['opponent']} ({m['rank_label'].split(' (')[0]})" if next_g else "—",
                "Stats": stats_note(p, s),
            })
        df_roster = pd.DataFrame(rows)
        st.dataframe(df_roster, hide_index=True, width="stretch", column_config=_INJURY_COL)
        if ctx:
            st.caption(f"Proj = 80% Sleeper projection + 20% this season's production, adjusted for byes, injuries "
                       f"(estimated return weeks) and matchup. ROS Proj = his projected points summed over weeks "
                       f"{ctx.weeks[0]}–{ctx.weeks[-1]}, every week counted fully.")

        league_rosters = current_data.get("league_rosters") or []
        if league_rosters:
            pp = playoff_position(league_rosters, current_data.get("roster_id"), current_data.get("playoff_teams"))
            title = "🏆 Standings"
            if pp:
                where = "in a playoff spot" if pp["in_position"] else "outside the playoffs"
                margin = pp["games_margin"]
                gap = (f", {abs(margin):g} game(s) {'ahead of' if margin >= 0 else 'behind'} the cutoff"
                       if margin is not None else "")
                title += f" — you're #{pp['rank']} of {pp['teams']}, {where}{gap}"
            if my_p_playoffs is not None:
                title += f" · {my_p_playoffs:.0%} playoff chances"
            with st.expander(title):
                st.dataframe(pd.DataFrame([{
                    "#": r["rank"], "Team": r["team_name"] + (" (you)" if r["roster_id"] == current_data.get("roster_id") else ""),
                    "W-L": f"{r['record']['wins']}-{r['record']['losses']}" + (f"-{r['record']['ties']}" if r['record'].get('ties') else ""),
                    "PF": r["record"]["points_for"], "FAAB left": r["faab_remaining"],
                    "Waiver #": r["record"].get("waiver_position"),
                    **({"Playoff chances": season_sim["p_playoffs"].get(r["roster_id"]),
                        "Projected wins": season_sim["exp_wins"].get(r["roster_id"])} if season_sim else {}),
                } for r in standings(league_rosters)]), hide_index=True, width="stretch",
                    column_config={"Playoff chances": st.column_config.ProgressColumn(
                        format="percent", min_value=0, max_value=1)})
                if pp:
                    st.caption(f"Top {pp['playoff_teams']} make the playoffs. Ranked by wins, then points for.")
                if season_sim:
                    st.caption(f"Playoff chances simulate the {len(season_sim['weeks'])} remaining regular-season "
                               "weeks of Sleeper's schedule from every team's projected best lineup"
                               + (", including the weekly median game" if current_data.get("median_game") else "")
                               + ". Waiver and trade values count each playoff week by your chance of being there.")

        missing = df_roster[df_roster["Stats"] == "⚠️ not found"]["Player"].tolist()
        if missing:
            st.caption(f"⚠️ No {_season_year} stats found for: {', '.join(missing)} "
                       "(rookie, didn't play, or an unmatched name). Their values show as 0.")

        fig = px.bar(df_roster.sort_values("Season PPG", ascending=False).head(12),
                     x="Player", y="Season PPG", color="Pos", title=f"Season PPG — {_season_year}")
        st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — WAIVERS
# ════════════════════════════════════════════════════════════════════════════
with tab_waivers:
    st.subheader("Waiver Wire")

    if not waivers:
        st.info("Sync a league to load the waiver pool.")
    else:
        if engine:
            worth = [c for c in engine["candidates"] if c["gain_total"] > 0]
            if faab_on:
                st.markdown(f"#### 🎯 Bid recommendations — ${budget} left, {engine['weeks_left']} weeks to go")
            else:
                st.markdown(f"#### 🎯 Waiver targets — priority #{my_waiver_pos or '?'}, "
                            f"{engine['weeks_left']} weeks to go")
            st.caption("Ranked by how much each player improves **your best starting lineup** from now through the "
                       "fantasy playoffs, after dropping your least valuable player. "
                       + ("Bids are priced from this league's past winning bids and which rivals would also want "
                          "the player. The **🧾 FAAB Planner** tab turns this into a set of claims that fit your budget."
                          if faab_on else
                          "No FAAB here: P(win) is the chance no interested rival ahead of you in waiver priority "
                          "claims him first.")
                       + (f" Playoff weeks count by your {my_p_playoffs:.0%} playoff chances (halved for each later "
                          "round; bye seeds skip round 1), so points sooner are worth more when the playoffs are "
                          "uncertain." if my_p_playoffs is not None else ""))
            if worth:
                st.dataframe(pd.DataFrame([{
                    "Player": c["name"], "Pos": c.get("position"), "Team": c.get("team"),
                    f"+Pts/wk{WEIGHTED}": c["gain_per_week"], f"+Pts ROS{WEIGHTED}": c["gain_total"],
                    "Drop": c["drop_name"] or "(open spot)",
                    **({"Worth to you": c["max_pay"], "Bid": c["suggested_bid"]} if faab_on else {}),
                    "P(win)": c["p_win"] if c["p_win_known"] else None,
                    "Rivals interested": c["interested_rivals"] if c["rivals"] else None,
                    "Notes": candidate_notes(c),
                } for c in worth]), hide_index=True, width="stretch",
                    column_config={"Worth to you": st.column_config.NumberColumn(format="$%d"),
                                   "Bid": st.column_config.NumberColumn(format="$%d"),
                                   "P(win)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
            else:
                st.info("No available player improves your starting lineup — "
                        + ("save your FAAB this week." if faab_on else "keep your waiver priority this week."))
            st.divider()
            st.markdown("#### Browse the pool")

        present = {w.get("position") for w in waivers}
        pos_opts = [p for p in ["QB", "RB", "WR", "TE", "K", "DEF"] if p in present]
        fc1, fc2 = st.columns([3, 1])
        sel_pos = fc1.multiselect("Filter by Position", pos_opts,
                                  default=[p for p in ["RB", "WR", "TE"] if p in pos_opts])
        hide_inactive = fc2.checkbox("Hide Out / IR / Doubtful", value=True)

        filtered = [w for w in waivers if not sel_pos or w.get("position") in sel_pos]
        if hide_inactive:
            filtered = [w for w in filtered if not is_inactive(inj_for(w))]
        enriched = _index.rank_available_players(filtered)

        sort_by = st.radio("Sort by", ["Trending", "Season PPG", "Recent PPG"], horizontal=True)
        sk = {"Trending": "trend_count", "Season PPG": "season_ppg", "Recent PPG": "recent_ppg"}[sort_by]
        enriched_sorted = sorted(enriched, key=lambda x: x.get(sk) or 0, reverse=True)

        table_rows = []
        for p in enriched_sorted:
            next_g, m = matchup_for(p)
            table_rows.append({
                "Name": p["name"], "Pos": p.get("position", ""), "Team": p.get("team", ""),
                "Season PPG": p["season_ppg"], "Recent PPG": p["recent_ppg"], "Games": p["games"],
                "Trending": p.get("trend_count", 0),
                "Next Matchup": f"vs {next_g['opponent']} ({m['rank_label']})" if next_g else None,
                "Injury": inj_for(p) or "",
                "Implied Total": _vegas_totals.get(p.get("team")),
            })
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch", column_config=_INJURY_COL)

        st.divider()
        st.subheader("Player Deep Dive")

        if not enriched_sorted:
            st.caption("No players match the current filters.")
        else:
            # Options are player IDs under a fixed key, so re-sorting or filtering keeps the same player selected
            pool_by_id = {pkey(p): p for p in enriched_sorted}
            sel_id = st.selectbox("Select player", list(pool_by_id), key=f"waiver_pick_{selected_league}",
                                  format_func=lambda k: player_label(pool_by_id[k]))
            p_data = pool_by_id[sel_id]
            selected_waiver = p_data["name"]
            pos = p_data.get("position", "")
            team = p_data.get("team", "")
            player_stats = stats_for(p_data)
            next_g, def_info = matchup_for(p_data)

            if not player_stats["found"]:
                st.warning(f"No {_season_year} stats found for {selected_waiver} — the score below "
                           "reflects no production, not a confirmed bad player.")

            ws = score_waiver(player_stats, pos, p_data.get("trend_count", 0),
                              defense_multiplier=def_info["multiplier"], replacement_levels=_repl_levels)
            cand = candidate_by_id.get(pkey(p_data))
            # The drop to compare against: the engine's pick (cheapest over the rest of the season) when
            # it ran, else the history-based recommendation
            if engine:
                drop_player = cand["drop"] if cand else None
                drop = None
            else:
                drop = recommend_drop(pos, roster, stats_by_name, _repl_levels)
                drop_player = next((p for p in roster if p["name"] == drop["name"]), None) if drop else None
            col_g, col_m = st.columns([1, 2])
            with col_g:
                st.plotly_chart(_score_gauge(ws["score"], "History score"), width="stretch")
                st.caption("Past production vs. a replacement-level player — not specific to your roster"
                           + (". The call on the right comes from your projected lineup gain." if engine else "."))
            with col_m:
                if engine:
                    gain_wk = cand["gain_per_week"] if cand else 0
                    v = "CLAIM" if gain_wk >= 1 else ("MARGINAL" if gain_wk > 0 else "PASS")
                    color = {"CLAIM": "green", "MARGINAL": "orange", "PASS": "red"}[v]
                    st.markdown(f"### :{color}[{v}] for your roster")
                else:
                    v = ws["verdict"]
                    st.markdown(f"### :{verdict_color(v)}[{v}]")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Expected Pts/Wk", ws["expected_pts"])
                m2.metric("Consistency", f"{ws['consistency_pct']:.0f}%")
                m3.metric("vs Replacement", f"{ws['surplus_vs_replacement']:+.1f}")
                if def_info["pts_allowed"] is not None:
                    m4.metric("Matchup Adj", f"×{def_info['multiplier']:.2f}",
                              f"{'above' if def_info['multiplier'] > 1 else 'below'} avg")

                if engine:
                    if cand and cand["gain_total"] > 0:
                        bid_amount = cand["suggested_bid"]
                        b1, b2, b3 = st.columns(3)
                        if faab_on:
                            b1.metric("Suggested bid", f"${bid_amount}",
                                      f"{cand['p_win']:.0%} to win" if cand["p_win_known"] else None,
                                      delta_color="off")
                            b2.metric("Worth to you (max)", f"${cand['max_pay']}",
                                      help="His rest-of-season lineup gain relative to a typical good add, "
                                           "priced so your budget lasts the season.")
                        else:
                            b1.metric("Claim odds", f"{cand['p_win']:.0%}" if cand["p_win_known"] else "—",
                                      help="Chance no interested rival ahead of you in waiver priority claims him.")
                            b2.metric("Your waiver priority", f"#{my_waiver_pos}" if my_waiver_pos else "—")
                        b3.metric("Lineup gain", f"+{cand['gain_per_week']:.1f} pts/wk",
                                  f"+{cand['gain_total']:.0f} rest of season{WEIGHTED}", delta_color="off",
                                  help="Per week = the weighted gain ÷ the weighted weeks you'd hold him "
                                       "(2 weeks for a DEF/K streamer).")
                        st.caption(cand["bid_note"])
                        if cand["drop_name"]:
                            st.markdown(f"**✂️ Drop:** {cand['drop_name']} — the roster spot that costs your "
                                        "lineup the least over the rest of the season")
                        if notes := candidate_notes(cand):
                            st.caption(notes)
                    else:
                        bid_amount = 0
                        st.metric("Suggested bid" if faab_on else "Claim", "$0" if faab_on else "Pass")
                        st.caption("He wouldn't improve your best starting lineup in any remaining week"
                                   + ("." if cand else " (not among the top projected players in the pool)."))
                else:
                    bid_amount = round(ws["faab_bid_pct"] * budget) if faab_on else 0
                    if faab_on:
                        st.metric("Recommended FAAB Bid", f"${bid_amount}",
                                  help=f"{ws['faab_bid_pct'] * 100:.0f}% of your ${budget} remaining budget "
                                       "(formula fallback — no current week to project).")
                    else:
                        st.caption("No FAAB in this league — claims go by waiver priority.")
                    if drop:
                        drop_color = "red" if drop["surplus"] < 0 else "orange"
                        st.markdown(f"**✂️ Drop:** :{drop_color}[{drop['name']}]"
                                    f" ({drop['position']} · {drop['adj_value']} adj pts) — {drop['reason']}")

            if engine and cand and cand["gain_total"] > 0:
                bc1, bc2 = st.columns(2)
                with bc1:
                    gain_df = pd.DataFrame({"Week": list(cand["weekly_gain"]),
                                            "Lineup gain": list(cand["weekly_gain"].values())})
                    fig = px.bar(gain_df, x="Week", y="Lineup gain",
                                 title="Your lineup gain by week (projected, unweighted)")
                    for w in playoff_weeks:
                        fig.add_vrect(x0=w - 0.5, x1=w + 0.5, fillcolor="gold", opacity=0.12, line_width=0)
                    fig.update_layout(height=320, margin=dict(t=50, b=20, l=20, r=20))
                    st.plotly_chart(fig, width="stretch")
                    st.caption("Shaded: fantasy playoff weeks.")
                with bc2:
                    if not faab_on:
                        st.caption(f"🔢 {cand['bid_note']}")
                    elif cand["curve"]:
                        curve_df = pd.DataFrame(cand["curve"], columns=["Bid ($)", "P(win)"])
                        fig = px.line(curve_df, x="Bid ($)", y="P(win)", title="Chance your claim wins, by bid")
                        fig.add_hline(y=TARGET_WIN_PROB, line_dash="dot", line_color="gray")
                        fig.add_vline(x=cand["suggested_bid"], line_color="seagreen",
                                      annotation_text=f"bid ${cand['suggested_bid']}")
                        fig.add_vline(x=cand["max_pay"], line_dash="dash", line_color="firebrick",
                                      annotation_text=f"max ${cand['max_pay']}", annotation_position="bottom right")
                        fig.update_layout(height=320, margin=dict(t=50, b=20, l=20, r=20), yaxis_tickformat=".0%")
                        st.plotly_chart(fig, width="stretch")
                    elif not cand["rivals"] and not rival_rosters:
                        st.caption("No rival rosters in this league's sync, so the claim is treated as uncontested.")
                    else:
                        st.caption("Rival bids weren't modeled for this player (outside the top candidates).")
                if cand["rivals"]:
                    with st.expander(f"👥 Who else wants him ({cand['interested_rivals']} of {len(cand['rivals'])} rivals)"):
                        st.dataframe(pd.DataFrame([{
                            "Team": r["team"], "Their +pts/wk": r["gain_per_week"], "Interested": "✅" if r["interested"] else "",
                            "FAAB left": r["budget"], "Waiver #": r["waiver_position"],
                            "Bidding style": f"×{r['aggression']:.2f}", "Est. bid": r["est_bid"] if r["interested"] else None,
                        } for r in cand["rivals"]]), hide_index=True, width="stretch")
                        st.caption("A rival is interested if the player adds ≥0.5 pts/wk to their best lineup. "
                                   "Bidding style compares their past winning bids with the league average "
                                   "(shrunk toward average when they've made few claims). Estimates, not certainties.")

            ctx1, ctx2, ctx3 = st.columns(3)
            with ctx1:
                st.markdown("**🛡️ Matchup**")
                if next_g:
                    hw = "🏠 Home" if next_g["is_home"] else "✈️ Away"
                    st.write(f"Week {next_g['week']} {hw} vs **{next_g['opponent']}**")
                    st.write(f"Rank: {def_info.get('rank') or '—'} | {def_info['rank_label']}")
                    st.write(f"Avg allowed vs {pos}: {def_info.get('pts_allowed') or '—'} pts/gm")
                    wx = cached_weather(next_g["home_team"], next_g["gameday"])
                    if wx and wx.get("dome"):
                        st.write("🏟️ Indoor — no weather")
                    elif wx:
                        st.write(f"🌡️ {wx.get('temp_f', '?')}°F | 💨 {wx.get('wind_mph', '?')} mph")
                        for flag in wx.get("flags", []):
                            st.warning(flag)
                    else:
                        st.caption("Forecast not available yet")
                else:
                    st.write("No upcoming game")

            with ctx2:
                st.markdown("**📈 Usage Trends**")
                usage = _index.get_usage_trends(p_data)
                if usage:
                    for metric, label in [("target_share", "Target Share"),
                                          ("air_yards_share", "Air Yards Share"), ("wopr", "WOPR")]:
                        if metric in usage:
                            u = usage[metric]
                            arrow = "↑" if u["trend"] > 0.01 else ("↓" if u["trend"] < -0.01 else "→")
                            st.write(f"{label}: {u['recent_avg']:.1%} {arrow} (season: {u['season_avg']:.1%})")
                    if "recent_carries_pg" in usage:
                        st.write(f"Carries/gm (recent): {usage['recent_carries_pg']}")
                else:
                    st.write("No usage data available")

            with ctx3:
                st.markdown("**🎰 Vegas & Injury**")
                vt = _vegas_totals.get(team)
                if vt is not None:
                    st.metric("Team Implied Total", vt)
                else:
                    st.write("Vegas: no line (offseason or no ODDS_API_KEY)")
                inj = inj_for(p_data)
                if inj:
                    st.error(f"🏥 {inj}")
                else:
                    st.success("✅ No injury designation")

            weekly_pts = player_stats.get("weekly", [])
            if weekly_pts:
                fig = px.line(x=player_stats["weeks"], y=weekly_pts, markers=True,
                              labels={"x": "Week", "y": "Fantasy Points"},
                              title=f"{selected_waiver} — Weekly Scoring")
                fig.add_hline(y=ws["expected_pts"], line_dash="dash", line_color="steelblue",
                              annotation_text="Expected")
                st.plotly_chart(fig, width="stretch")

            # A toggle rather than an expander: expander bodies run even while collapsed
            if st.toggle("🎲 Monte Carlo simulation", key="waiver_mc"):
                center = proj_now(p_data)  # this week's projection, 0 on bye / out
                waiver_samples = samples_for(p_data, player_stats, center)
                smry = _summary(waiver_samples)
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Floor (10th pct)", smry["floor"])
                mc2.metric("Median", smry["median"])
                mc3.metric("Ceiling (90th pct)", smry["ceiling"])
                mc4.metric("Std Dev", smry["std"])

                rc_map = {selected_waiver: waiver_samples}
                if drop_player:
                    rc_map[f"Drop: {drop_player['name']}"] = samples_for(
                        drop_player, roster_stats[pkey(drop_player)], proj_now(drop_player))
                st.plotly_chart(_raincloud_fig(rc_map, "Simulated outcomes this week (10,000 games)"),
                                width="stretch")
                if ctx:
                    st.caption(f"Centered on each player's week {week_now} projection, with the spread of his "
                               "own weekly scores.")

            with st.expander("🤖 AI Narrative Analysis (optional)"):
                st.caption("Needs `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY` in .env or Streamlit Secrets.")
                ai_ctx = player_context(p_data, player_stats)
                # With the engine, its CLAIM/PASS call is the verdict; the history score is context only
                ai_ctx.update({"History value score (not roster-specific)": ws["score"] if engine
                               else f"{ws['score']} ({ws['verdict']})",
                               "Model FAAB bid": f"${bid_amount}" if faab_on else "No FAAB (waiver priority)"})
                if cand:
                    ai_ctx.update({"Lineup gain for my roster": f"{cand['gain_per_week']} pts/wk, "
                                                                f"{cand['gain_total']} rest of season{WEIGHTED}",
                                   "Suggested drop": cand["drop_name"], "Max worth to me": f"${cand['max_pay']}",
                                   "Rival teams interested": cand["interested_rivals"],
                                   "Notes": candidate_notes(cand) or None})
                _ai_block("ai_waiver", (selected_league, pkey(p_data), int(budget), scoring_key(scoring), week_now),
                          "Get AI Analysis",
                          lambda: FantasyAgent(st.session_state.ai_provider).analyze_waiver(
                              selected_waiver, player_stats, roster, scoring, budget, ai_ctx))


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — FAAB PLANNER
# ════════════════════════════════════════════════════════════════════════════
with tab_planner:
    st.subheader("FAAB Planner")
    if not engine:
        st.info("The planner needs an in-season league synced from Sleeper (it projects from the current week).")
    else:
        plan = engine["plan"]
        profile = engine["profile"]
        day = waiver_day(current_data.get("waiver_settings"))
        spend = sum(c["suggested_bid"] for c in plan)
        pc1, pc2, pc3, pc4 = st.columns(4)
        if faab_on:
            pc1.metric("Budget left", f"${budget}")
            pc2.metric("This plan spends", f"${spend}", f"${budget - spend} left after", delta_color="off")
        else:
            pc1.metric("Waiver priority", f"#{my_waiver_pos}" if my_waiver_pos else "—")
            pc2.metric("Claims in this plan", len(plan))
        pc3.metric("Weeks to go", engine["weeks_left"], help=f"Weeks {ctx.weeks[0]}–{ctx.weeks[-1]}, through the "
                                                             "fantasy championship. FAAB is worth nothing after that.")
        pc4.metric("Waivers process", day or "—")
        if not faab_on:
            st.caption("This league uses waiver priority, not FAAB: submit the claims in the order below. "
                       "P(win) is the chance no interested rival ahead of you in the order claims him first.")

        st.markdown(f"#### Claims to submit for week {week_now}")
        if plan:
            st.dataframe(pd.DataFrame([{
                "Priority": c["priority"], "Add": c["name"], "Pos": c.get("position"),
                **({"Bid": c["suggested_bid"]} if faab_on else {}),
                "P(win)": c["p_win"] if c["p_win_known"] else None, "Drop": c["drop_name"] or "(open spot)",
                f"+Pts ROS{WEIGHTED}": c["gain_total"],
                "Alone": c["standalone_gain"] if c["standalone_gain"] != c["gain_total"] else None,
                "Expected +pts": c["expected_gain"], "Notes": candidate_notes(c),
            } for c in plan]), hide_index=True, width="stretch",
                column_config={"Bid": st.column_config.NumberColumn(format="$%d"),
                               "P(win)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                               "Alone": st.column_config.NumberColumn(
                                   help="Gain if this were your only claim. It's lower in the plan because the "
                                        "claims above it are assumed to go through.")})
            st.caption("Each claim is valued assuming the ones above it succeed, so the plan won't stack two "
                       "players for one lineup spot or drop someone you just added. Later claims are repriced at "
                       "that smaller value. Expected +pts = P(win) × lineup gain."
                       + (f" Bids aim for a {TARGET_WIN_PROB:.0%} chance to win without paying more than the player "
                          "is worth to you, re-bid from the budget the claims above leave." if faab_on else ""))
        else:
            st.info("No claim improves your lineup enough to be worth a bid this week.")

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("#### 📈 This league's FAAB market")
            if not faab_on:
                st.caption("No FAAB in this league — waiver claims go by priority order.")
            elif profile["claims_won"]:
                bt = profile["budget_total"]
                pct = lambda k: f"${profile[k] * bt:.0f} ({profile[k]:.0%})" if profile.get(k) is not None else "—"
                st.markdown("\n".join([
                    f"- **{profile['claims_won']}** winning claims, **{profile['claims_outbid']}** outbid",
                    f"- Winning bids, typical add: {pct('win_pct_p50')} · strong add (75th pct): "
                    f"{pct('win_pct_p75')} · top add (90th pct): {pct('win_pct_p90')}",
                    f"- {profile['zero_dollar_share']:.0%} of claims won for $0" if profile["zero_dollar_share"]
                    is not None else "",
                    (f"- {profile['contested']} contested claims; winners beat second place by a median "
                     f"${profile['median_overbid']:.0f}") if profile.get("median_overbid") is not None else "",
                ]))
            else:
                st.caption("No winning FAAB claims yet this season — bids use the budget formula instead.")
        with mc2:
            st.markdown("#### 👥 Rival budgets")
            mine_id = current_data.get("roster_id")
            st.dataframe(pd.DataFrame([{
                "Team": r["team_name"] + (" (you)" if r["roster_id"] == mine_id else ""),
                "FAAB left": r["faab_remaining"], "Waiver #": r["record"].get("waiver_position"),
                "Claims won": profile["by_roster"].get(r["roster_id"], {}).get("wins", 0),
                "Bidding style": profile["by_roster"].get(r["roster_id"], {}).get("aggression"),
            } for r in sorted(current_data.get("league_rosters") or [], key=lambda r: -r["faab_remaining"])]),
                hide_index=True, width="stretch",
                column_config={"FAAB left": st.column_config.NumberColumn(format="$%d"),
                               "Bidding style": st.column_config.NumberColumn(
                                   format="×%.2f", help="Average winning bid vs. the league average")})
            richer = sum(1 for r in current_data.get("league_rosters") or []
                         if r["roster_id"] != mine_id and r["faab_remaining"] > budget)
            if richer and faab_on:
                st.caption(f"{richer} rival(s) have more FAAB than you"
                           + (" — target players few teams need rather than bidding wars." if richer >= 3 else "."))

        st.markdown("#### 🧾 Your FAAB log")
        log_rows = faab_log(current_data.get("transactions") or [], current_data.get("roster_id"), _index, _players_db)
        if log_rows:
            spent = sum(r["Cost"] for r in log_rows)
            st.dataframe(pd.DataFrame(log_rows), hide_index=True, width="stretch",
                         column_config={"Cost": st.column_config.NumberColumn(format="$%d")})
            st.caption(f"${spent} spent on {len(log_rows)} adds this season. Points since added count every "
                       "game after the add, including bench weeks. Team defenses have no stats source, so they show blank.")
        else:
            st.caption("No completed adds yet this season.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — WHO TO START
# ════════════════════════════════════════════════════════════════════════════
with tab_start_sit:
    st.subheader("Who to Start")

    if not roster:
        st.info("Sync a league to compare roster players.")
    else:
        default_style = risk_mode(risk_p)
        sc1, sc2 = st.columns([3, 2])
        selected_ids = sc1.multiselect("Select 2–4 players to compare", list(roster_by_id), max_selections=4,
                                       format_func=lambda k: roster_by_id[k]["name"])
        style = sc2.radio("Start/sit style", ["Safe", "Balanced", "Upside"], horizontal=True,
                          index=["Safe", "Balanced", "Upside"].index(default_style),
                          key=f"ss_style_{selected_league}",
                          help="Safe ranks by floor (10th percentile), Balanced by projection, Upside by ceiling "
                               "(90th percentile).")
        sc2.caption(f"Suggested this week: **{default_style}**"
                    + (f" — you have a {risk_p:.0%} chance to win"
                       + (" (average of the head-to-head and median games)." if p_median is not None else ".")
                       if risk_p is not None else " (no matchup projection)."))
        if st.button("⚖️ Compare"):
            st.session_state.ss_request = tuple(selected_ids)

        if len(selected_ids) < 2:
            st.caption("Select at least 2 players.")
        elif st.session_state.get("ss_request") != tuple(selected_ids):
            st.caption("Click **Compare** to run the analysis.")
        else:
            # Keyed by display name from here on (charts and the AI prompt use names)
            selected_players = [roster_by_id[k]["name"] for k in selected_ids]
            sel_by_name = {roster_by_id[k]["name"]: roster_by_id[k] for k in selected_ids}
            named_stats = {n: roster_stats[pkey(p_obj)] for n, p_obj in sel_by_name.items()}
            matchups, injuries, projected, samples = {}, {}, {}, {}
            for n, p_obj in sel_by_name.items():
                next_g, m = matchup_for(p_obj)
                matchups[n] = (next_g, m)
                injuries[n] = inj_for(p_obj)
                if ctx:
                    # Shape from his own weekly scores, centered on this week's projection
                    projected[n] = proj_now(p_obj)
                    samples[n] = samples_for(p_obj, named_stats[n], projected[n])
                else:
                    mult = m["multiplier"] * injury_factor(injuries[n])
                    samples[n] = samples_for(p_obj, named_stats[n]) * mult
                    projected[n] = round(float(np.mean(samples[n])), 2)

            summaries = {n: _summary(samples[n]) for n in selected_players}
            metric_key = {"Safe": "floor", "Balanced": "mean", "Upside": "ceiling"}[style]
            score = {n: (projected[n] if metric_key == "mean" else summaries[n][metric_key]) for n in selected_players}
            order = sorted(selected_players, key=lambda n: -score[n])
            shares = win_shares([samples[n] for n in selected_players])
            win_probs = dict(zip(selected_players, shares))
            ranked = [{"name": n, "rank": i + 1, "start_value": projected[n], "injury": injuries[n],
                       "consistency": player_value(named_stats[n])["consistency"]} for i, n in enumerate(order)]
            mc_result = {"win_probs": win_probs, "summaries": summaries, "samples": {n: samples[n] for n in order}}

            top = ranked[0]
            edge = score[order[0]] - score[order[1]]
            conf = edge / max(score[order[0]], 0.1)
            conf_label = "Clear" if conf >= 0.25 else ("Lean" if conf >= 0.10 else "Toss-up")
            banner_color = "green" if conf_label == "Clear" else ("orange" if conf_label == "Lean" else "gray")
            metric_name = {"floor": "floor", "mean": "projection", "ceiling": "ceiling"}[metric_key]
            st.markdown(f"## :{banner_color}[START: {top['name']}]  "
                        f"— {conf_label} · {style}: {metric_name} edge {edge:.1f} pts · "
                        f"P(outscores the others) {round(win_probs[top['name']] * 100)}%")
            if ctx:
                st.caption(f"Week {week_now} projection: 80% Sleeper projection + 20% this season's production, "
                           "adjusted for matchup, bye and injury. The spread comes from each player's own weekly "
                           "scores this season.")
            else:
                st.caption("Projection = adjusted value × opponent defense multiplier × injury factor "
                           "(Questionable ×0.85, Doubtful ×0.25, Out/IR ×0).")

            missing = [n for n in selected_players if not named_stats[n]["found"]]
            if missing:
                st.warning(f"No {_season_year} stats for: {', '.join(missing)} — the outcome spread is a guess.")

            cols = st.columns(len(ranked))
            for col, r in zip(cols, ranked):
                name = r["name"]
                is_start = r["rank"] == 1
                mc_smry = summaries[name]
                next_g, m = matchups[name]
                with col:
                    st.markdown(f"**{name}**")
                    start_lbl = "✅ START" if is_start else f"#{r['rank']} BENCH"
                    st.markdown(f":{'green' if is_start else 'red'}[{start_lbl}]")
                    st.metric("Projected", f"{r['start_value']:.1f}")
                    st.metric("Floor / Ceiling", f"{mc_smry['floor']} / {mc_smry['ceiling']}")
                    st.metric("P(top scorer)", f"{round(win_probs[name] * 100)}%")
                    st.metric("Consistency", f"{r['consistency'] * 100:.0f}%")
                    if next_g:
                        st.caption(f"vs {next_g['opponent']} — {m['rank_label']} (×{m['multiplier']:.2f})")
                    else:
                        st.caption("No upcoming game")
                    if r["injury"]:
                        st.error(f"🏥 {r['injury']}")

            rows = []
            for name, stats in named_stats.items():
                rows += [
                    {"Player": name, "Metric": "Season PPG", "Value": stats["season"]["ppg"]},
                    {"Player": name, "Metric": "Recent PPG", "Value": stats["recent"]["ppg"] or 0},
                    {"Player": name, "Metric": "Projected", "Value": projected[name]},
                ]
            st.plotly_chart(px.bar(pd.DataFrame(rows), x="Metric", y="Value", color="Player", barmode="group",
                                   title="Head-to-Head Comparison",
                                   category_orders={"Metric": ["Season PPG", "Recent PPG", "Projected"]}),
                            width="stretch")

            st.plotly_chart(_raincloud_fig(mc_result["samples"], "Simulated Outcome Distributions (10,000 games)"),
                            width="stretch")

            if len(selected_players) == 2:
                p1, p2 = selected_players
                p1_wins = float(np.mean(samples[p1] > samples[p2]) + 0.5 * np.mean(samples[p1] == samples[p2]))
                hc1, hc2 = st.columns(2)
                hc1.metric(f"P({p1} outscores)", f"{p1_wins * 100:.0f}%")
                hc2.metric(f"P({p2} outscores)", f"{(1 - p1_wins) * 100:.0f}%")

            with st.expander("Weekly scoring history"):
                fig2 = go.Figure()
                for name, stats in named_stats.items():
                    if stats["weekly"]:
                        fig2.add_trace(go.Scatter(x=stats["weeks"], y=stats["weekly"], mode="lines+markers", name=name))
                fig2.update_layout(xaxis_title="Week", yaxis_title="Fantasy Points", title="Season Weekly Scores")
                st.plotly_chart(fig2, width="stretch")

            with st.expander("🤖 AI Narrative Analysis (optional — 2 players only)"):
                if len(selected_players) == 2:
                    contexts = {n: player_context(sel_by_name[n], named_stats[n]) for n in (p1, p2)}
                    # The app's own verdict, so the narrative explains it rather than re-deriving from season PPG
                    model_summary = {
                        "Projection (this week)": f"{p1} {projected[p1]:.1f} vs {p2} {projected[p2]:.1f}",
                        "Floor / ceiling": f"{p1} {summaries[p1]['floor']} / {summaries[p1]['ceiling']} vs "
                                           f"{p2} {summaries[p2]['floor']} / {summaries[p2]['ceiling']}",
                        "P(P1 outscores P2)": f"{p1_wins:.0%}",
                        "Style": style,
                        "App's pick": top["name"],
                    }
                    _ai_block("ai_ss", (selected_league, *selected_ids, style, week_now), "Get AI Analysis",
                              lambda: FantasyAgent(st.session_state.ai_provider).analyze_start_sit(
                                  p1, p2, named_stats[p1], named_stats[p2], scoring, contexts,
                                  model_summary=model_summary))
                else:
                    st.caption("AI narrative is only available for 2-player comparisons.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRADE ANALYZER
# ════════════════════════════════════════════════════════════════════════════
with tab_trade:
    st.subheader("Trade Analyzer")
    if not roster:
        st.info("Sync a league to judge trades by your starting lineup. Without a roster, trades compare "
                "value over replacement only.")

    deadline = current_data.get("trade_deadline")
    # Sleeper stores "no deadline" as a week past the regular season (e.g. 99)
    no_deadline = bool(deadline) and int(deadline) > LAST_NFL_WEEK
    to_deadline = None if no_deadline else weeks_until_deadline(deadline, week_now)
    if no_deadline:
        st.info("This league has no trade deadline.")
    elif to_deadline is not None and to_deadline < 0:
        st.warning(f"The trade deadline (week {deadline}) has passed — trades are closed for this season.")
    elif to_deadline == 0:
        st.warning(f"This is the trade deadline week (week {deadline}).")
    elif to_deadline is not None:
        st.info(f"Trade deadline: week {deadline}, {to_deadline} week(s) away.")

    if ctx and rival_rosters and (to_deadline is None or to_deadline >= 0):
        with st.expander("🔎 Trade finder: 1-for-1 trades that help both teams"):
            st.caption("Tries swapping each of your top 12 players for each rival's top 12, and keeps trades that "
                       "raise both teams' best starting lineups for the rest of the season (each team's playoff "
                       "weeks weighted by its own playoff chances). Trades that help both sides get accepted; "
                       "the list is sorted by the smaller of the two gains.")
            if st.button("Find trades"):
                st.session_state.trade_finder = _engine_key
            if st.session_state.get("trade_finder") == _engine_key:
                ideas = load_trade_ideas(_engine_key, roster, current_data, ctx, _players_db, my_weights, rival_weights)
                if ideas:
                    st.dataframe(pd.DataFrame([{
                        "Trade with": t["team"], "You give": f"{t['give']} ({t['give_pos']})",
                        "You get": f"{t['get']} ({t['get_pos']})", f"Your gain (ROS pts{WEIGHTED})": t["gain_me"],
                        f"Their gain (ROS pts{WEIGHTED})": t["gain_them"]} for t in ideas]),
                        hide_index=True, width="stretch")
                else:
                    st.info("No 1-for-1 trade improves both lineups right now.")

    roster_ids = {p.get("player_id") for p in roster if p.get("player_id")}
    trade_pool = {p["player_id"]: p for p in _trade_pool if p["player_id"] not in roster_ids}

    give_ids = st.multiselect("You Give (from your roster)", list(roster_by_id),
                              format_func=lambda k: player_label(roster_by_id[k]))
    recv_ids = st.multiselect("You Receive", list(trade_pool),
                              format_func=lambda pid: player_label(trade_pool[pid]),
                              placeholder="Search any NFL player…")
    give_players = [roster_by_id[k] for k in give_ids]
    receiving_players = [trade_pool[pid] for pid in recv_ids]
    giving = [p["name"] for p in give_players]
    receiving = [p["name"] for p in receiving_players]

    request = (selected_league, tuple(give_ids), tuple(recv_ids))
    if st.button("🤝 Analyze Trade", disabled=not (giving or receiving)):
        st.session_state.trade_request = request

    if (giving or receiving) and st.session_state.get("trade_request") == request:
        stats_map = {p["name"]: roster_stats.get(pkey(p)) or stats_for(p) for p in give_players + receiving_players}
        all_stats = {**stats_by_name, **stats_map}

        missing = [n for n, s in stats_map.items() if not s["found"]]
        if missing:
            st.warning(f"No {_season_year} stats for: {', '.join(missing)} — valued at 0.")

        trade = score_trade(roster, giving, receiving_players, all_stats, _repl_levels, roster_positions, ctx=ctx)
        # Simulations use this week's projection for each player (0 on bye / out) when there is one
        centers = {p["name"]: proj_now(p) for p in roster + receiving_players} if ctx else None
        if trade["method"] == "lineup":
            mc = simulate_lineup_change([s["name"] for s in trade["before"]["starters"]],
                                        [s["name"] for s in trade["after"]["starters"]], all_stats, centers)
            p_better = mc["p_better"]
            rc = {"Lineup Before": mc["before_samples"], "Lineup After": mc["after_samples"]}
            rc_title = "Weekly Starting-Lineup Total (10,000 simulations)"
        else:
            mc = simulate_trade({n: stats_map[n] for n in giving}, {n: stats_map[n] for n in receiving}, centers)
            p_better = mc["p_receive_better"]
            rc = {"You Give": mc["give_samples"], "You Receive": mc["receive_samples"]}
            rc_title = "Total Weekly Output (10,000 simulations)"

        history_line = (f"Starting lineup {trade['history_delta']:+.1f} adj pts/wk "
                        f"({trade['history_delta_pct']:+.1%}) by past production"
                        if trade["method"] == "lineup" else
                        f"Value over replacement {trade['history_delta']:+.1f} pts/wk")
        if trade["basis"] == "projection":
            headline = (f"Rest of season {trade['ros_delta']:+.1f} lineup pts ({trade['delta_pct']:+.1%}), "
                        f"{trade['ros_delta_per_week']:+.1f}/wk · P(better week) = {round(p_better * 100)}%")
        elif trade["method"] == "lineup":
            headline = f"{history_line} · P(better week) = {round(p_better * 100)}%"
        else:
            headline = f"{history_line} · P(receive side wins) = {round(p_better * 100)}%"

        verdict = trade["verdict"]
        st.markdown(f"## :{verdict_color(verdict)}[{verdict}]  — {headline}")
        if trade["basis"] == "projection":
            ros_weighted = sum((trade["ros_after"][w] - trade["ros_before"][w]) * my_weights.get(w, 1.0)
                               for w in ctx.weeks)
            st.metric(f"Rest of season{WEIGHTED}", f"{ros_weighted:+.1f} lineup pts",
                      help=f"Your best lineup's projected points over weeks {ctx.weeks[0]}–{ctx.weeks[-1]}, "
                           "after vs. before, with playoff weeks counted by your playoff chances.")
            st.caption(f"Verdict from your best starting lineup's weekly projections, weeks {ctx.weeks[0]}–"
                       f"{ctx.weeks[-1]}. History view: **{trade['history_verdict']}** — {history_line}.")
        else:
            st.caption("Verdict is based on your best possible starting lineup before vs. after the trade, so "
                       "bench depth only counts if it would start. Value over replacement (VORP) is shown per player.")
        if trade["dropped"]:
            st.warning(f"✂️ To stay within your {trade['roster_limit']}-player roster you'd have to cut: "
                       f"{', '.join(trade['dropped'])} (counted in the after lineup).")
        if trade["excluded"]:
            st.caption(f"Not counted in either lineup (IR / TAXI): {', '.join(trade['excluded'])}.")

        vc1, vc2 = st.columns(2)
        for col, players, side, header, total in [
                (vc1, give_players, trade["you_give"], "You Give", trade["give_vorp"]),
                (vc2, receiving_players, trade["you_receive"], "You Receive", trade["recv_vorp"])]:
            with col:
                st.markdown(f"**{header}**")
                for p in players:
                    val = side[p["name"]]
                    inj = inj_for(p)
                    st.metric(f"{p['name']}{f' 🏥 {inj}' if inj else ''}", f"{val['adj_value']} adj pts/wk",
                              f"VORP {val['surplus']:+.1f}", delta_color="off")
                st.metric("Total VORP", f"{total} pts/wk")

        if trade["method"] == "lineup":
            with st.expander("Starting lineup before vs. after (past production)"):
                lc1, lc2 = st.columns(2)
                for col, key, title in [(lc1, "before", "Before"), (lc2, "after", "After")]:
                    lineup = trade[key]
                    col.markdown(f"**{title}** — {lineup['total']} adj pts/wk")
                    col.dataframe(pd.DataFrame([{"Slot": s["slot"], "Player": s["name"] or "(empty)",
                                                 "Value": s["value"]} for s in lineup["starters"]]),
                                  hide_index=True, width="stretch")

        bar_rows = ([{"Side": "Give", "Player": n, "Adj Value": v["adj_value"]} for n, v in trade["you_give"].items()]
                    + [{"Side": "Receive", "Player": n, "Adj Value": v["adj_value"]} for n, v in trade["you_receive"].items()])
        # Never empty: the Analyze button is disabled until a player is picked on either side
        st.plotly_chart(px.bar(pd.DataFrame(bar_rows), x="Player", y="Adj Value", color="Side",
                               color_discrete_map={"Give": "firebrick", "Receive": "seagreen"},
                               title="Trade Value Comparison"), width="stretch")

        st.plotly_chart(_raincloud_fig(rc, rc_title), width="stretch")

        with st.expander(f"📅 Playoff Schedule Strength (Weeks {playoff_weeks[0]}–{playoff_weeks[-1]})"):
            psc1, psc2 = st.columns(2)
            for col, players, header in [(psc1, give_players, "You Give"), (psc2, receiving_players, "You Receive")]:
                with col:
                    st.markdown(f"**{header}**")
                    for p in players:
                        pss = get_playoff_schedule_strength(p.get("team", ""), _schedules_df, _matchups,
                                                            p.get("position", ""), playoff_weeks)
                        st.markdown(f"{p['name']}: :{grade_color(pss['grade'])}[Grade {pss['grade']}] "
                                    f"(avg mult {pss['avg_multiplier']:.2f})")
                        for wk in pss["weeks"]:
                            st.caption(f"  Wk {wk['week']}: vs {wk['opponent']} — {wk['matchup']}")

        with st.expander("🤖 AI Narrative Analysis (optional)"):
            contexts = {p["name"]: player_context(p, stats_map[p["name"]]) for p in give_players + receiving_players}
            summary = {"Verdict": verdict, "Lineup / value change": headline,
                       "History-based view": f"{trade['history_verdict']}: {history_line}",
                       "Players cut to fit the roster": ", ".join(trade["dropped"]) or None,
                       "VORP given": trade["give_vorp"], "VORP received": trade["recv_vorp"]}
            _ai_block("ai_trade", request, "Get AI Verdict",
                      lambda: FantasyAgent(st.session_state.ai_provider).analyze_trade(
                          giving, receiving, stats_map, scoring, contexts, summary))


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — ROSTER CONSTRUCTION
# ════════════════════════════════════════════════════════════════════════════
with tab_roster_mgmt:
    st.subheader("Roster Construction Analysis")

    if not roster:
        st.info("Sync a league first.")
    elif not waivers:
        st.info("Sync a league to load the waiver pool for recommendations.")
    else:
        avail_pool = [w for w in waivers
                      if w.get("position") in {"QB", "RB", "WR", "TE"} and not is_inactive(inj_for(w))]
        avail_stats = {p["name"]: stats_for(p) for p in avail_pool}
        avail_mults = {p["name"]: matchup_for(p)[1]["multiplier"] for p in avail_pool}
        analysis = analyze_roster_construction(roster, avail_pool, stats_by_name, avail_stats,
                                               _repl_levels, avail_mults, roster_positions)

        pg = analysis["position_grades"]
        weaknesses = analysis["weaknesses"]

        st.markdown("### Positional Strength")
        if pg:
            graded = list(pg.items())
            for i in range(0, len(graded), 3):  # three per row, so narrow screens stay readable
                for col, (pos, g) in zip(st.columns(3), graded[i:i + 3]):
                    col.metric(pos, f"Grade {g['grade']}", f"×{g['strength_ratio']:.2f} vs replacement")
                    col.caption(f"{g['depth_count']} players · starters avg {g['avg_starter_val']:.1f} adj pts")
            st.caption("Graded positions follow this league's lineup slots. DEF isn't graded — "
                       "team defense stats aren't in the nflverse data.")
        else:
            st.caption("This league's lineup has no position-specific slots to grade.")

        if weaknesses:
            st.warning(f"⚠️ Positional weaknesses: **{', '.join(weaknesses)}** — prioritized in recommendations below")
        else:
            st.success("✅ No critical positional weaknesses")

        st.divider()
        need_pct = {pos: {"F": 0.20, "D": 0.12}.get(pg[pos]["grade"], 0.0) for pos in weaknesses}
        if engine:
            # Same lineup-gain math as the Waivers tab, boosted where the roster is weak
            st.markdown("### Need-Adjusted Waiver Targets")
            st.caption(f"Score = projected lineup gain per week{WEIGHTED} from the Waivers tab, +20% at a position "
                       "graded F and +12% at D.")
            target_rows = [{
                "Player": c["name"], "Pos": c.get("position", ""), "Team": c.get("team", ""),
                "Need Score": round(c["gain_per_week"] * (1 + need_pct.get(c.get("position"), 0.0)), 2),
                "Base Score": c["gain_per_week"],
                "Fills Need": "⭐ YES" if c.get("position") in weaknesses else "—",
                "Drop": c["drop_name"] or "(open spot)",
                "Injury": inj_for(c) or "",
            } for c in engine["candidates"] if c["gain_total"] > 0 and c.get("position") in pg]
            y_label = "+Pts/wk"
        else:
            st.markdown("### Need-Adjusted Waiver Targets (history-only)")
            st.caption("No current week to project, so this uses past production: score = generic waiver score "
                       "(incl. next-game matchup) + need bonus (up to +20 for filling a weakness)")
            target_rows = []
            for t in analysis["targeted_adds"]:
                next_g, _ = matchup_for(t)
                target_rows.append({
                    "Player": t["name"], "Pos": t.get("position", ""), "Team": t.get("team", ""),
                    "Need Score": t["need_adjusted_score"], "Base Score": t["base_score"],
                    "Fills Need": "⭐ YES" if t["fills_need"] else "—",
                    "Verdict": t["verdict"], "Adj Value": t["adj_value"],
                    "Next Game": f"vs {next_g['opponent']}" if next_g else None,
                    "Injury": inj_for(t) or "",
                })
            y_label = "Score"

        if target_rows:
            df_targets = pd.DataFrame(target_rows).sort_values("Need Score", ascending=False)
            st.dataframe(df_targets, hide_index=True, width="stretch", column_config=_INJURY_COL)

            fig = px.bar(df_targets.head(10), x="Player", y=["Need Score", "Base Score"],
                         barmode="overlay", title="Need-Adjusted vs Base Score",
                         labels={"value": y_label, "variable": ""},
                         color_discrete_map={"Need Score": "steelblue", "Base Score": "lightgray"})
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No available players found that would improve your lineup at a graded position.")
