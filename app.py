import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.agent import FantasyAgent
from src.config import load_config
from src.defense import EMPTY_MATCHUP, build_matchup_table, compute_defense_rankings, get_matchup_info
from src.evaluator import (
    DEFAULT_ROSTER_POSITIONS, analyze_roster_construction, compute_positional_scarcity,
    injury_factor, rank_players, recommend_drop, score_trade, score_waiver, starter_counts,
)
from src.injuries import build_injury_map, injury_status, is_inactive
from src.league_sync import auto_sync, leagues_due, sync_age_seconds
from src.players import RELEVANT_POSITIONS, build_sleeper_name_index, enrich_with_ids, player_label, sleeper_to_player
from src.schedule import (
    fetch_schedules, get_playoff_schedule_strength, get_weather_for_game, next_game_map, playoff_weeks_for,
)
from src.scoring import compute_fantasy_points, effective_scoring, scoring_key, scoring_label
from src.simulator import (
    _summary, simulate_lineup_change, simulate_multi_start, simulate_player, simulate_start_sit, simulate_trade,
)
from src.sleeper_integration import fetch_all_players, players_db_age_seconds, sync_sleeper_league
from src.state_manager import list_leagues, load_league, save_league
from src.vegas import fetch_implied_totals
from src.wrangler import StatsIndex, current_nfl_season, fetch_weekly_stats

load_dotenv()

st.set_page_config(page_title="FF Analyzer", layout="wide")
st.title("🏈 FF Analyzer")

STALE_SYNC_DAYS = 2


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

@st.cache_data(ttl=3600, show_spinner=False)
def load_vegas():
    return fetch_implied_totals()

@st.cache_resource(ttl=6 * 3600, show_spinner="Loading Sleeper player database…")
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
    df = load_weekly_stats()
    if not df.empty:
        df = df.assign(fp=compute_fantasy_points(df, dict(scoring_items)))
    index = StatsIndex(df)
    matchups = build_matchup_table(compute_defense_rankings(df))
    repl = compute_positional_scarcity(index, starter_counts(num_teams, list(roster_positions)))
    return index, matchups, repl

@st.cache_data(ttl=3 * 3600, show_spinner=False)
def cached_weather(home_team: str, gameday: str):
    return get_weather_for_game(home_team, gameday)


def refresh_all_data():
    fetch_weekly_stats(force=True)
    fetch_schedules(force=True)
    fetch_all_players(force=True)
    fetch_implied_totals(force=True)
    st.cache_data.clear()
    st.cache_resource.clear()


# ── Sidebar: settings, data, league ───────────────────────────────────────────

if "ai_provider" not in st.session_state:
    # Default to whichever provider has a key configured
    st.session_state.ai_provider = "gemini" if os.getenv("GOOGLE_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") else "claude"

st.sidebar.header("⚙️ Settings")
st.session_state.ai_provider = st.sidebar.radio(
    "AI Provider", ["claude", "gemini"],
    index=0 if st.session_state.ai_provider == "claude" else 1,
    format_func=lambda x: "Claude (Anthropic)" if x == "claude" else "Gemini (Google)",
)
if st.sidebar.button("🔄 Refresh data", help="Re-download stats, schedule, Sleeper players/injuries and odds"):
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
    return {"attempts": {}, "errors": {}}


# Keep the default leagues current: re-sync any whose saved data is older than auto_sync_minutes
_sync = _sync_state()
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

_weekly_df = load_weekly_stats()
if _weekly_df.empty:
    st.error("Couldn't load NFL stats from nflverse and no cached copy exists. "
             "Check your connection and click **🔄 Refresh data**.")
    st.stop()

_schedules_df = load_schedules()
_next_games = load_next_games()
_vegas_totals = load_vegas()
_players_db, _name_index, _injury_map, _trade_pool = load_players_db()

scoring = effective_scoring(current_data.get("scoring_settings"))
roster_positions = current_data.get("roster_positions") or DEFAULT_ROSTER_POSITIONS
num_teams = int(current_data.get("total_rosters") or 12)
_index, _matchups, _repl_levels = load_analytics(scoring_key(scoring), num_teams, tuple(roster_positions))

roster = enrich_with_ids(current_data.get("roster", []), _players_db, _name_index)
waivers = enrich_with_ids(current_data.get("waivers", []), _players_db, _name_index)
playoff_weeks = playoff_weeks_for(current_data.get("playoff_week_start"))

_season_year = _index.season
_current_season = current_nfl_season()
status_bits = [f"Stats: {_season_year} regular season through week {_index.last_week}",
               f"Scoring: {scoring_label(scoring)}"]
if _schedules_df.empty:
    status_bits.append("Schedule unavailable")
db_age = players_db_age_seconds()
if db_age is not None:
    status_bits.append(f"Injuries as of {db_age / 3600:.0f}h ago")
st.caption(" · ".join(status_bits))
if _season_year and _season_year < _current_season:
    st.info(f"No {_current_season} stats yet — values use the {_season_year} season. "
            f"Matchups use the {_current_season} schedule with {_season_year} defensive data.")
if not _players_db:
    st.warning("Sleeper player database unavailable — injury statuses and player lookups are disabled.")

# ── Sidebar: team, sync status, league settings, budget ─────────────────────

def _fmt_age(seconds: float) -> str:
    if seconds < 3600:
        return f"{max(1, round(seconds / 60))} min ago"
    if seconds < 86400:
        return f"{seconds / 3600:.0f}h ago"
    return f"{seconds / 86400:.0f} days ago"


def _lineup_summary(positions: list) -> str:
    counts = {}
    for pos in positions:
        counts[pos] = counts.get(pos, 0) + 1
    starting = [f"{n} {pos}" if n > 1 else pos for pos, n in counts.items() if pos not in ("BN", "IR", "TAXI")]
    extras = [f"{counts[k]} {label}" for k, label in (("BN", "bench"), ("IR", "IR"), ("TAXI", "taxi")) if k in counts]
    return ", ".join(starting) + (f" (+ {', '.join(extras)})" if extras else "")


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
        st.sidebar.caption(f"Synced {_fmt_age(age)}{auto}")
        if not is_default and age > STALE_SYNC_DAYS * 86400:
            st.sidebar.warning("Roster and waiver pool may be stale — sync to drop players already claimed.")
    elif roster:
        st.sidebar.warning("Sync time unknown (older file). Sync to refresh the roster and waiver pool.")
    if err := _sync["errors"].get(selected_league):
        st.sidebar.warning(f"Auto-sync failed: {err}. Showing the last saved data.")

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
                         f"**Lineup:** {_lineup_summary(current_data.get('roster_positions') or DEFAULT_ROSTER_POSITIONS)}",
                         f"**Scoring:** {scoring_label(scoring)}"]
        settings_rows += [f"- {label}: {scoring[key]:+g}" for key, label in _NOTABLE_SCORING if scoring.get(key)]
        if current_data.get("faab_enabled", True):
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
faab_enabled = current_data.get("faab_enabled", True)
with st.sidebar.expander("💰 FAAB Budget", expanded=True):
    if not faab_enabled:
        st.caption("This league doesn't use FAAB — bids are shown for reference only.")
    budget = st.number_input("Remaining Budget ($)", min_value=0, max_value=10000, value=synced_budget, step=1,
                             key=f"budget_{selected_league}",
                             help="Synced from Sleeper. Change it to test bids; it isn't saved.")


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
    return "n/a (team D)" if p.get("position") == "DEF" else "⚠️ not found"

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


roster_by_name = {p["name"]: p for p in roster}
roster_stats = {p["name"]: stats_for(p) for p in roster}


# ── UI Helpers ────────────────────────────────────────────────────────────────

def _verdict_color(v: str) -> str:
    return {"STRONG ADD": "green", "ADD": "green", "STREAM": "orange", "AVOID": "red",
            "ACCEPT": "green", "NEGOTIATE": "orange", "DECLINE": "red"}.get(v, "gray")

def _grade_color(grade: str) -> str:
    return {"A": "green", "B": "blue", "C": "orange", "D": "red", "F": "red"}.get(grade, "gray")

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

def _ai_block(state_key: str, request_id, button_label: str, run):
    """AI button whose output survives reruns (keyed to the current request)."""
    if st.button(button_label, key=f"{state_key}_btn"):
        with st.spinner(f"Analyzing with {st.session_state.ai_provider}..."):
            st.session_state[state_key] = (request_id, run())
    stored = st.session_state.get(state_key)
    if stored and stored[0] == request_id:
        st.markdown(stored[1])


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_roster, tab_waivers, tab_start_sit, tab_trade, tab_roster_mgmt = st.tabs(
    ["📋 Roster", "🔄 Waivers", "⚖️ Who to Start", "🤝 Trade", "🏗️ Roster Construction"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ROSTER
# ════════════════════════════════════════════════════════════════════════════
_SLOT_ORDER = {slot: i for i, slot in enumerate(dict.fromkeys(
    [s for s in roster_positions if s not in ("BN", "IR", "TAXI")] + ["BN", "IR", "Taxi"]))}

with tab_roster:
    st.subheader(current_data.get("team_name") or "Your Roster")
    if not roster:
        st.info("Sync a league from the sidebar.")
    else:
        mu = current_data.get("matchup")
        if mu and mu.get("bye"):
            st.caption(f"Week {mu['week']}: bye")
        elif mu and mu.get("opponent"):
            opp_rec = mu.get("opponent_record") or {}
            opp_rec_str = f" ({opp_rec.get('wins', 0)}-{opp_rec.get('losses', 0)})" if opp_rec else ""
            mc1, mc2, mc3 = st.columns([1, 1, 2])
            mc1.metric(f"Week {mu['week']} — You", f"{mu['points']:g}")
            mc2.metric(f"{mu['opponent']}{opp_rec_str}", f"{mu['opponent_points']:g}")
            mc3.caption("Live points from Sleeper as of the last sync.")

        has_slots = any(p.get("slot") for p in roster)
        flagged = [p["name"] for p in roster
                   if p.get("slot") not in (None, "BN", "IR", "Taxi") and is_inactive(inj_for(p))]
        if flagged:
            st.error(f"🚨 Starting an inactive player: {', '.join(flagged)} — swap them out in Sleeper.")
        n_starting = len([s for s in roster_positions if s not in ("BN", "IR", "TAXI")])
        n_started = len([p for p in roster if p.get("slot") not in (None, "BN", "IR", "Taxi")])
        if has_slots and n_started < n_starting:
            st.warning(f"⚠️ {n_starting - n_started} empty starting slot(s) in your Sleeper lineup.")

        rows = []
        for p in sorted(roster, key=lambda p: _SLOT_ORDER.get(p.get("slot"), len(_SLOT_ORDER))):
            s = roster_stats[p["name"]]
            next_g, m = matchup_for(p)
            rows.append({
                **({"Slot": p.get("slot", "")} if has_slots else {}),
                "Player": p["name"], "Pos": p.get("position", ""), "Team": p.get("team", ""),
                "Season PPG": s["season"]["ppg"], "Recent PPG": s["recent"]["ppg"],
                "Games": s["season"]["games"],
                "Injury": inj_for(p) or "",
                "Next Matchup": f"vs {next_g['opponent']} ({m['rank_label']})" if next_g else "—",
                "Stats": stats_note(p, s),
            })
        df_roster = pd.DataFrame(rows)
        st.dataframe(df_roster, hide_index=True, width="stretch")

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
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

        st.divider()
        st.subheader("Player Deep Dive")

        if not enriched_sorted:
            st.caption("No players match the current filters.")
        else:
            sel_i = st.selectbox("Select player", range(len(enriched_sorted)),
                                 format_func=lambda i: player_label(enriched_sorted[i]))
            p_data = enriched_sorted[sel_i]
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
            drop = recommend_drop(pos, roster, roster_stats, _repl_levels) if roster else None

            col_g, col_m = st.columns([1, 2])
            with col_g:
                st.plotly_chart(_score_gauge(ws["score"]), width="stretch")
            with col_m:
                v = ws["verdict"]
                st.markdown(f"### :{_verdict_color(v)}[{v}]")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Expected Pts/Wk", ws["expected_pts"])
                m2.metric("Consistency", f"{ws['consistency_pct']:.0f}%")
                m3.metric("vs Replacement", f"{ws['surplus_vs_replacement']:+.1f}")
                if def_info["pts_allowed"] is not None:
                    m4.metric("Matchup Adj", f"×{def_info['multiplier']:.2f}",
                              f"{'above' if def_info['multiplier'] > 1 else 'below'} avg")

                bid_amount = round(ws["faab_bid_pct"] * budget)
                st.metric("Recommended FAAB Bid", f"${bid_amount}",
                          help=f"{ws['faab_bid_pct'] * 100:.0f}% of your ${budget} remaining budget")

                if drop:
                    drop_color = "red" if drop["surplus"] < 0 else "orange"
                    st.markdown(f"**✂️ Drop:** :{drop_color}[{drop['name']}]"
                                f" ({drop['position']} · {drop['adj_value']} adj pts) — {drop['reason']}")

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

            with st.expander("🎲 Monte Carlo Simulation"):
                waiver_samples = simulate_player(player_stats, selected_waiver)
                smry = _summary(waiver_samples)
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Floor (10th pct)", smry["floor"])
                mc2.metric("Median", smry["median"])
                mc3.metric("Ceiling (90th pct)", smry["ceiling"])
                mc4.metric("Std Dev", smry["std"])

                rc_map = {selected_waiver: waiver_samples}
                if drop:
                    rc_map[f"Drop: {drop['name']}"] = simulate_player(roster_stats[drop["name"]], drop["name"])
                st.plotly_chart(_raincloud_fig(rc_map, "Simulated Outcome Distributions (10,000 games)"),
                                width="stretch")

            with st.expander("🤖 AI Narrative Analysis (optional)"):
                st.caption("Needs `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY` in .env.")
                ctx = player_context(p_data, player_stats)
                ctx.update({"Model score": f"{ws['score']} ({ws['verdict']})",
                            "Model FAAB bid": f"${bid_amount}"})
                _ai_block("ai_waiver", (selected_league, p_data.get("player_id") or selected_waiver),
                          "Get AI Analysis",
                          lambda: FantasyAgent(st.session_state.ai_provider).analyze_waiver(
                              selected_waiver, player_stats, roster, scoring, budget, ctx))


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHO TO START
# ════════════════════════════════════════════════════════════════════════════
with tab_start_sit:
    st.subheader("Who to Start")

    if not roster:
        st.info("Sync a league to compare roster players.")
    else:
        selected_players = st.multiselect("Select 2–4 players to compare", list(roster_by_name), max_selections=4)
        if st.button("⚖️ Compare"):
            st.session_state.ss_request = tuple(selected_players)

        if len(selected_players) < 2:
            st.caption("Select at least 2 players.")
        elif st.session_state.get("ss_request") != tuple(selected_players):
            st.caption("Click **Compare** to run the analysis.")
        else:
            named_stats = {n: roster_stats[n] for n in selected_players}
            adjustments, multipliers, matchups = {}, {}, {}
            for n in selected_players:
                p_obj = roster_by_name[n]
                next_g, m = matchup_for(p_obj)
                inj = inj_for(p_obj)
                adjustments[n] = {"matchup": m["multiplier"], "injury": inj}
                multipliers[n] = m["multiplier"] * injury_factor(inj)
                matchups[n] = (next_g, m)

            ranked = rank_players(named_stats, adjustments)
            mc_result = simulate_multi_start(named_stats, multipliers)
            top = ranked[0]
            conf_label = top.get("confidence_label", "")
            banner_color = "green" if conf_label == "Clear" else ("orange" if conf_label == "Lean" else "gray")
            mc_pct = round(mc_result["win_probs"].get(top["name"], 0) * 100)
            st.markdown(f"## :{banner_color}[START: {top['name']}]  "
                        f"— {conf_label} · Projected edge {top.get('margin_to_next', 0):.1f} pts · "
                        f"MC win prob {mc_pct}%")
            st.caption("Projection = adjusted value × opponent defense multiplier × injury factor "
                       "(Questionable ×0.85, Doubtful ×0.25, Out/IR ×0).")

            missing = [n for n in selected_players if not named_stats[n]["found"]]
            if missing:
                st.warning(f"No {_season_year} stats for: {', '.join(missing)} — comparison is unreliable.")

            cols = st.columns(len(ranked))
            for col, r in zip(cols, ranked):
                name = r["name"]
                is_start = r["rank"] == 1
                mc_smry = mc_result["summaries"][name]
                next_g, m = matchups[name]
                with col:
                    st.markdown(f"**{name}**")
                    start_lbl = "✅ START" if is_start else f"#{r['rank']} BENCH"
                    st.markdown(f":{'green' if is_start else 'red'}[{start_lbl}]")
                    st.metric("Projected", r["start_value"], help=f"Base adj value {r['adj_value']}")
                    st.metric("Consistency", f"{r['consistency'] * 100:.0f}%")
                    st.metric("MC Win Prob", f"{round(mc_result['win_probs'][name] * 100)}%")
                    st.metric("Floor / Ceiling", f"{mc_smry['floor']} / {mc_smry['ceiling']}")
                    if next_g:
                        st.caption(f"vs {next_g['opponent']} — {m['rank_label']} (×{m['multiplier']:.2f})")
                    else:
                        st.caption("No upcoming game")
                    if r["injury"]:
                        st.error(f"🏥 {r['injury']}")

            rows = []
            for name, stats in named_stats.items():
                v = next(r for r in ranked if r["name"] == name)
                rows += [
                    {"Player": name, "Metric": "Season PPG", "Value": stats["season"]["ppg"]},
                    {"Player": name, "Metric": "Recent PPG", "Value": stats["recent"]["ppg"] or 0},
                    {"Player": name, "Metric": "Projected", "Value": v["start_value"]},
                ]
            st.plotly_chart(px.bar(pd.DataFrame(rows), x="Metric", y="Value", color="Player", barmode="group",
                                   title="Head-to-Head Comparison",
                                   category_orders={"Metric": ["Season PPG", "Recent PPG", "Projected"]}),
                            width="stretch")

            st.plotly_chart(_raincloud_fig(mc_result["samples"], "Simulated Outcome Distributions (10,000 games)"),
                            width="stretch")

            if len(selected_players) == 2:
                p1, p2 = selected_players
                hth = simulate_start_sit(p1, named_stats[p1], p2, named_stats[p2], multipliers)
                hc1, hc2 = st.columns(2)
                hc1.metric(f"P({p1} outscores)", f"{hth['p1_win_prob'] * 100:.0f}%")
                hc2.metric(f"P({p2} outscores)", f"{hth['p2_win_prob'] * 100:.0f}%")

            with st.expander("Weekly scoring history"):
                fig2 = go.Figure()
                for name, stats in named_stats.items():
                    if stats["weekly"]:
                        fig2.add_trace(go.Scatter(x=stats["weeks"], y=stats["weekly"], mode="lines+markers", name=name))
                fig2.update_layout(xaxis_title="Week", yaxis_title="Fantasy Points", title="Season Weekly Scores")
                st.plotly_chart(fig2, width="stretch")

            with st.expander("🤖 AI Narrative Analysis (optional — 2 players only)"):
                if len(selected_players) == 2:
                    p1, p2 = selected_players
                    contexts = {n: player_context(roster_by_name[n], named_stats[n]) for n in (p1, p2)}
                    _ai_block("ai_ss", (selected_league, p1, p2), "Get AI Analysis",
                              lambda: FantasyAgent(st.session_state.ai_provider).analyze_start_sit(
                                  p1, p2, named_stats[p1], named_stats[p2], scoring, contexts))
                else:
                    st.caption("AI narrative is only available for 2-player comparisons.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRADE ANALYZER
# ════════════════════════════════════════════════════════════════════════════
with tab_trade:
    st.subheader("Trade Analyzer")

    roster_ids = {p.get("player_id") for p in roster if p.get("player_id")}
    trade_pool = {p["player_id"]: p for p in _trade_pool if p["player_id"] not in roster_ids}

    giving = st.multiselect("You Give (from your roster)", list(roster_by_name))
    recv_ids = st.multiselect("You Receive", list(trade_pool),
                              format_func=lambda pid: player_label(trade_pool[pid]),
                              placeholder="Search any NFL player…")
    receiving_players = [trade_pool[pid] for pid in recv_ids]
    receiving = [p["name"] for p in receiving_players]

    request = (selected_league, tuple(giving), tuple(p["player_id"] for p in receiving_players))
    if st.button("🤝 Analyze Trade", disabled=not (giving or receiving)):
        st.session_state.trade_request = request

    if (giving or receiving) and st.session_state.get("trade_request") == request:
        give_players = [roster_by_name[n] for n in giving]
        stats_map = {p["name"]: (roster_stats.get(p["name"]) or stats_for(p)) for p in give_players + receiving_players}
        all_stats = {**roster_stats, **stats_map}

        missing = [n for n, s in stats_map.items() if not s["found"]]
        if missing:
            st.warning(f"No {_season_year} stats for: {', '.join(missing)} — valued at 0.")

        trade = score_trade(roster, giving, receiving_players, all_stats, _repl_levels, roster_positions)
        if trade["method"] == "lineup":
            mc = simulate_lineup_change([s["name"] for s in trade["before"]["starters"]],
                                        [s["name"] for s in trade["after"]["starters"]], all_stats)
            p_better = mc["p_better"]
            rc = {"Lineup Before": mc["before_samples"], "Lineup After": mc["after_samples"]}
            rc_title = "Weekly Starting-Lineup Total (10,000 simulations)"
            headline = (f"Starting lineup {trade['delta']:+.1f} pts/wk ({trade['delta_pct'] * 100:+.1f}%) · "
                        f"P(better week) = {round(p_better * 100)}%")
        else:
            mc = simulate_trade({n: stats_map[n] for n in giving}, {n: stats_map[n] for n in receiving})
            p_better = mc["p_receive_better"]
            rc = {"You Give": mc["give_samples"], "You Receive": mc["receive_samples"]}
            rc_title = "Total Weekly Output (10,000 simulations)"
            headline = f"Value over replacement {trade['delta']:+.1f} pts/wk · P(receive side wins) = {round(p_better * 100)}%"

        verdict = trade["verdict"]
        st.markdown(f"## :{_verdict_color(verdict)}[{verdict}]  — {headline}")
        st.caption("Verdict is based on your best possible starting lineup before vs. after the trade, so "
                   "bench depth only counts if it would start. Value over replacement (VORP) is shown per player.")

        vc1, vc2 = st.columns(2)
        for col, side, header, total in [(vc1, trade["you_give"], "You Give", trade["give_vorp"]),
                                         (vc2, trade["you_receive"], "You Receive", trade["recv_vorp"])]:
            with col:
                st.markdown(f"**{header}**")
                for name, val in side.items():
                    inj = inj_for(roster_by_name.get(name) or next(p for p in receiving_players if p["name"] == name))
                    st.metric(f"{name}{f' 🏥 {inj}' if inj else ''}", f"{val['adj_value']} adj pts/wk",
                              f"VORP {val['surplus']:+.1f}", delta_color="off")
                st.metric("Total VORP", f"{total} pts/wk")

        if trade["method"] == "lineup":
            with st.expander("Starting lineup before vs. after"):
                lc1, lc2 = st.columns(2)
                for col, key, title in [(lc1, "before", "Before"), (lc2, "after", "After")]:
                    lineup = trade[key]
                    col.markdown(f"**{title}** — {lineup['total']} adj pts/wk")
                    col.dataframe(pd.DataFrame([{"Slot": s["slot"], "Player": s["name"] or "(empty)",
                                                 "Value": s["value"]} for s in lineup["starters"]]),
                                  hide_index=True, width="stretch")

        bar_rows = ([{"Side": "Give", "Player": n, "Adj Value": v["adj_value"]} for n, v in trade["you_give"].items()]
                    + [{"Side": "Receive", "Player": n, "Adj Value": v["adj_value"]} for n, v in trade["you_receive"].items()])
        if bar_rows:
            st.plotly_chart(px.bar(pd.DataFrame(bar_rows), x="Player", y="Adj Value", color="Side",
                                   color_discrete_map={"Give": "firebrick", "Receive": "seagreen"},
                                   title="Trade Value Comparison"), width="stretch")

        st.plotly_chart(_raincloud_fig(rc, rc_title), width="stretch")

        wk_str = "–".join(str(w) for w in (playoff_weeks[0], playoff_weeks[-1]))
        with st.expander(f"📅 Playoff Schedule Strength (Weeks {wk_str})"):
            psc1, psc2 = st.columns(2)
            for col, players, header in [(psc1, give_players, "You Give"), (psc2, receiving_players, "You Receive")]:
                with col:
                    st.markdown(f"**{header}**")
                    for p in players:
                        pss = get_playoff_schedule_strength(p.get("team", ""), _schedules_df, _matchups,
                                                            p.get("position", ""), playoff_weeks)
                        st.markdown(f"{p['name']}: :{_grade_color(pss['grade'])}[Grade {pss['grade']}] "
                                    f"(avg mult {pss['avg_multiplier']:.2f})")
                        for wk in pss["weeks"]:
                            st.caption(f"  Wk {wk['week']}: vs {wk['opponent']} — {wk['matchup']}")

        with st.expander("🤖 AI Narrative Analysis (optional)"):
            contexts = {p["name"]: player_context(p, stats_map[p["name"]]) for p in give_players + receiving_players}
            summary = {"Verdict": verdict, "Lineup / value change": headline,
                       "VORP given": trade["give_vorp"], "VORP received": trade["recv_vorp"]}
            _ai_block("ai_trade", request, "Get AI Verdict",
                      lambda: FantasyAgent(st.session_state.ai_provider).analyze_trade(
                          giving, receiving, stats_map, scoring, contexts, summary))


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ROSTER CONSTRUCTION
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
        analysis = analyze_roster_construction(roster, avail_pool, roster_stats, avail_stats,
                                               _repl_levels, avail_mults, roster_positions)

        pg = analysis["position_grades"]
        weaknesses = analysis["weaknesses"]

        st.markdown("### Positional Strength")
        grade_cols = st.columns(len(pg))
        for col, (pos, g) in zip(grade_cols, pg.items()):
            col.metric(pos, f"Grade {g['grade']}", f"×{g['strength_ratio']:.2f} vs replacement")
            col.caption(f"{g['depth_count']} players · starters avg {g['avg_starter_val']:.1f} adj pts")
        st.caption("Graded positions follow this league's lineup slots. DEF isn't graded — "
                   "team defense stats aren't in the nflverse data.")

        if weaknesses:
            st.warning(f"⚠️ Positional weaknesses: **{', '.join(weaknesses)}** — prioritized in recommendations below")
        else:
            st.success("✅ No critical positional weaknesses")

        st.divider()
        st.markdown("### Need-Adjusted Waiver Targets")
        st.caption("Score = waiver score (incl. next-game matchup) + need bonus (up to +20 for filling a weakness)")

        targets = analysis["targeted_adds"]
        if targets:
            target_rows = []
            for t in targets:
                next_g, _ = matchup_for(t)
                target_rows.append({
                    "Player": t["name"], "Pos": t.get("position", ""), "Team": t.get("team", ""),
                    "Need Score": t["need_adjusted_score"], "Base Score": t["base_score"],
                    "Fills Need": "⭐ YES" if t["fills_need"] else "—",
                    "Verdict": t["verdict"], "Adj Value": t["adj_value"],
                    "Next Game": f"vs {next_g['opponent']}" if next_g else None,
                    "Injury": inj_for(t) or "",
                })
            df_targets = pd.DataFrame(target_rows).sort_values("Need Score", ascending=False)
            st.dataframe(df_targets, hide_index=True, width="stretch")

            fig = px.bar(df_targets.head(10), x="Player", y=["Need Score", "Base Score"],
                         barmode="overlay", title="Need-Adjusted vs Base Waiver Score",
                         labels={"value": "Score", "variable": ""},
                         color_discrete_map={"Need Score": "steelblue", "Base Score": "lightgray"})
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No available players found — sync your league to populate the waiver pool.")
