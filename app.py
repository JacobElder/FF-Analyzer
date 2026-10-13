import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.agent import FantasyAgent, DEFAULT_SCORING
from src.state_manager import get_roster, update_roster, get_league_list
from src.evaluator import (
    score_waiver, score_trade,
    rank_players, player_value, compute_positional_scarcity,
    analyze_roster_construction, recommend_drop,
)
from src.sleeper_integration import sync_sleeper_league
from src.wrangler import (
    fetch_weekly_stats, get_player_stats, rank_available_players,
    get_usage_trends, current_nfl_season,
)
from src.simulator import simulate_multi_start, simulate_start_sit, simulate_trade, simulate_player, _summary
from src.defense import compute_defense_rankings, get_matchup_info
from src.schedule import fetch_schedules, get_next_game, get_weather_for_game, get_playoff_schedule_strength
from src.injuries import fetch_injury_report, injury_badge
from src.vegas import fetch_implied_totals

st.set_page_config(page_title="FF Analyzer", layout="wide")
st.title("🏈 FF Analyzer")


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def load_weekly_stats():
    df = fetch_weekly_stats()
    season = int(df["season"].iloc[0]) if not df.empty and "season" in df.columns else current_nfl_season()
    return df, season

@st.cache_data(ttl=86400)
def load_schedules():
    return fetch_schedules()

@st.cache_data(ttl=14400)   # 4h during season
def load_injuries():
    return fetch_injury_report()

@st.cache_data(ttl=3600)    # 1h
def load_vegas():
    return fetch_implied_totals()

@st.cache_data(ttl=604800)
def load_sleeper_players():
    from src.sleeper_integration import fetch_all_players
    return fetch_all_players()


# ── Session state ─────────────────────────────────────────────────────────────

if "ai_provider" not in st.session_state:
    st.session_state.ai_provider = "claude"


# ── Load data up front ────────────────────────────────────────────────────────

_weekly_df, _season_year = load_weekly_stats()
_schedules_df = load_schedules()
_injury_map = load_injuries()
_vegas_totals = load_vegas()
st.caption(f"Stats: {_season_year} NFL season")


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Settings")
st.session_state.ai_provider = st.sidebar.radio(
    "AI Provider", ["claude", "gemini"],
    index=0 if st.session_state.ai_provider == "claude" else 1,
    format_func=lambda x: "Claude (Anthropic)" if x == "claude" else "Gemini (Google)",
)

st.sidebar.divider()
st.sidebar.header("📁 League")
leagues = get_league_list()
selected_league = st.sidebar.selectbox("Select League", leagues)

with st.sidebar.expander("➕ Sync from Sleeper"):
    sleeper_user = st.text_input("Sleeper Username", value="JElder52")
    target_l_name = st.text_input("League ID or Name", value="1236161337357242368")
    if st.button("Sync"):
        with st.spinner("Syncing from Sleeper..."):
            data, err = sync_sleeper_league(sleeper_user, target_l_name)
            if data:
                update_roster(data, target_l_name)
                st.success(f"Synced: {data['team_name']}")
                st.rerun()
            else:
                st.error(err)

current_data = get_roster(selected_league)
roster = current_data.get("roster", [])
scoring = current_data.get("scoring_settings") or DEFAULT_SCORING
waivers = current_data.get("waivers", [])

synced_budget = current_data.get("budget", 100)
with st.sidebar.expander("💰 FAAB Budget", expanded=True):
    new_budget = st.number_input(
        "Current Budget ($)", min_value=0, max_value=10000,
        value=int(synced_budget), step=1,
    )
    if new_budget != synced_budget:
        if st.button("Save Budget"):
            current_data["budget"] = new_budget
            update_roster(current_data, selected_league)
            st.rerun()
budget = new_budget

st.sidebar.divider()
st.sidebar.header("📋 Roster")
if roster:
    roster_df_sidebar = pd.DataFrame(roster)[["name", "position"]]
    # Inject injury badges
    roster_df_sidebar["status"] = roster_df_sidebar["name"].apply(
        lambda n: injury_badge(n, _injury_map) or "✅"
    )
    st.sidebar.dataframe(roster_df_sidebar, hide_index=True, use_container_width=True)
else:
    st.sidebar.info("No roster. Sync a league first.")


# ── Shared computations ───────────────────────────────────────────────────────

# Defense rankings from the loaded weekly_df (free — no extra API)
@st.cache_data(ttl=86400)
def _defense_rankings(scoring_key: str):
    # scoring_key is just a hashable proxy for the scoring dict
    return compute_defense_rankings(_weekly_df, scoring)

_defense_df = _defense_rankings(str(sorted(scoring.items())))

# Dynamic replacement levels
@st.cache_data(ttl=86400)
def _replacement_levels():
    return compute_positional_scarcity(_weekly_df, scoring)

_repl_levels = _replacement_levels()


# ── UI Helpers ────────────────────────────────────────────────────────────────

def _verdict_color(v: str) -> str:
    return {"STRONG ADD": "green", "ADD": "green", "STREAM": "orange", "AVOID": "red",
            "ACCEPT": "green", "NEGOTIATE": "orange", "DECLINE": "red"}.get(v, "gray")

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
    Raincloud plot: half-violin + embedded box + jittered points.
    2 players → mirrored butterfly violin.
    3-4 players → grouped violins.
    names_samples: {player_name: np.ndarray of MC samples}
    """
    names = list(names_samples.keys())
    rng = np.random.default_rng(42)
    fig = go.Figure()

    if len(names) == 2:
        for i, (name, samples) in enumerate(names_samples.items()):
            pts = rng.choice(samples, size=min(300, len(samples)), replace=False)
            color = _RAIN_COLORS[i]
            side = "negative" if i == 0 else "positive"
            pointpos = 1.6 if side == "negative" else -1.6
            fig.add_trace(go.Violin(
                y=pts, name=name, side=side,
                fillcolor=color, line_color=color, opacity=0.65,
                box_visible=True, meanline_visible=True,
                points="all", jitter=0.05, pointpos=pointpos,
                marker=dict(color=color, size=2.5, opacity=0.35),
                spanmode="hard",
            ))
        fig.update_layout(violinmode="overlay")
    else:
        for i, (name, samples) in enumerate(names_samples.items()):
            pts = rng.choice(samples, size=min(300, len(samples)), replace=False)
            color = _RAIN_COLORS[i % len(_RAIN_COLORS)]
            fig.add_trace(go.Violin(
                y=pts, x=[name] * len(pts), name=name,
                fillcolor=color, line_color=color, opacity=0.65,
                box_visible=True, meanline_visible=True,
                points="all", jitter=0.3, pointpos=-0.6,
                marker=dict(color=color, size=2.5, opacity=0.35),
                spanmode="hard",
            ))
        fig.update_layout(violinmode="group")

    fig.update_layout(
        title=title,
        yaxis_title="Simulated Fantasy Points",
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig

def _grade_color(grade: str) -> str:
    return {"A": "green", "B": "blue", "C": "orange", "D": "red", "F": "red"}.get(grade, "gray")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_roster, tab_waivers, tab_start_sit, tab_trade, tab_roster_mgmt = st.tabs(
    ["📋 Roster", "🔄 Waivers", "⚖️ Who to Start", "🤝 Trade", "🏗️ Roster Construction"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ROSTER
# ════════════════════════════════════════════════════════════════════════════
with tab_roster:
    st.subheader("Your Roster")
    if not roster:
        st.info("Sync a league from the sidebar.")
    else:
        enriched = rank_available_players(roster, _weekly_df, scoring)

        # Enrich with injury and upcoming matchup
        rows = []
        for p in enriched:
            name = p["name"]
            pos = p.get("position", "")
            team = p.get("team", "")

            # Injury
            inj = injury_badge(name, _injury_map)
            inj_str = inj if inj else ""

            # Next game opponent + defense rank
            next_g = get_next_game(team, _schedules_df) if team else None
            if next_g:
                opp = next_g["opponent"]
                m = get_matchup_info(opp, pos, _defense_df)
                matchup_str = f"vs {opp} ({m['rank_label']})"
            else:
                matchup_str = "Offseason"

            rows.append({
                "Player": name, "Pos": pos,
                "Season PPG": p.get("season_ppg", 0),
                "Recent PPG": p.get("recent_ppg", 0),
                "Injury": inj_str,
                "Next Matchup": matchup_str,
            })

        df_roster = pd.DataFrame(rows)
        st.dataframe(df_roster, hide_index=True, use_container_width=True)

        fig = px.bar(
            df_roster.sort_values("Season PPG", ascending=False).head(12),
            x="Player", y="Season PPG", color="Pos",
            title=f"Season PPG — {_season_year}",
        )
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — WAIVERS
# ════════════════════════════════════════════════════════════════════════════
with tab_waivers:
    st.subheader("Waiver Wire")

    if not waivers:
        st.info("Sync a league to load the waiver pool.")
    else:
        pos_opts = sorted({w["position"] for w in waivers} & {"QB", "RB", "WR", "TE", "K", "DEF"})
        sel_pos = st.multiselect("Filter by Position", pos_opts, default=["RB", "WR", "TE"])
        filtered = [w for w in waivers if w["position"] in sel_pos] if sel_pos else waivers

        with st.spinner("Loading stats..."):
            enriched = rank_available_players(filtered, _weekly_df, scoring)

        sort_by = st.radio("Sort by", ["Trending", "Season PPG", "Recent PPG"], horizontal=True)
        sk = {"Trending": "trend_count", "Season PPG": "season_ppg", "Recent PPG": "recent_ppg"}[sort_by]
        enriched_sorted = sorted(enriched, key=lambda x: x.get(sk, 0), reverse=True)

        # Build display table with injury + matchup
        table_rows = []
        for p in enriched_sorted:
            name, pos, team = p["name"], p.get("position", ""), p.get("team", "")
            inj = injury_badge(name, _injury_map)
            next_g = get_next_game(team, _schedules_df) if team else None
            opp_str, mult = "—", 1.0
            if next_g:
                opp = next_g["opponent"]
                m = get_matchup_info(opp, pos, _defense_df)
                opp_str = f"vs {opp} ({m['rank_label']})"
                mult = m["multiplier"]
            vegas_total = _vegas_totals.get(team, None)
            table_rows.append({
                "Name": name, "Pos": pos, "Team": team,
                "Season PPG": p.get("season_ppg", 0),
                "Recent PPG": p.get("recent_ppg", 0),
                "Trending": p.get("trend_count", 0),
                "Next Matchup": opp_str,
                "Injury": inj or "",
                "Implied Total": vegas_total or "—",
            })

        df_w = pd.DataFrame(table_rows)
        st.dataframe(df_w, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Player Deep Dive")
        waiver_names = [p["name"] for p in enriched_sorted]
        selected_waiver = st.selectbox("Select player", waiver_names)

        if selected_waiver:
            p_data = next((p for p in enriched_sorted if p["name"] == selected_waiver), {})
            pos = p_data.get("position", "WR")
            team = p_data.get("team", "")
            player_stats = get_player_stats(selected_waiver, _weekly_df, scoring)

            # Defense matchup for this player's next game
            next_g = get_next_game(team, _schedules_df) if team else None
            def_info = {"multiplier": 1.0, "rank_label": "Unknown", "pts_allowed": None}
            opp_team = None
            if next_g:
                opp_team = next_g["opponent"]
                def_info = get_matchup_info(opp_team, pos, _defense_df)

            ws = score_waiver(
                player_stats, pos, p_data.get("trend_count", 0),
                defense_multiplier=def_info["multiplier"],
                replacement_levels=_repl_levels,
            )

            # ── Row 1: gauge + core metrics ──
            col_g, col_m = st.columns([1, 2])
            with col_g:
                st.plotly_chart(_score_gauge(ws["score"]), use_container_width=True)
            with col_m:
                v = ws["verdict"]
                st.markdown(f"### :{_verdict_color(v)}[{v}]")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Expected Pts/Wk", ws["expected_pts"])
                m2.metric("Consistency", f"{ws['consistency_pct']:.0f}%")
                m3.metric("vs Replacement", f"{ws['surplus_vs_replacement']:+.1f}")
                if def_info["pts_allowed"]:
                    delta = f"{'above' if def_info['multiplier'] > 1 else 'below'} avg"
                    m4.metric("Matchup Adj", f"×{def_info['multiplier']:.2f}", delta)

                budget_slider = st.slider("FAAB Budget", 0, max(budget * 2, 200), int(budget))
                bid_amount = round(ws["faab_bid_pct"] * budget_slider)
                st.metric("Recommended FAAB Bid", f"${bid_amount}",
                          help=f"{ws['faab_bid_pct']*100:.0f}% of budget")

                if roster:
                    roster_stats_drop = {p["name"]: get_player_stats(p["name"], _weekly_df, scoring) for p in roster}
                    drop = recommend_drop(pos, roster, roster_stats_drop, _repl_levels)
                    if drop:
                        drop_color = "red" if drop["surplus"] < 0 else "orange"
                        st.markdown(
                            f"**✂️ Drop:** :{drop_color}[{drop['name']}]"
                            f" ({drop['position']} · {drop['adj_value']} adj pts) — {drop['reason']}"
                        )

            # ── Row 2: Context panels ──
            ctx1, ctx2, ctx3 = st.columns(3)

            with ctx1:
                st.markdown("**🛡️ Matchup**")
                if next_g:
                    hw = "🏠 Home" if next_g["is_home"] else "✈️ Away"
                    st.write(f"Week {next_g['week']} {hw} vs **{opp_team}**")
                    st.write(f"Rank: {def_info.get('rank', '—')} | {def_info['rank_label']}")
                    st.write(f"Avg allowed vs {pos}: {def_info.get('pts_allowed', '—')} pts/gm")
                    # Weather
                    wx = get_weather_for_game(next_g)
                    if wx:
                        if wx.get("dome"):
                            st.write("🏟️ Indoor — no weather")
                        else:
                            st.write(f"🌡️ {wx.get('temp_f', '?')}°F | 💨 {wx.get('wind_mph', '?')} mph")
                            for flag in wx.get("flags", []):
                                st.warning(flag)
                else:
                    st.write("No upcoming game (offseason)")

            with ctx2:
                st.markdown("**📈 Usage Trends**")
                usage = get_usage_trends(selected_waiver, _weekly_df)
                if usage:
                    for metric, label in [("target_share", "Target Share"),
                                           ("air_yards_share", "Air Yards Share"),
                                           ("wopr", "WOPR")]:
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
                if vt:
                    st.metric("Team Implied Total", vt)
                else:
                    st.write("Vegas: offseason / no key")
                inj = injury_badge(selected_waiver, _injury_map)
                if inj:
                    st.error(f"🏥 {inj}")
                else:
                    st.success("✅ Healthy")

            # ── Weekly sparkline ──
            weekly_pts = player_stats.get("weekly", [])
            if weekly_pts:
                fig = px.line(
                    x=list(range(1, len(weekly_pts) + 1)), y=weekly_pts,
                    labels={"x": "Week", "y": "Fantasy Points"},
                    title=f"{selected_waiver} — Weekly Scoring",
                )
                fig.add_hline(y=ws["expected_pts"], line_dash="dash", line_color="steelblue",
                              annotation_text="Expected")
                st.plotly_chart(fig, use_container_width=True)

            # ── Monte Carlo ──
            with st.expander("🎲 Monte Carlo Simulation"):
                waiver_samples = simulate_player(player_stats)
                smry = _summary(waiver_samples)
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Floor (10th pct)", smry["floor"])
                mc2.metric("Median", smry["median"])
                mc3.metric("Ceiling (90th pct)", smry["ceiling"])
                mc4.metric("Std Dev", smry["std"])

                # Build raincloud — add drop candidate for comparison if available
                rc_map = {selected_waiver: waiver_samples}
                if roster:
                    roster_stats_drop = {p["name"]: get_player_stats(p["name"], _weekly_df, scoring) for p in roster}
                    drop_cand = recommend_drop(pos, roster, roster_stats_drop, _repl_levels)
                    if drop_cand:
                        drop_stats = get_player_stats(drop_cand["name"], _weekly_df, scoring)
                        rc_map[f"Drop: {drop_cand['name']}"] = simulate_player(drop_stats)

                st.plotly_chart(
                    _raincloud_fig(rc_map, "Simulated Outcome Distributions (10,000 games)"),
                    use_container_width=True,
                )

            with st.expander("🤖 AI Narrative Analysis (optional)"):
                st.caption("Add `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY` to .env to enable.")
                if st.button("Get AI Analysis", key="waiver_ai"):
                    with st.spinner(f"Analyzing with {st.session_state.ai_provider}..."):
                        agent = FantasyAgent(provider=st.session_state.ai_provider)
                        analysis = agent.analyze_waiver(selected_waiver, player_stats, roster, scoring, budget)
                    st.markdown(analysis)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHO TO START
# ════════════════════════════════════════════════════════════════════════════
with tab_start_sit:
    st.subheader("Who to Start")

    if not roster:
        st.info("Sync a league to compare roster players.")
    else:
        roster_names = [p["name"] for p in roster]
        selected_players = st.multiselect(
            "Select 2–4 players to compare", roster_names, max_selections=4,
        )

        if len(selected_players) < 2:
            st.caption("Select at least 2 players.")
        elif st.button("⚖️ Compare"):
            with st.spinner("Fetching stats..."):
                named_stats = {n: get_player_stats(n, _weekly_df, scoring) for n in selected_players}

            ranked = rank_players(named_stats)
            mc_result = simulate_multi_start(named_stats)
            top = ranked[0]
            conf_label = top.get("confidence_label", "")
            conf_pct = top.get("confidence_pct", 0)

            banner_color = "green" if conf_label == "Clear" else ("orange" if conf_label == "Lean" else "gray")
            mc_pct = round(mc_result["win_probs"].get(top["name"], 0) * 100)
            st.markdown(
                f"## :{banner_color}[START: {top['name']}]  "
                f"— {conf_label} · Adj value edge {top.get('margin_to_next', 0):.1f} pts · "
                f"MC win prob {mc_pct}%"
            )

            # Player cards
            cols = st.columns(len(ranked))
            for col, r in zip(cols, ranked):
                name = r["name"]
                is_start = r["rank"] == 1
                lbl_color = "green" if is_start else "red"
                mc_smry = mc_result["summaries"][name]
                win_p = round(mc_result["win_probs"][name] * 100)

                # Matchup
                p_obj = next((p for p in roster if p["name"] == name), {})
                team = p_obj.get("team", "")
                next_g = get_next_game(team, _schedules_df) if team else None
                opp_str = f"vs {next_g['opponent']}" if next_g else "Offseason"
                def_i = get_matchup_info(next_g["opponent"], p_obj.get("position", ""), _defense_df) if next_g else {}

                with col:
                    lbl = "✅ START" if is_start else f"#{r['rank']} BENCH"
                    st.markdown(f"**{name}**")
                    st.markdown(f":{lbl_color}[{lbl}]")
                    st.metric("Adj Value", r["adj_value"])
                    st.metric("Consistency", f"{r['consistency']*100:.0f}%")
                    st.metric("MC Win Prob", f"{win_p}%")
                    st.metric("Floor / Ceiling", f"{mc_smry['floor']} / {mc_smry['ceiling']}")
                    st.caption(f"{opp_str} — {def_i.get('rank_label', '')}")
                    inj = injury_badge(name, _injury_map)
                    if inj:
                        st.error(f"🏥 {inj}")

            # Comparison bar chart
            rows = []
            for name, stats in named_stats.items():
                s = stats.get("season") or {}
                rc = stats.get("recent") or {}
                v = next(r for r in ranked if r["name"] == name)
                rows += [
                    {"Player": name, "Metric": "Season PPG", "Value": s.get("ppg", 0)},
                    {"Player": name, "Metric": "Recent PPG", "Value": rc.get("ppg", 0)},
                    {"Player": name, "Metric": "Adj Value", "Value": v["adj_value"]},
                ]
            fig = px.bar(pd.DataFrame(rows), x="Metric", y="Value", color="Player",
                         barmode="group", title="Head-to-Head Comparison",
                         category_orders={"Metric": ["Season PPG", "Recent PPG", "Adj Value"]})
            st.plotly_chart(fig, use_container_width=True)

            # Raincloud — MC outcome distributions
            rc_samples = {n: simulate_player(named_stats[n]) for n in selected_players}
            st.plotly_chart(
                _raincloud_fig(rc_samples, "Simulated Outcome Distributions (10,000 games)"),
                use_container_width=True,
            )

            if len(selected_players) == 2:
                p1, p2 = selected_players[0], selected_players[1]
                hth = simulate_start_sit(p1, named_stats[p1], p2, named_stats[p2])
                hc1, hc2 = st.columns(2)
                hc1.metric(f"P({p1} wins)", f"{hth['p1_win_prob']*100:.0f}%")
                hc2.metric(f"P({p2} wins)", f"{hth['p2_win_prob']*100:.0f}%")

            with st.expander("Weekly scoring history"):
                fig2 = go.Figure()
                for name, stats in named_stats.items():
                    wp = stats.get("weekly", [])
                    if wp:
                        fig2.add_trace(go.Scatter(x=list(range(1, len(wp)+1)), y=wp,
                                                   mode="lines+markers", name=name))
                fig2.update_layout(xaxis_title="Week", yaxis_title="Fantasy Points",
                                   title="Season Weekly Scores")
                st.plotly_chart(fig2, use_container_width=True)

            with st.expander("🤖 AI Narrative Analysis (optional — 2 players only)"):
                st.caption("Add API key to .env to enable.")
                if len(selected_players) == 2 and st.button("Get AI Analysis", key="ss_ai"):
                    p1, p2 = selected_players[0], selected_players[1]
                    with st.spinner("Getting AI take..."):
                        agent = FantasyAgent(provider=st.session_state.ai_provider)
                        rec = agent.analyze_start_sit(p1, p2, named_stats[p1], named_stats[p2], scoring)
                    st.markdown(rec)
                elif len(selected_players) > 2:
                    st.caption("AI narrative only available for 2-player comparisons.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRADE ANALYZER
# ════════════════════════════════════════════════════════════════════════════
with tab_trade:
    st.subheader("Trade Analyzer")
    roster_names = [p["name"] for p in roster] if roster else []

    giving = st.multiselect("You Give (from your roster)", roster_names)
    receiving_text = st.text_area("You Receive (one name per line)")
    receiving = [n.strip() for n in receiving_text.splitlines() if n.strip()]

    if st.button("🤝 Analyze Trade") and (giving or receiving):
        all_names = giving + receiving
        with st.spinner("Looking up stats..."):
            stats_map = {n: get_player_stats(n, _weekly_df, scoring) for n in all_names}

        giving_stats = {n: stats_map[n] for n in giving}
        receiving_stats = {n: stats_map[n] for n in receiving}

        det_result = score_trade(giving_stats, receiving_stats)
        mc_result = simulate_trade(giving_stats, receiving_stats)

        # Use MC verdict as primary (more nuanced)
        verdict = mc_result["verdict"]
        color = _verdict_color(verdict)
        ratio_pct = round((det_result["ratio"] - 1) * 100, 1)
        sign = "+" if ratio_pct >= 0 else ""
        mc_pct = round(mc_result["p_receive_better"] * 100)

        st.markdown(
            f"## :{color}[{verdict}]  "
            f"— Adj value {sign}{ratio_pct}% · P(receive wins) = {mc_pct}%"
        )

        # Value + MC columns
        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown("**You Give**")
            for name, val in det_result["you_give"].items():
                inj = injury_badge(name, _injury_map)
                inj_str = f" 🏥 {inj}" if inj else ""
                st.metric(f"{name}{inj_str}", f"{val} adj pts/wk")
            st.metric("Total", f"{det_result['give_total']} pts/wk")
            g = mc_result["give"]
            st.caption(f"MC: Median {g['median']} | Floor {g['floor']} | Ceiling {g['ceiling']}")

        with vc2:
            st.markdown("**You Receive**")
            for name, val in det_result["you_receive"].items():
                inj = injury_badge(name, _injury_map)
                inj_str = f" 🏥 {inj}" if inj else ""
                st.metric(f"{name}{inj_str}", f"{val} adj pts/wk")
            st.metric("Total", f"{det_result['recv_total']} pts/wk")
            r = mc_result["receive"]
            st.caption(f"MC: Median {r['median']} | Floor {r['floor']} | Ceiling {r['ceiling']}")

        # Bar chart — individual player values
        rows = []
        for name in giving:
            rows.append({"Side": "Give", "Player": name, "Adj Value": det_result["you_give"].get(name, 0)})
        for name in receiving:
            rows.append({"Side": "Receive", "Player": name, "Adj Value": det_result["you_receive"].get(name, 0)})
        if rows:
            fig = px.bar(pd.DataFrame(rows), x="Player", y="Adj Value", color="Side",
                         color_discrete_map={"Give": "firebrick", "Receive": "seagreen"},
                         title="Trade Value Comparison")
            st.plotly_chart(fig, use_container_width=True)

        # Raincloud — total projected output distributions for each side
        give_total_samples = sum(simulate_player(s) for s in giving_stats.values())
        recv_total_samples = sum(simulate_player(s) for s in receiving_stats.values())
        trade_rc = {"You Give": give_total_samples, "You Receive": recv_total_samples}
        st.plotly_chart(
            _raincloud_fig(trade_rc, "Total Weekly Output Distributions (10,000 simulations)"),
            use_container_width=True,
        )

        # Playoff schedule strength for each player
        all_pos_map = {p["name"]: p.get("position", "") for p in roster}
        all_team_map = {p["name"]: p.get("team", "") for p in roster}

        with st.expander("📅 Playoff Schedule Strength (Weeks 14–16)"):
            psc1, psc2 = st.columns(2)
            for col, side_names, header in [(psc1, giving, "You Give"), (psc2, receiving, "You Receive")]:
                with col:
                    st.markdown(f"**{header}**")
                    for name in side_names:
                        team = all_team_map.get(name, "")
                        pos = all_pos_map.get(name, "WR")
                        pss = get_playoff_schedule_strength(team, _schedules_df, _defense_df, pos)
                        grade = pss["grade"]
                        gc = _grade_color(grade)
                        st.markdown(f"{name}: :{gc}[Grade {grade}] (avg mult {pss['avg_multiplier']:.2f})")
                        for wk in pss["weeks"]:
                            st.caption(f"  Wk {wk['week']}: vs {wk['opponent']} — {wk['matchup']}")

        with st.expander("🤖 AI Narrative Analysis (optional)"):
            st.caption("Add API key to .env to enable.")
            if st.button("Get AI Verdict", key="trade_ai"):
                with st.spinner("Getting AI take..."):
                    agent = FantasyAgent(provider=st.session_state.ai_provider)
                    verdict_text = agent.analyze_trade(giving, receiving, stats_map, scoring)
                st.markdown(verdict_text)


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
        with st.spinner("Analyzing roster..."):
            # Stats for all roster players
            roster_stats = {p["name"]: get_player_stats(p["name"], _weekly_df, scoring) for p in roster}

            # Stats for available players (limit to top 100 by trend + position for speed)
            pos_filter = ["QB", "RB", "WR", "TE"]
            avail_pool = [w for w in waivers if w.get("position") in pos_filter][:100]
            avail_stats = {p["name"]: get_player_stats(p["name"], _weekly_df, scoring) for p in avail_pool}

            analysis = analyze_roster_construction(
                roster, avail_pool, roster_stats, avail_stats, _repl_levels
            )

        pg = analysis["position_grades"]
        weaknesses = analysis["weaknesses"]

        # Positional grade cards
        st.markdown("### Positional Strength")
        grade_cols = st.columns(len(pg))
        for col, (pos, g) in zip(grade_cols, pg.items()):
            gc = _grade_color(g["grade"])
            col.metric(pos, f"Grade {g['grade']}", f"×{g['strength_ratio']:.2f} vs replacement")
            col.caption(f"{g['depth_count']} players · starters avg {g['avg_starter_val']:.1f} adj pts")

        if weaknesses:
            st.warning(f"⚠️ Positional weaknesses: **{', '.join(weaknesses)}** — prioritized in recommendations below")
        else:
            st.success("✅ No critical positional weaknesses")

        st.divider()

        # Need-adjusted waiver targets
        st.markdown("### Need-Adjusted Waiver Targets")
        st.caption("Score = base waiver score + need bonus (up to +20 for filling a weakness)")

        targets = analysis["targeted_adds"]
        if targets:
            target_rows = []
            for t in targets:
                name = t["name"]
                pos = t.get("position", "")
                team = t.get("team", "")
                next_g = get_next_game(team, _schedules_df) if team else None
                opp_str = f"vs {next_g['opponent']}" if next_g else "—"
                inj = injury_badge(name, _injury_map)
                target_rows.append({
                    "Player": name, "Pos": pos, "Team": team,
                    "Need Score": t["need_adjusted_score"],
                    "Base Score": t["base_score"],
                    "Fills Need": "⭐ YES" if t["fills_need"] else "—",
                    "Verdict": t["verdict"],
                    "Adj Value": t["adj_value"],
                    "Next Game": opp_str,
                    "Injury": inj or "",
                })

            df_targets = pd.DataFrame(target_rows)
            st.dataframe(
                df_targets.sort_values("Need Score", ascending=False),
                hide_index=True, use_container_width=True,
            )

            # Bar chart: top 10 need-adjusted
            top10 = df_targets.sort_values("Need Score", ascending=False).head(10)
            fig = px.bar(
                top10, x="Player", y=["Need Score", "Base Score"],
                barmode="overlay", title="Need-Adjusted vs Base Waiver Score",
                labels={"value": "Score", "variable": ""},
                color_discrete_map={"Need Score": "steelblue", "Base Score": "lightgray"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No available players found — sync your league to populate the waiver pool.")
