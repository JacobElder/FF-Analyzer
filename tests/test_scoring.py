import pandas as pd
import pytest

from src.scoring import compute_fantasy_points, effective_scoring


def _row(**kw):
    return pd.DataFrame([{"position": "WR", **kw}])


def test_receiving_line_half_ppr():
    pts = compute_fantasy_points(_row(receptions=6, receiving_yards=80, receiving_tds=1), effective_scoring({}))
    assert pts.iloc[0] == pytest.approx(6 * 0.5 + 8 + 6)


def test_six_point_passing_td_league():
    df = _row(position="QB", passing_yards=250, passing_tds=2, passing_interceptions=1)
    four = compute_fantasy_points(df, effective_scoring({}))
    six = compute_fantasy_points(df, effective_scoring({"pass_td": 6}))
    assert six.iloc[0] - four.iloc[0] == pytest.approx(4.0)


def test_te_premium_only_applies_to_tes():
    sc = effective_scoring({"rec": 1, "bonus_rec_te": 0.5})
    te = compute_fantasy_points(_row(position="TE", receptions=4), sc)
    wr = compute_fantasy_points(_row(position="WR", receptions=4), sc)
    assert te.iloc[0] == pytest.approx(6.0) and wr.iloc[0] == pytest.approx(4.0)


def test_fumbles_lost_from_all_sources():
    df = _row(rushing_fumbles_lost=1, receiving_fumbles_lost=1, sack_fumbles_lost=0)
    assert compute_fantasy_points(df, effective_scoring({})).iloc[0] == pytest.approx(-4.0)


def test_league_values_override_defaults_and_bad_values_ignored():
    sc = effective_scoring({"rec": "1", "pass_td": None})
    assert sc["rec"] == 1.0 and sc["pass_td"] == 4.0
