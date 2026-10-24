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


# ── added: kickers, labels, invariants ───────────────────────────────────────

import math  # noqa: E402

from src.scoring import DEFAULT_SCORING, scoring_key, scoring_label  # noqa: E402


def test_long_field_goal_split_and_single_key_leagues():
    df = _row(position="K", fg_made_50_59=1, fg_made_60_=1)
    split = compute_fantasy_points(df, effective_scoring({"fgm_50_59": 5, "fgm_60p": 6}))
    single = compute_fantasy_points(df, effective_scoring({"fgm_50p": 4}))
    assert split.iloc[0] == pytest.approx(11.0) and single.iloc[0] == pytest.approx(8.0)


def test_negative_plays():
    df = _row(position="QB", passing_interceptions=2)
    assert compute_fantasy_points(df, effective_scoring({})).iloc[0] == pytest.approx(-4.0)


def test_scoring_is_linear_in_stats():
    sc = effective_scoring({"rec": 1})
    a = dict(receptions=3, receiving_yards=40, rushing_tds=1)
    b = dict(receptions=5, receiving_yards=12, receiving_tds=1)
    both = {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}
    total = compute_fantasy_points(_row(**both), sc).iloc[0]
    assert total == pytest.approx(compute_fantasy_points(_row(**a), sc).iloc[0]
                                  + compute_fantasy_points(_row(**b), sc).iloc[0])


def test_missing_position_column_skips_position_bonuses():
    df = pd.DataFrame([{"receptions": 4}])
    assert compute_fantasy_points(df, effective_scoring({"rec": 1, "bonus_rec_te": 1})).iloc[0] == 4.0


def test_nan_and_inf_league_values_are_ignored():
    sc = effective_scoring({"rec": "nan", "pass_td": float("inf")})
    assert sc["rec"] == DEFAULT_SCORING["rec"] and sc["pass_td"] == DEFAULT_SCORING["pass_td"]
    assert all(math.isfinite(v) for v in sc.values())


@pytest.mark.parametrize("rec,label", [(0, "Standard"), (0.5, "Half-PPR"), (1, "PPR"), (1.5, "PPR")])
def test_scoring_label(rec, label):
    assert scoring_label({"rec": rec}) == label


def test_scoring_key_is_order_independent():
    assert scoring_key({"a": 1, "b": 2}) == scoring_key({"b": 2.0, "a": 1})
