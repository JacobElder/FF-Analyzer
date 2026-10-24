"""Shared fixtures for the waiver/FAAB engine tests: a tiny league with flat projections."""

from src.valuation import ValuationContext

WEEKS = [6, 7, 8]
POSITIONS = ["QB", "RB", "WR", "TE", "BN", "BN"]


class NoHistory:
    """StatsIndex stand-in: no past production, so values come from projections only."""
    def get_player_stats(self, p):
        return {"found": False, "season": {"games": 0, "ppg": 0}, "recent": {"ppg": None}}


class FlatProjections:
    available = True

    def __init__(self, pts):
        self.pts = pts

    def get(self, pid, week):
        return self.pts.get(pid)


def _ctx(pts, injuries=None, byes=None):
    teams = {"T1", "T2", "T3"}
    opponents = {t: {w: "XX" for w in WEEKS if (t, w) not in (byes or set())} for t in teams}
    inj = {"by_id": injuries or {}, "by_name": {}}
    return ValuationContext(NoHistory(), FlatProjections(pts), opponents, {}, inj, WEEKS)


def _p(pid, pos, team="T1", slot="BN"):
    return {"name": pid, "player_id": pid, "position": pos, "team": team, "slot": slot}


ROSTER = [_p("qb", "QB", slot="QB"), _p("rb", "RB", slot="RB"), _p("wr", "WR", slot="WR"),
          _p("te", "TE", slot="TE"), _p("wr2", "WR"), _p("rb2", "RB")]
PTS = {"qb": 20, "rb": 15, "wr": 14, "te": 8, "wr2": 6, "rb2": 3}


def _tx(week, roster_id, bid, status="complete", adds=("p1",), note=None, type_="waiver"):
    return {"week": week, "type": type_, "status": status, "bid": bid, "roster_id": roster_id,
            "adds": list(adds), "note": note}


def _rival(interested=True, est=20, budget=100, pos=3, aggression=1.0):
    return {"interested": interested, "est_bid": est, "budget": budget, "waiver_position": pos,
            "aggression": aggression}
