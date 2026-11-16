"""League eviction (python tests/test_league_evict.py).

Run v6 crashed at 49k games: the league hit its 60-member cap, evicted the oldest snapshot,
and a game that had already drawn that snapshot as its opponent tried to load it.
refresh_opponents must re-pick those games' opponents from members that still exist.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hax.env import ENT_DIM, N_ACTIONS, SELF_DIM  # noqa: E402
from hax.league import League  # noqa: E402
from hax.model import HaxNet  # noqa: E402
from hax.train import refresh_opponents  # noqa: E402


def test_evicted_opponent_is_replaced():
    ctor = lambda: HaxNet(self_dim=SELF_DIM, ent_dim=ENT_DIM, n_actions=N_ACTIONS, hidden=16, ent_hidden=8)  # noqa: E731
    with tempfile.TemporaryDirectory() as d:
        lg = League(d, ctor, max_size=3)
        for g in range(3):
            lg.add(ctor().state_dict(), "snapshot", g)
        env_opp = ["m0000", "m0001", None, "m0002"]
        assert not refresh_opponents(env_opp, lg)  # nothing evicted yet
        lg.add(ctor().state_dict(), "snapshot", 3)  # 4th member -> m0000 evicted
        assert "m0000" not in {m["id"] for m in lg.members}
        assert refresh_opponents(env_opp, lg)
        assert env_opp[0] in {m["id"] for m in lg.members} and env_opp[1:] == ["m0001", None, "m0002"]
        for mid in env_opp:
            if mid is not None:
                lg.model(mid)  # loads without a KeyError


if __name__ == "__main__":
    test_evicted_opponent_is_replaced()
    print("PASS test_evicted_opponent_is_replaced")
