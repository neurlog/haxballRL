"""Promotion evaluation games (python tests/test_eval_gate.py).

With --promote-eval-games the 70% gate is measured on separate games vs the champion, played
without learning. eval_vs must return exactly one result per game, from the learner's side,
with the learner red in half the games, and the 1v1 defaults must train 75% self-play / 25% PFSP.
"""
import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hax.env import ENT_DIM, N_ACTIONS, SELF_DIM  # noqa: E402
from hax.model import HaxNet  # noqa: E402
from hax.train import build_parser, eval_vs  # noqa: E402

KW = dict(team_size=1, stadium="classic", time_limit_s=5.0, overtime_limit_s=2.0)


def net(seed):
    torch.manual_seed(seed)
    return HaxNet(self_dim=SELF_DIM, ent_dim=ENT_DIM, n_actions=N_ACTIONS, hidden=32, ent_hidden=16).eval()


def test_one_result_per_game():
    res = eval_vs(net(0), net(1), KW, 6, seed=3)
    print("  results:", res)
    assert len(res) == 6 and all(r in (0.0, 0.5, 1.0) for r in res)


def test_results_are_from_the_learners_side():
    a = net(0)
    res = eval_vs(a, a, KW, 8, seed=5)  # same net both sides: results mirror around 0.5 on average
    assert len(res) == 8 and all(r in (0.0, 0.5, 1.0) for r in res)


def test_1v1_defaults():
    spec = importlib.util.spec_from_file_location("t1", os.path.join(ROOT, "1v1-classic", "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = build_parser(mod.DEFAULTS).parse_args([])
    assert (a.frac_champ, a.frac_league, a.frac_lazy) == (0.0, 0.25, 0.0)
    assert a.promote_eval_games == 200 and a.promote_threshold == 0.70 and a.promote_every == 500


if __name__ == "__main__":
    for t in (test_one_result_per_game, test_results_are_from_the_learners_side, test_1v1_defaults):
        t()
        print("PASS", t.__name__)
