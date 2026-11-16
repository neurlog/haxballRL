"""Train the 1v1 Classic agent. Runs are saved under 1v1-classic/runs/<name>.

    python 1v1-classic/train.py                         # fresh run "main"
    python 1v1-classic/train.py --resume                # continue it
    python 1v1-classic/train.py --name exp2 --total-games 10000

All flags: python 1v1-classic/train.py --help
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hax.presets import REWARDS  # noqa: E402
from hax.train import main  # noqa: E402

DEFAULTS = dict(
    team_size=1,
    stadium="classic",
    time_limit=120.0,  # 2-minute games (golden goal still capped at 60 s)
    **REWARDS,  # goals +-3, shaping, corner and idle rules: see hax/presets.py
    # opponents: 75% self-play, 25% PFSP over the league, no lazy opponents. Promotion is decided
    # by 200 separate evaluation games vs the current champion every 500 games (no learning in
    # them): >= 70% (draw = half) and >= 50% average vs the league -> promoted.
    frac_champ=0.0,
    frac_league=0.25,
    frac_lazy=0.0,
    promote_eval_games=200,
)

if __name__ == "__main__":
    main(defaults=DEFAULTS, mode_dir=HERE)
