"""Train the 3v3 team on the Big stadium. Runs are saved under 3v3-big/runs/<name>.

    python 3v3-big/train.py                                     # from scratch, run "main"
    python 3v3-big/train.py --init-from 1v1-classic/models/main-75kv9.pt   # start from the 1v1 champion
    python 3v3-big/train.py --resume                            # continue

All flags: python 3v3-big/train.py --help
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hax.presets import REWARDS  # noqa: E402
from hax.train import main  # noqa: E402

DEFAULTS = dict(
    REWARDS,  # exactly the 1v1 rewards: goals +-3, shaping, corner and idle rules (hax/presets.py)
    team_size=3,
    stadium="big",  # 3v3 is played on Big
    n_envs=96,  # 6 players per game: ~same samples per update as 1v1 with 192 envs
    time_limit=180.0,  # 3-minute games
    # team play (see hax/env.py): zero-sum pass reward, anti-bunching, team spirit
    pass_reward=0.1,  # x min(distance / 300 px, 1), to the whole passing team; the opponents lose it
    pass_min_dist=70.0,
    pass_backward=True,  # backward passes count too
    crowd_penalty=0.01,  # per decision, per teammate closer than crowd_dist
    crowd_dist=40.0,  # players are 30 px wide: only near-touching counts as crowding
    team_spirit_early=0.1,  # RLGym: start low, or followers get paid for trailing the chaser
    team_spirit_late=0.8,
    # opponents, as 1v1: 75% self-play, 25% PFSP over the league, no lazy opponents; promotion by
    # 200 separate evaluation games vs the current champion every 500 games (>= 70%)
    frac_champ=0.0,
    frac_league=0.25,
    frac_lazy=0.0,
    promote_eval_games=200,
)

if __name__ == "__main__":
    main(defaults=DEFAULTS, mode_dir=HERE)
