"""Train the 3v3 team on the Big stadium. Runs are saved under 3v3-big/runs/<name>.

    python 3v3-big/train.py                                     # from scratch, run "main"
    python 3v3-big/train.py --init-from 1v1-classic/models/champion_50k.pt   # start from the 1v1 champion
    python 3v3-big/train.py --resume                            # continue

All flags: python 3v3-big/train.py --help
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hax.train import main  # noqa: E402

DEFAULTS = dict(
    team_size=3,
    stadium="big",  # 3v3 is played on Big
    n_envs=96,  # 6 players per game: ~same samples per update as 1v1 with 192 envs
    # team play (see hax/env.py): zero-sum forward-pass reward, anti-bunching, team spirit
    pass_reward=0.1,
    pass_min_dist=100.0,
    crowd_penalty=0.01,  # level with the early chase reward, so bunching is actually felt
    crowd_dist=40.0,  # players are 30 px wide: only near-touching counts as crowding
    team_spirit_early=0.1,  # RLGym: start low, or followers get paid for trailing the chaser
    team_spirit_late=0.8,
)

if __name__ == "__main__":
    main(defaults=DEFAULTS, mode_dir=HERE)
