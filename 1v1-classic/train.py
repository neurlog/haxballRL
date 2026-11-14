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

from hax.train import main  # noqa: E402

DEFAULTS = dict(team_size=1, stadium="classic")

if __name__ == "__main__":
    main(defaults=DEFAULTS, mode_dir=HERE)
