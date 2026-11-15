"""Watch or play 1v1 Classic games. See `python 1v1-classic/play.py --help`.

    python 1v1-classic/play.py --red gen2 --blue gen1           # two champions
    python 1v1-classic/play.py --red human --blue champion      # you vs the champion
    python 1v1-classic/play.py --red latest --blue gen1 --fast --games 64  # quick eval, no window
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hax.play import main  # noqa: E402

if __name__ == "__main__":
    main(mode_dir=HERE, team_size=1, stadium="classic")
