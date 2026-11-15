"""Watch or play 3v3 Classic games. See `python 3v3-classic/play.py --help`.

    python 3v3-classic/play.py --red champion --blue gen1          # two champions, 3 players each
    python 3v3-classic/play.py --red human --blue champion         # you control red player 1, AI teammates
    python 3v3-classic/play.py --red latest --blue gen1 --fast --games 64   # quick eval, no window
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hax.play import main  # noqa: E402

if __name__ == "__main__":
    main(mode_dir=HERE, team_size=3, stadium="classic")
