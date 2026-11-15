# 3v3 Classic

The 3v3 team on the Classic map. Same trainer, league and game rules as 1v1. The
differences are three players per team and rewards for team play. Runs are saved to
`3v3-classic/runs/<name>/` (not uploaded to git).

## Train

```bash
python 3v3-classic/train.py --total-games 10000                       # from scratch
python 3v3-classic/train.py --total-games 10000 \
    --init-from 1v1-classic/models/champion_50k.pt                    # start from the 1v1 champion
python 3v3-classic/train.py --resume --total-games 20000              # continue a run
```

3v3 runs about 3x slower than 1v1 per game: six players instead of two, so roughly 15
minutes per 1000 games.

## Team-play rewards (on top of the 1v1 rewards)

| term | what it does | why it can't be farmed |
|---|---|---|
| pass (`--pass-reward 0.1`) | One teammate's touch, then another teammate's touch, with no opponent touch in between. Pays 0.1 x min(distance / 300 px, 1) to the whole team; the opponents lose the same. | Only counts if the ball travelled at least 100 px (`--pass-min-dist`) **and** moved towards the opponent goal. 1 s cooldown per team. Corner ping-pong is too short, and back-and-forth always has a backwards leg. |
| chase | Velocity-towards-the-ball reward goes only to the player on each team nearest the ball. | Stops all three swarming the ball. |
| crowding (`--crowd-penalty 0.001`) | Each pair of teammates closer than 80 px (`--crowd-dist`) costs both up to 0.001 per decision. | It only penalises bunching. It never rewards distance, because HaxballGym found that rewarding spacing made bots spread out and stop playing. |
| team spirit (`--team-spirit-early 0.3 --team-spirit-late 0.8`) | Individual rewards (chase, touch, crowding) are blended with the team average, rising over the first 3000 games (as in RLGym). | Players learn to value a teammate's touch like their own. |

There is no hand-coded goalkeeper. Roles such as one player staying back are left to emerge
from the team rewards: goals and wins are shared, only one player is paid to chase, and
bunching is penalised.

`tests/test_team_rewards.py` checks the guards: a forward pass pays the team and costs the
opponents; backward, short and intercepted passes pay nothing; 60 teammate-to-teammate
touches of corner ping-pong pay nothing; and the bunching and nearest-player rules work.

## Watch / play

```bash
python 3v3-classic/play.py --red champion --blue gen1
python 3v3-classic/play.py --red human --blue champion    # you are red player 1, with AI teammates
```
