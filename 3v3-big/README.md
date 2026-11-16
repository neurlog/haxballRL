# 3v3 Big

The 3v3 team on the **Big** stadium (where 3v3 is normally played): a 1100 x 480 pitch,
goals 160 wide, spawn distance 400. Same trainer, league and game rules as 1v1. The
differences are three players per team, the bigger map and rewards for team play.

Observations are scaled by stadium size, so positions stay in about [-1, 1] on Big as
they do on Classic. Starting from the 1v1 Classic champion therefore still makes sense,
but it has to adapt to the bigger pitch. Runs are saved to
`3v3-big/runs/<name>/` (not uploaded to git).

## Trained model

`models/champion_5k.pt` is the champion after 5,000 games from scratch (gen 4). It's in git
so anyone can watch it or keep training from it. Measured over 90 s of self-play:
teammates average 183 px apart, never all three within 80 px, and a teammate stays 100+ px
behind the ball 67% of the time. An earlier version of the rewards had them clumped
40 px apart 94% of the time. It beats its own gen 1 in 32 of 32 games.

```bash
python 3v3-big/play.py --red 3v3-big/models/champion_5k.pt --blue 3v3-big/models/champion_5k.pt
python 3v3-big/play.py --red human --blue 3v3-big/models/champion_5k.pt
```

## Train

```bash
python 3v3-big/train.py --total-games 10000                       # from scratch
python 3v3-big/train.py --total-games 10000 \
    --init-from 1v1-classic/models/main-75kv9.pt                    # start from the 1v1 champion
python 3v3-big/train.py --resume --total-games 20000              # continue a run
```

3v3 runs about 3x slower than 1v1 per game: six players instead of two, so roughly 15
minutes per 1000 games.

## Team-play rewards (on top of the 1v1 rewards)

| term | what it does | why it can't be farmed |
|---|---|---|
| pass (`--pass-reward 0.1`) | One teammate's touch, then another teammate's touch, with no opponent touch in between. Pays 0.1 x min(distance / 300 px, 1) to the whole team; the opponents lose the same. | Only counts if the ball travelled at least 100 px (`--pass-min-dist`) **and** moved towards the opponent goal. 1 s cooldown per team. Corner ping-pong is too short, and back-and-forth always has a backwards leg. |
| chase and touch | The velocity-towards-the-ball and touch rewards go only to the player on each team nearest the ball. | Stops all three swarming the ball: a scrum of teammates earns one player's reward, not three. |
| crowding (`--crowd-penalty 0.01`) | Each pair of teammates closer than 40 px (`--crowd-dist`; players are 30 px wide, so only near-touching) costs both up to 0.01 per decision. | It only penalises bunching. It never rewards distance, because HaxballGym found that rewarding spacing made bots spread out and stop playing. |
| team spirit (`--team-spirit-early 0.1 --team-spirit-late 0.8`) | Individual rewards (chase, touch, crowding) are blended with the team average, rising over the first 3000 games (as in RLGym). | Players learn to value a teammate's touch like their own. |

There is no hand-coded goalkeeper. Roles such as one player staying back are left to emerge
from the team rewards: goals and wins are shared, only one player is paid to chase, and
bunching is penalised.

`tests/test_team_rewards.py` checks the guards: a forward pass pays the team and costs the
opponents; backward, short and intercepted passes pay nothing; 60 teammate-to-teammate
touches of corner ping-pong pay nothing; and the bunching and nearest-player rules work.

## Watch / play

```bash
python 3v3-big/play.py --red champion --blue gen1
python 3v3-big/play.py --red human --blue champion    # you are red player 1, with AI teammates
```
