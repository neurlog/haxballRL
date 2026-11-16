"""Shared reward settings, so 1v1 and 3v3 train on exactly the same rewards.

Used as `DEFAULTS = dict(..., **REWARDS, ...)` in 1v1-classic/train.py and 3v3-big/train.py.
Team-only terms (pass reward, crowding, team spirit) live in the 3v3 defaults; in 1v1 they do nothing.
"""

REWARDS = dict(
    # +3 per goal scored, -3 per goal conceded, no separate win reward.
    goal_reward=3.0,
    win_reward=0.0,
    # kickoff dawdling: once your own kickoff has waited 5 s, your team loses 0.002 per decision
    kickoff_penalty=0.002,
    kickoff_grace=5.0,
    # shaping. PPO normalises advantages, so only the ratio to the goal reward matters: these are the
    # weights that worked against a +-1 goal (v5: going for the ball within 500 games), x5, sized for a
    # +-5 goal (with goals now +-3 the shaping counts ~1.7x more than that). Without the x5, runs v6/v8
    # moved at random for 2k+ games (random goals drowned the chase).
    #   chase 0.025 -> 0.0005 and touch 0.25 -> 0.015 over the first 3k games, then both fade to zero
    #   5k -> 10k, after which only goals (and ball-to-goal) are left
    #   ball-to-goal 0.004 for the whole run, zero-sum: (yours - the opponent's) / 2. Plain ball-to-goal
    #   paid BOTH players when the ball came out of a corner, and run v4 camped the corners for it
    speed_early=0.025,
    speed_late=0.0005,
    touch_early=0.25,
    touch_late=0.015,
    btg_early=0.004,
    btg_late=0.004,
    keep_btg=True,
    btg_zero_sum=True,
    push_fade_start=5000,
    push_fade_games=5000,
    fade_chase=True,
    # corner stalling: a 50 px square in each corner (drawn see-through red in the viewers only).
    # Once the ball has sat in one for 5 s of play, BOTH players lose 0.01 per decision (0.3/s each,
    # so 10 s of it costs as much as conceding a +-3 goal) until it leaves. Run v5 learned to pin the ball in a
    # corner (26% of play at 53k games) because, with zero-sum rewards, a pin cost nobody anything.
    corner_box=50.0,
    corner_hold=5.0,
    corner_penalty=0.01,
    # off for the first 5k games (still logged), full by 6k: early on a randomly kicked ball rests in
    # a corner all the time and the penalty, paid by both alike, was only noise
    corner_start_games=5000,
    corner_ramp_games=1000,
    # idle ball: if the ball hasn't travelled 40 px in 5 s of play, the player NEAREST it loses a
    # share of idle_penalty per decision (table below) until it does. Run v6 parked both
    # bots in their own goals and left the ball alone (98% of games 0-0 at 40k games). Only the
    # nearest player pays: it's the one that can move it, and the bot defending isn't punished.
    # A touch inside a corner square also resets it, so corner play is judged by the corner rule only.
    idle_penalty=0.5 / 30,  # 0.0167 per decision = 0.5/s: 6 s of the full share costs a +-3 goal
    # share of it by score and distance (rows winning / level / losing, columns nearest / everyone else):
    #   winning: nearest pays half, the rest nothing   (you're ahead: holding is less of a sin)
    #   level:   nearest pays in full, the rest half    (someone has to go and win it)
    #   losing:  everyone pays in full                  (you have to go and get the ball)
    idle_table="0.5,0,1,0.5,1,1",
    idle_dist=40.0,
    idle_hold=5.0,
    # ...but not from the start: early on it swamped everything and the bots never learned to go to
    # the ball. Off for the first 5k games (still logged as idle s/g), then up to full by 6k.
    idle_start_games=5000,
    idle_ramp_games=1000,
)
