# 1v1 Classic

All 1v1 work on the Classic map lives here.

`models/champion_50k.pt` is the trained champion after 50,000 games (gen 12). It's in
git so anyone who clones the repo can play it, use it in the real-game bot
(`headless-bot/`), or start 3v3 from it:

```bash
python 1v1-classic/play.py --red human --blue 1v1-classic/models/champion_50k.pt
python headless-bot/export_policy.py 1v1-classic/models/champion_50k.pt
```

Training runs are saved to `1v1-classic/runs/<name>/` (not uploaded to git):

```
runs/<name>/
  latest.pt              learner, saved every 25 updates and on Ctrl+C (resume with --resume)
  ckpt_0001000.pt ...    learner every 1000 games
  log.csv                training metrics (also charted in the viewer window)
  config.json            the flags the run started with
  league/
    champion.pt          the current champion
    champion_gen003.pt   every champion that ever held the title
    m0000.pt ...         league members (retired champions, snapshots every 1000 games)
    league.json          members and the learner's recent results against each
```

## Train

```bash
python 1v1-classic/train.py --total-games 10000           # fresh run called "main"
python 1v1-classic/train.py --resume --total-games 20000  # continue it
python 1v1-classic/train.py --name exp2                   # a separate run
```

## Watch / play

```bash
python 1v1-classic/play.py --red gen2 --blue gen1              # champion gen 2 vs gen 1
python 1v1-classic/play.py --red champion --blue latest        # champion vs the latest learner
python 1v1-classic/play.py --red human --blue champion         # you vs the champion
python 1v1-classic/play.py --red gen3 --blue gen1 --fast --games 64   # 64 games, no window
```

Add `--run exp2` to read a different run.
