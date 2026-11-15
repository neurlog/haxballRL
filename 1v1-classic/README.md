# 1v1 Classic

All 1v1 work on the Classic map lives here. Runs are saved to `1v1-classic/runs/<name>/`:

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
