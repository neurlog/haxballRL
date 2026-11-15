"""AlphaStar-style league for the main agent.

* champion  - frozen copy of the best agent so far. The learner is promoted when it
              beats the current champion >= 75% (checked every N games).
* members   - the hall of fame: every retired champion and a learner snapshot every N games.
* PFSP      - opponents are drawn from the members the learner does worst against:
              with prob 1-explore from the `topk` hardest, weighted by (1 - winrate)^p;
              with prob `explore` uniformly, so win rates of "solved" members stay fresh
              and forgetting gets noticed.

Results are 1 win / 0.5 draw / 0 loss, over the last `window` games per member, with a
Beta(1,1) prior (an unseen member counts as 0.5).
"""

from __future__ import annotations

import collections
import json
import os
import random

import torch

class League:
    def __init__(self, run_dir, model_ctor, topk=10, explore=0.1, power=2.0, max_size=60, window=60, cache_size=16):
        self.dir = os.path.join(run_dir, "league")
        os.makedirs(self.dir, exist_ok=True)
        self.model_ctor = model_ctor
        self.topk, self.explore, self.power = topk, explore, power
        self.max_size, self.window, self.cache_size = max_size, window, cache_size
        self.members = []
        self.states = {}
        self.cache = collections.OrderedDict()
        self.next_id = 0
        self.champion_state = None
        self.champion_gen = 0
        self.champion_since_games = 0
        self._champion_model = None

    # ------------------------------------------------------------------ members
    def _add_meta(self, mid, kind, gen, games, results=()):
        self.members.append(dict(id=mid, kind=kind, gen=gen, games_at=games,
                                 results=collections.deque(results, maxlen=self.window)))

    def add(self, state, kind, games, gen=None):
        mid = f"m{self.next_id:04d}"
        self.next_id += 1
        self.states[mid] = state
        torch.save(state, os.path.join(self.dir, f"{mid}.pt"))
        self._add_meta(mid, kind, gen, games)
        self._evict()
        return mid

    def _evict(self):
        while len(self.members) > self.max_size:
            victims = [m for m in self.members if m["kind"] == "snapshot"] or \
                      [m for m in self.members if m["kind"] == "champion"]
            if not victims:
                return
            v = victims[0]
            self.members.remove(v)
            self.states.pop(v["id"], None)
            self.cache.pop(v["id"], None)
            try:
                os.remove(os.path.join(self.dir, f"{v['id']}.pt"))
            except OSError:
                pass

    def get(self, mid):
        return next((m for m in self.members if m["id"] == mid), None)

    def winrate(self, m):
        r = m["results"]
        return (sum(r) + 1.0) / (len(r) + 2.0)

    def record(self, mid, result):
        m = self.get(mid)
        if m is not None:
            m["results"].append(result)

    def hardest(self, k=None):
        return sorted(self.members, key=self.winrate)[: (k or self.topk)]

    def sample(self, rng=random):
        """A member id, or None while the league has no model members yet."""
        if not self.members:
            return None
        if rng.random() < self.explore:
            return rng.choice(self.members)["id"]
        cands = self.hardest()
        w = [(1.0 - self.winrate(m)) ** self.power + 1e-6 for m in cands]
        return rng.choices(cands, weights=w, k=1)[0]["id"]

    def league_winrate(self, min_games=5):
        """Average win rate vs league members (past champions / snapshots)."""
        xs = [self.winrate(m) for m in self.members if len(m["results"]) >= min_games]
        return sum(xs) / len(xs) if xs else None

    def model(self, mid):
        if mid in self.cache:
            self.cache.move_to_end(mid)
            return self.cache[mid]
        net = self.model_ctor()
        net.load_state_dict(self.states[mid])
        net.eval()
        self.cache[mid] = net
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return net

    @staticmethod
    def name(m):
        return f"champ{m['gen']}" if m["kind"] == "champion" else f"snap@{m['games_at']}"

    # ------------------------------------------------------------------ champion
    def set_champion(self, state, gen, games):
        self.champion_state = state
        self.champion_gen = gen
        self.champion_since_games = games
        self._champion_model = None
        torch.save(state, os.path.join(self.dir, "champion.pt"))
        torch.save(state, os.path.join(self.dir, f"champion_gen{gen:03d}.pt"))

    def champion_model(self):
        if self._champion_model is None:
            net = self.model_ctor()
            net.load_state_dict(self.champion_state)
            net.eval()
            self._champion_model = net
        return self._champion_model

    def promote(self, learner_state, games):
        """Retire the current champion into the league and crown the learner."""
        self.add(self.champion_state, "champion", games, gen=self.champion_gen)
        self.set_champion(learner_state, self.champion_gen + 1, games)

    # ------------------------------------------------------------------ persistence
    def save(self):
        meta = dict(
            next_id=self.next_id, champion_gen=self.champion_gen, champion_since_games=self.champion_since_games,
            members=[dict(id=m["id"], kind=m["kind"], gen=m["gen"], games_at=m["games_at"],
                          results=list(m["results"])) for m in self.members],
        )
        tmp = os.path.join(self.dir, "league.json.tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, os.path.join(self.dir, "league.json"))

    def load(self):
        path = os.path.join(self.dir, "league.json")
        if not os.path.exists(path):
            return False
        with open(path) as f:
            meta = json.load(f)
        self.members, self.states, self.cache = [], {}, collections.OrderedDict()
        for m in meta["members"]:
            p = os.path.join(self.dir, f"{m['id']}.pt")
            if m["kind"] == "bot" or not os.path.exists(p):  # skip the old scripted-bot member
                continue
            self.states[m["id"]] = torch.load(p, map_location="cpu", weights_only=True)
            self._add_meta(m["id"], m["kind"], m["gen"], m["games_at"], m["results"])
        self.next_id = meta["next_id"]
        self.champion_gen = meta["champion_gen"]
        self.champion_since_games = meta.get("champion_since_games", 0)
        self.champion_state = torch.load(os.path.join(self.dir, "champion.pt"), map_location="cpu", weights_only=True)
        self._champion_model = None
        return True

    def summary(self, k=5):
        return " ".join(f"{self.name(m)}:{self.winrate(m):.2f}({len(m['results'])})" for m in self.hardest(k))
