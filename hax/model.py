"""Entity policy network. The same weights work for 1v1, 2v2 and 3v3.

obs = [self (SELF_DIM)] + [teammate rows (ENT_DIM each)] + [opponent rows (ENT_DIM each)].
Teammates and opponents go through their own shared encoders and are pooled with
mean + max, so the number of players can change. The teammate encoder's last layer
starts at zero, so a 1v1 policy behaves exactly the same on day one of 3v3.

Kept to Linear + ReLU + mean/max so it can be re-implemented in a few lines of
JavaScript for the node-haxball bot (headless-bot/policy.js).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


def mlp(sizes, last_act=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class HaxNet(nn.Module):
    def __init__(self, self_dim: int, ent_dim: int, n_actions: int = 18, hidden: int = 256, ent_hidden: int = 128):
        super().__init__()
        self.self_dim, self.ent_dim, self.n_actions = self_dim, ent_dim, n_actions
        self.hidden, self.ent_hidden = hidden, ent_hidden
        self.self_enc = mlp([self_dim, hidden, hidden])
        self.mate_enc = mlp([ent_dim, ent_hidden, ent_hidden])
        self.opp_enc = mlp([ent_dim, ent_hidden, ent_hidden])
        nn.init.zeros_(self.mate_enc[2].weight)
        nn.init.zeros_(self.mate_enc[2].bias)
        self.trunk = mlp([hidden + 4 * ent_hidden, hidden, hidden])
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def kwargs(self):
        return dict(self_dim=self.self_dim, ent_dim=self.ent_dim, n_actions=self.n_actions,
                    hidden=self.hidden, ent_hidden=self.ent_hidden)

    def _pool(self, enc, x):  # x (B, k, ent_dim)
        if x.shape[1] == 0:
            z = x.new_zeros(x.shape[0], self.ent_hidden)
            return torch.cat([z, z], -1)
        h = enc(x)
        return torch.cat([h.mean(1), h.max(1).values], -1)

    def forward(self, obs, n_mates: int, n_opps: int):
        b = obs.shape[0]
        s = obs[:, : self.self_dim]
        ents = obs[:, self.self_dim:].reshape(b, n_mates + n_opps, self.ent_dim)
        h = torch.cat([self.self_enc(s), self._pool(self.mate_enc, ents[:, :n_mates]),
                       self._pool(self.opp_enc, ents[:, n_mates:])], -1)
        h = self.trunk(h)
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, n_mates, n_opps, deterministic=False):
        logits, v = self(obs, n_mates, n_opps)
        if deterministic:
            a = logits.argmax(-1)
            return a, torch.zeros_like(v), v
        d = Categorical(logits=logits)
        a = d.sample()
        return a, d.log_prob(a), v

    def evaluate(self, obs, actions, n_mates, n_opps):
        logits, v = self(obs, n_mates, n_opps)
        d = Categorical(logits=logits)
        return d.log_prob(actions), d.entropy(), v


def load_policy(path, map_location="cpu"):
    """Load a policy from a training checkpoint or a bare league state dict."""
    c = torch.load(path, map_location=map_location, weights_only=False)
    sd = c["model"] if isinstance(c, dict) and "model" in c else c
    kw = c.get("model_kwargs") if isinstance(c, dict) and "model_kwargs" in c else None
    if kw is None:
        kw = dict(self_dim=sd["self_enc.0.weight"].shape[1], ent_dim=sd["opp_enc.0.weight"].shape[1],
                  n_actions=sd["pi.weight"].shape[0], hidden=sd["self_enc.0.weight"].shape[0],
                  ent_hidden=sd["opp_enc.0.weight"].shape[0])
    net = HaxNet(**kw)
    net.load_state_dict(sd)
    net.eval()
    return net
