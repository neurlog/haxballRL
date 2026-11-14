"""Export a trained checkpoint to policy.json for the Node bot (no torch needed in Node).

    python headless-bot/export_policy.py 1v1-classic/runs/main/league/champion.pt
    python headless-bot/export_policy.py 1v1-classic/runs/main/latest.pt --out headless-bot/policy.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hax.env import ENT_DIM, SELF_DIM  # noqa: E402
from hax.model import load_policy  # noqa: E402


def export(ckpt: str) -> dict:
    net = load_policy(ckpt)
    sd = net.state_dict()

    def layers(prefix, idxs):
        return [{"w": sd[f"{prefix}.{i}.weight"].tolist(), "b": sd[f"{prefix}.{i}.bias"].tolist()} for i in idxs]

    return dict(
        name=os.path.relpath(os.path.abspath(ckpt), os.path.dirname(HERE)),
        self_dim=SELF_DIM, ent_dim=ENT_DIM, ent_hidden=net.ent_hidden, hidden=net.hidden,
        self_enc=layers("self_enc", (0, 2)), mate_enc=layers("mate_enc", (0, 2)), opp_enc=layers("opp_enc", (0, 2)),
        trunk=layers("trunk", (0, 2)), pi={"w": sd["pi.weight"].tolist(), "b": sd["pi.bias"].tolist()},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=os.path.join(HERE, "policy.json"))
    args = ap.parse_args()
    pol = export(args.ckpt)
    with open(args.out, "w") as f:
        json.dump(pol, f)
    print(f"wrote {args.out} from {pol['name']}")


if __name__ == "__main__":
    main()
