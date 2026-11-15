"""Reward-weight schedule (python tests/test_schedule.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hax.train import build_parser, reward_weights  # noqa: E402


def close(a, b):
    return all(abs(x - y) < 1e-9 for x, y in zip(a, b))


def test_schedule():
    args = build_parser().parse_args([])
    rows = {g: reward_weights(args, g) for g in (0, 3000, 10000, 15000, 20000, 30000)}
    for g, (sp, to, bt, _, stage, fade) in rows.items():
        print(f"  {g:6d} games: speed {sp:.4f}  touch {to:.4f}  ball-to-goal {bt:.4f}  (stage {stage:.2f}, fade {fade:.2f})")
    assert close(rows[0][:3], (0.02, 0.2, 0.01))                                    # early values
    assert close(rows[3000][:3], (0.002, 0.02, 0.01)) and close(rows[10000][:3], (0.002, 0.02, 0.01))  # late, no fade yet
    assert abs(rows[15000][1] - 0.01) < 1e-9 and abs(rows[15000][2] - 0.005) < 1e-9  # halfway through the fade
    assert rows[20000][1] == rows[20000][2] == 0.0 == rows[30000][1] == rows[30000][2]  # faded out
    assert abs(rows[30000][0] - 0.002) < 1e-9                       # velocity-to-ball stays


if __name__ == "__main__":
    test_schedule()
    print("PASS test_schedule")
