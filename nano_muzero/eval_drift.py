"""Drift curves: how fast does the imagined trajectory decay with depth? (M1.d eval)

For sampled replay positions, unroll the trained model along the ACTUAL stored actions and
measure |v^k - z_{t+k}| at each depth k. The whole argument for unrolled training in one
table: the K=5-trained model beats the K=1-trained model at depth, because only it was ever
graded on composed dynamics. Depths beyond the trained K show compounding drift (scene 1.9,
failure mode 1).

  python -m nano_muzero.eval_drift          # prints the table, writes data/eval/drift_curves.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nano_muzero import baseline
from nano_muzero.train import load_offline_net

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "eval" / "drift_curves.json"


def drift_table(models: dict, replay_path, n_positions: int = 50, max_k: int = 8, seed: int = 0):
    """abs value-error per unroll depth, teacher-forced along stored actions.
    Only real rows count (a game that ended k-3 steps ago has no depth-k truth)."""
    data = baseline.validate_replay(replay_path)
    game_id = data["game_id"][:]
    ends = np.searchsorted(game_id, game_id, side="right")
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(game_id), size=n_positions, replace=False)

    errs = {name: {k: [] for k in range(1, max_k + 1)} for name in models}
    for name, net in models.items():
        for j in rows:
            depth = min(max_k, ends[j] - 1 - j)  # how many stored actions follow row j
            if depth < 1:
                continue
            with torch.no_grad():
                obs = torch.from_numpy(data["obs"][j]).unsqueeze(0)
                acts = torch.from_numpy(data["action"][j : j + depth].astype(np.int64)).unsqueeze(0)
                _, values, _ = net.unroll(obs, acts)
            for k in range(1, depth + 1):
                errs[name][k].append(abs(float(values[k][0]) - float(data["z"][j + k])))
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--positions", type=int, default=50)
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--replay", type=Path, default=baseline.REPLAY_PATH)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    models = {"K=1 trained": load_offline_net(1), "K=5 trained": load_offline_net(5)}
    errs = drift_table(models, args.replay, args.positions, args.max_k, args.seed)

    names = list(models)
    print(f"unroll value drift |v^k - z|, {args.positions} sampled positions, teacher-forced")
    print("| depth k | n | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    export = {"positions": args.positions, "seed": args.seed, "curves": {n: {} for n in names}}
    for k in range(1, args.max_k + 1):
        n = len(errs[names[0]][k])
        cells = []
        for name in names:
            mean = float(np.mean(errs[name][k])) if errs[name][k] else float("nan")
            export["curves"][name][k] = {"mean_abs_err": mean, "n": n}
            cells.append(f"{mean:.3f}" if n else "-")
        print(f"| {k} | {n} | " + " | ".join(cells) + " |")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(export, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
