"""The value-equivalence probe (scene 1.4's lab, reproduced locally): what actually lives
inside the latent?

A linear probe is trained to decode board occupancy (mine / theirs / empty per cell, from
the to-play view) out of frozen latents s^0 = h(obs). Compared against it: how well f's
value head reads the SAME latents. The finding the lab narrates: whatever board information
survives in the latent survives by accident; the value information survives by construction
-- and the --recon on checkpoint (trained WITH a reconstruction loss) makes occupancy more
decodable without making the value better.

  python -m nano_muzero.probe        # table + data/eval/probe_latents.npz for the browser lab
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nano_muzero import baseline
from nano_muzero.train import load_offline_net

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "eval" / "probe_latents.npz"


def occupancy_labels(obs: np.ndarray) -> np.ndarray:
    """(N,2,3,3) planes -> (N,9) classes: 0 empty, 1 mine, 2 theirs (to-play view)."""
    mine = obs[:, 0].reshape(len(obs), 9)
    theirs = obs[:, 1].reshape(len(obs), 9)
    return (mine + 2 * theirs).astype(np.int64)


def fit_probe(latents: torch.Tensor, labels: torch.Tensor, gen, steps: int = 400):
    """One linear layer latent -> 9 cells x 3 classes; full-batch Adam; returns val accuracy
    per cell. This is deliberately the weakest possible decoder: if IT can read the board,
    the board is linearly present."""
    n = len(latents)
    perm = torch.randperm(n, generator=gen)
    split = int(0.8 * n)
    tr, va = perm[:split], perm[split:]
    probe = torch.nn.Linear(latents.shape[1], 27)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(steps):
        logits = probe(latents[tr]).view(-1, 9, 3)
        loss = F.cross_entropy(logits.reshape(-1, 3), labels[tr].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(latents[va]).view(-1, 9, 3).argmax(-1)
    return (pred == labels[va]).float().mean(0).numpy(), va


def evaluate(net, data, gen):
    obs = torch.from_numpy(data["obs"][:])
    labels = torch.from_numpy(occupancy_labels(data["obs"][:]))
    z = torch.from_numpy(data["z"][:])
    with torch.no_grad():
        s0, _, v = net.initial(obs)
    cell_acc, va = fit_probe(s0, labels, gen)
    value_mae = float((v[va] - z[va]).abs().mean())
    decisive = z[va] != 0
    sign_acc = float((torch.sign(v[va][decisive]) == z[va][decisive]).float().mean())
    return s0.numpy(), cell_acc, value_mae, sign_acc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay", type=Path, default=baseline.REPLAY_PATH)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    data = baseline.validate_replay(args.replay)
    results, latents = {}, {}
    for name, (unroll, recon) in {"recon off (MuZero)": (5, "off"), "recon on (ablation)": (5, "on")}.items():
        gen = torch.Generator()
        gen.manual_seed(args.seed)
        net = load_offline_net(unroll, recon)
        s0, cell_acc, value_mae, sign_acc = evaluate(net, data, gen)
        results[name] = (cell_acc, value_mae, sign_acc)
        latents[f"latents_{recon}"] = s0

    print("linear probe on frozen latents (val split) vs the value head on the same latents")
    print("| checkpoint | occupancy acc (mean over cells) | worst cell | value MAE | value sign acc |")
    print("|---|---|---|---|---|")
    for name, (cell_acc, value_mae, sign_acc) in results.items():
        print(f"| {name} | {cell_acc.mean():.3f} | {cell_acc.min():.3f} | {value_mae:.3f} | {sign_acc:.3f} |")
    print("per-cell occupancy accuracy grids (cells 0..8):")
    for name, (cell_acc, _, _) in results.items():
        grid = " / ".join(",".join(f"{a:.2f}" for a in cell_acc[r * 3 : r * 3 + 3]) for r in range(3))
        print(f"  {name}: {grid}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        meta=json.dumps({"seed": args.seed, "keys": list(latents)}),
        obs=data["obs"][:], labels=occupancy_labels(data["obs"][:]), z=data["z"][:], **latents,
    )
    print(f"wrote {OUT_PATH} (for the in-browser probe lab)")


if __name__ == "__main__":
    main()
