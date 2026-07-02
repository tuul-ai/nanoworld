"""Training. This file grows with the module: M1.d offline phase (here), M1.e self-play
(--full), M1.f Reanalyze (--reanalyze), M1.g break-it flags.

M1.d, the supervised world-model phase: train h/g/f on the FROZEN AlphaZero replay with the
K-step unrolled loss, before any MuZero self-play. From a sampled position o_t: encode once,
then push the actions that were ACTUALLY played (teacher forcing: targets only exist for
what happened) through g for K steps; at every step, grade the three heads against the real
trajectory:

    p^k -> pi_{t+k}   (the stored MCTS visit distribution)
    v^k -> z_{t+k}    (final outcome from that position's mover, D1 convention)
    r^k -> u_{t+k}    (0 mid-game; the game-ending transition pays the outcome)

Past the end of a game the state is absorbing: value and reward targets go to 0, the policy
loss is masked (no search distribution exists there), and the unroll continues with random
actions -- the model must learn that finished games stay finished no matter what you do.

Paper-faithful trick set (Appendix G), all three visible below: gradient halved entering g
(model.py), latent rescaled to [0,1] (model.py), and each recurrent step's loss scaled by
1/K so deeper unrolls do not mean more total gradient.

Run the two M1.d ablations (each ~30 s on CPU):
    python -m nano_muzero.train --offline --unroll 5
    python -m nano_muzero.train --offline --unroll 1
    python -m nano_muzero.train --offline --unroll 5 --recon on
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nano_muzero import baseline
from nano_muzero.model import MuZeroNet
from shared.log import CSVLogger, config_hash, new_run_dir, save_ckpt, write_manifest
from shared.nets import mlp
from shared.seed import seed_all

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "data" / "ckpts"


def offline_ckpt_path(unroll: int, recon: str) -> Path:
    return CKPT_DIR / f"muzero_offline_k{unroll}{'_recon' if recon == 'on' else ''}.pt"


class ReconDecoder(nn.Module):
    """Ablation-only auxiliary head for --recon on: latent -> the 18 observation bits.
    This is exactly the loss MuZero refuses to have; we bolt it on to see what changes."""

    def __init__(self, latent_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.net = mlp(latent_dim, [hidden], 18)

    def forward(self, s):
        return self.net(s)


class ReplayWindows:
    """Sample K-step teacher-forced windows out of the frozen replay."""

    def __init__(self, replay_path: Path, unroll: int):
        data = baseline.validate_replay(replay_path)
        self.obs = torch.from_numpy(data["obs"][:])
        self.pi = torch.from_numpy(data["pi"][:])
        self.z = torch.from_numpy(data["z"][:])
        self.action = torch.from_numpy(data["action"][:].astype(np.int64))
        game_id = data["game_id"][:]
        # game_end[j] = one past the last row of j's game (games are stored contiguously)
        ends = np.searchsorted(game_id, game_id, side="right")
        self.game_end = torch.from_numpy(ends.astype(np.int64))
        self.n, self.K = len(self.obs), unroll

    def sample(self, batch: int, gen: torch.Generator):
        j = torch.randint(0, self.n, (batch,), generator=gen)
        K = self.K
        actions = torch.zeros(batch, K, dtype=torch.long)
        pi_t = torch.zeros(batch, K + 1, 9)
        z_t = torch.zeros(batch, K + 1)
        u_t = torch.zeros(batch, K)
        pi_mask = torch.zeros(batch, K + 1)
        obs_t = torch.zeros(batch, K + 1, 18)  # recon targets (ablation)
        obs_mask = torch.zeros(batch, K + 1)
        for k in range(K + 1):
            row = j + k
            live = row < self.game_end[j]  # still inside the same game?
            safe = torch.where(live, row, j)  # clamp dead rows anywhere valid; masked anyway
            pi_t[:, k] = torch.where(live[:, None], self.pi[safe], torch.full((9,), 1 / 9.0))
            z_t[:, k] = torch.where(live, self.z[safe], torch.zeros(()))
            pi_mask[:, k] = live.float()
            obs_t[:, k] = torch.where(live[:, None], self.obs[safe].flatten(1), torch.zeros(18))
            obs_mask[:, k] = live.float()
            if k < K:
                # action a_{t+k+1} = the one taken at row j+k; past the end, any action --
                # absorbing means the model's answer must not depend on it
                rand_a = torch.randint(0, 9, (batch,), generator=gen)
                actions[:, k] = torch.where(live, self.action[safe], rand_a)
                # reward of that transition: the game-ending move pays z to its mover
                is_final = row == self.game_end[j] - 1
                u_t[:, k] = torch.where(is_final, self.z[safe], torch.zeros(()))
        return dict(obs=self.obs[j], actions=actions, pi=pi_t, z=z_t, u=u_t,
                    pi_mask=pi_mask, obs_t=obs_t, obs_mask=obs_mask, rows=j)


def unrolled_loss(net, batch, recon_decoder=None):
    """The MuZero loss (paper Eq. 1) on one minibatch; returns (total, parts dict).
    Recurrent steps (k >= 1) are scaled by 1/K (Appendix G)."""
    policies, values, rewards = net.unroll(batch["obs"], batch["actions"])
    K = len(rewards)
    parts = {"policy": 0.0, "value": 0.0, "reward": 0.0, "recon": 0.0}
    total = 0.0
    for k in range(K + 1):
        scale = 1.0 if k == 0 else 1.0 / K
        mask = batch["pi_mask"][:, k]
        ce = -(batch["pi"][:, k] * F.log_softmax(policies[k], dim=-1)).sum(-1)
        l_p = (ce * mask).sum() / mask.sum().clamp_min(1)
        l_v = F.mse_loss(values[k], batch["z"][:, k])
        total = total + scale * (l_p + l_v)
        parts["policy"] += float(l_p.detach())
        parts["value"] += float(l_v.detach())
        if k >= 1:
            l_r = F.mse_loss(rewards[k - 1], batch["u"][:, k - 1])
            total = total + scale * l_r
            parts["reward"] += float(l_r.detach())
    if recon_decoder is not None:
        # ablation: decode every unroll latent back to the observation bits
        s, _, _ = net.initial(batch["obs"])
        for k in range(K + 1):
            m = batch["obs_mask"][:, k]
            l_rec = (F.binary_cross_entropy_with_logits(
                recon_decoder(s), batch["obs_t"][:, k], reduction="none").mean(-1) * m
            ).sum() / m.sum().clamp_min(1)
            total = total + (1.0 if k == 0 else 1.0 / K) * l_rec
            parts["recon"] += float(l_rec.detach())
            if k < K:
                s, _, _, _ = net.recurrent(s, batch["actions"][:, k])
    return total, parts


def train_offline(args):
    gen = seed_all(args.seed)
    windows = ReplayWindows(args.replay, args.unroll)
    net = MuZeroNet(latent_dim=args.latent_dim, hidden=args.hidden)
    recon = ReconDecoder(args.latent_dim, args.hidden) if args.recon == "on" else None
    params = list(net.parameters()) + (list(recon.parameters()) if recon else [])
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    run_dir = new_run_dir("muzero-offline")
    logger = CSVLogger(run_dir, echo_every=max(1, args.steps // 20))

    for step in range(args.steps):
        batch = windows.sample(args.batch, gen)
        loss, parts = unrolled_loss(net, batch, recon)
        opt.zero_grad()
        loss.backward()
        opt.step()
        logger.log(step=step, loss=float(loss.detach()), **parts)

    ckpt = offline_ckpt_path(args.unroll, args.recon)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(ckpt, net, opt, step=args.steps,
              extra={"latent_dim": args.latent_dim, "hidden": args.hidden,
                     "unroll": args.unroll, "recon": args.recon, "phase": "offline"})
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_manifest(run_dir, {
        "run_id": run_dir.name, "module": "nano_muzero", "phase": 1,
        "created": datetime.datetime.now().astimezone().isoformat(),
        "config_hash": config_hash(config), "config": config, "seed": args.seed,
        "gpu_type": "cpu", "modal_run_id": None, "dataset_revision_sha": None,
        "artifacts": [{"path": str(ckpt.relative_to(REPO_ROOT)),
                       "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()}],
    })
    print(f"saved {ckpt}")
    return net


def load_offline_net(unroll: int, recon: str = "off") -> MuZeroNet:
    payload = torch.load(offline_ckpt_path(unroll, recon), map_location="cpu", weights_only=False)
    net = MuZeroNet(latent_dim=payload["extra"]["latent_dim"], hidden=payload["extra"]["hidden"])
    net.load_state_dict(payload["model"])
    net.eval()
    return net


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true", help="M1.d: train on the frozen replay")
    ap.add_argument("--unroll", type=int, default=5, help="K, the unroll depth")
    ap.add_argument("--recon", choices=["on", "off"], default="off")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--replay", type=Path, default=baseline.REPLAY_PATH)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not args.offline:
        ap.error("only --offline exists yet; --full arrives with M1.e")
    train_offline(args)


if __name__ == "__main__":
    main()
