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


def offline_ckpt_path(unroll: int, recon: str, ablate_value: bool = False, tag: str = "") -> Path:
    suffix = ('_recon' if recon == 'on' else '') + ('_valueblind' if ablate_value else '')
    suffix += f"_{tag}" if tag else ""
    return CKPT_DIR / f"muzero_offline_k{unroll}{suffix}.pt"


class ReconDecoder(nn.Module):
    """Ablation-only auxiliary head for --recon on: latent -> the 18 observation bits.
    This is exactly the loss MuZero refuses to have; we bolt it on to see what changes."""

    def __init__(self, latent_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.net = mlp(latent_dim, [hidden], 18)

    def forward(self, s):
        return self.net(s)


class ReplayWindows:
    """Sample K-step teacher-forced windows out of a flat replay (frozen npz or live buffer)."""

    def __init__(self, replay_path: Path, unroll: int):
        data = baseline.validate_replay(replay_path)
        self._init_arrays(data["obs"][:], data["pi"][:], data["z"][:],
                          data["action"][:], data["game_id"][:], unroll)

    @classmethod
    def from_arrays(cls, obs, pi, z, action, game_id, unroll: int):
        self = cls.__new__(cls)
        self._init_arrays(obs, pi, z, action, game_id, unroll)
        return self

    def _init_arrays(self, obs, pi, z, action, game_id, unroll):
        self.obs = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        self.pi = torch.from_numpy(np.asarray(pi, dtype=np.float32))
        self.z = torch.from_numpy(np.asarray(z, dtype=np.float32))
        self.action = torch.from_numpy(np.asarray(action).astype(np.int64))
        game_id = np.asarray(game_id)
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


def unrolled_loss(net, batch, recon_decoder=None, ablate_value=False):
    """The MuZero loss (paper Eq. 1) on one minibatch; returns (total, parts dict).
    Recurrent steps (k >= 1) are scaled by 1/K (Appendix G). ablate_value=True drops the
    value AND reward terms (M1.g value-blind run: the latent's only teacher is policy)."""
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
        total = total + scale * (l_p + (0.0 if ablate_value else l_v))
        # parts carry the SAME 1/K scaling as the optimized loss, so the CSV decomposition
        # sums to `loss` and K=1 vs K=5 logs are comparable (codex review finding)
        parts["policy"] += float(l_p.detach()) * scale
        parts["value"] += float(l_v.detach()) * scale
        if k >= 1:
            l_r = F.mse_loss(rewards[k - 1], batch["u"][:, k - 1])
            if not ablate_value:
                total = total + scale * l_r
            parts["reward"] += float(l_r.detach()) * scale
    if recon_decoder is not None:
        # ablation: decode every unroll latent back to the observation bits
        s, _, _ = net.initial(batch["obs"])
        for k in range(K + 1):
            m = batch["obs_mask"][:, k]
            l_rec = (F.binary_cross_entropy_with_logits(
                recon_decoder(s), batch["obs_t"][:, k], reduction="none").mean(-1) * m
            ).sum() / m.sum().clamp_min(1)
            total = total + (1.0 if k == 0 else 1.0 / K) * l_rec
            parts["recon"] += float(l_rec.detach()) * (1.0 if k == 0 else 1.0 / K)
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
        loss, parts = unrolled_loss(net, batch, recon, ablate_value=args.ablate_value)
        opt.zero_grad()
        loss.backward()
        opt.step()
        logger.log(step=step, loss=float(loss.detach()), **parts)

    ckpt = offline_ckpt_path(args.unroll, args.recon, args.ablate_value, args.tag)
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


# ---------------------------------------------------------- M1.e: the flywheel, box swapped
def selfplay_game(game, adapter, mcts_cfg, rng, temp_plies: int = 2):
    """One self-play game, the capstone's play_game with the search living in latent space.
    Only two arrows touch reality: acting (the REAL game adjudicates each move) and the
    final outcome. Everything inside the search is imagined."""
    from nano_muzero.mcts import run_mcts

    s, ply, records = game.initial_state(), 0, []
    while not game.is_terminal(s):
        obs = game.encode(s)
        counts, _ = run_mcts(adapter, obs, game.legal_moves(s), mcts_cfg, noise_rng=rng)
        pi = counts / counts.sum()
        a = int(rng.choice(game.n_actions, p=pi)) if ply < temp_plies else int(np.argmax(counts))
        records.append(dict(board=s, to_play=game.to_play(s), obs=obs, pi=pi, action=a))
        s = game.apply(s, a)
        ply += 1
    z = game.winner(s)
    for i, r in enumerate(records):
        r["z"] = float(z * r["to_play"])
        r["u"] = r["z"] if i == len(records) - 1 else 0.0  # the ending move pays its mover
    return records


class GameBuffer:
    """Rolling buffer of whole self-play games; flattens into ReplayWindows to sample."""

    def __init__(self, capacity_games: int):
        self.capacity = capacity_games
        self.games = []
        self.total_games = 0

    def add(self, game_records: list):
        self.games.append(game_records)
        self.total_games += 1
        if len(self.games) > self.capacity:
            self.games.pop(0)

    def windows(self, unroll: int) -> "ReplayWindows":
        rows = [(gi, r) for gi, g in enumerate(self.games) for r in g]
        return ReplayWindows.from_arrays(
            obs=np.stack([r["obs"] for _, r in rows]),
            pi=np.stack([r["pi"] for _, r in rows]),
            z=np.array([r["z"] for _, r in rows], dtype=np.float32),
            action=np.array([r["action"] for _, r in rows]),
            game_id=np.array([gi for gi, _ in rows]),
            unroll=unroll,
        )


def reanalyze(buffer: GameBuffer, adapter, mcts_cfg, fraction: float = 0.25):
    """M1.f: same positions, new labels. Re-run the search with the CURRENT net on the
    oldest games' REAL stored boards and mint fresh pi targets. z stays: on a finished board
    game the outcome is ground truth (the paper's bootstrapped value refresh only matters
    when z bootstraps from a search value, i.e. Atari, Module 2). Environment steps: 0."""
    from envs.tictactoe import TicTacToe
    from nano_muzero.mcts import run_mcts

    game = TicTacToe()
    n = max(1, int(len(buffer.games) * fraction))
    refreshed = 0
    for g in buffer.games[:n]:
        for r in g:
            counts, _ = run_mcts(adapter, r["obs"], game.legal_moves(r["board"]), mcts_cfg)
            r["pi"] = counts / counts.sum()
            refreshed += 1
    return refreshed


def train_full(args):
    """M1.e: play with latent MCTS, store, sample windows, unrolled loss, repeat."""
    from envs.tictactoe import NoisyTicTacToe, TicTacToe
    from nano_muzero.mcts import MCTSConfig, NetAdapter

    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    game = NoisyTicTacToe(p=0.1, seed=args.seed) if args.env == "noisy" else TicTacToe()
    net = MuZeroNet(latent_dim=args.latent_dim, hidden=args.hidden)
    if args.init_from:
        payload = torch.load(args.init_from, map_location="cpu", weights_only=False)
        net.load_state_dict(payload["model"])
        print(f"warm-started from {args.init_from}")
    adapter = NetAdapter(net)
    mcts_cfg = MCTSConfig(n_sims=args.sims)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = GameBuffer(args.buffer_games)
    run_dir = new_run_dir("muzero-full")
    latest = REPO_ROOT / "runs" / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(run_dir.name)
    logger = CSVLogger(run_dir)
    # checkpoint series for the 1.8 lab's scrubber: untrained, early, mid, final
    snap_dir = run_dir / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    snap_iters = {0, max(1, args.iters // 4), args.iters // 2, args.iters - 1}

    for it in range(args.iters):
        if it in snap_iters:
            save_ckpt(snap_dir / f"iter_{it:03d}.pt", net, step=it,
                      extra={"latent_dim": args.latent_dim, "hidden": args.hidden,
                             "unroll": args.unroll, "phase": "selfplay-snapshot"})
        if buffer.total_games < args.games_cap:
            n_new = min(args.games_per_iter, args.games_cap - buffer.total_games)
            for _ in range(n_new):
                buffer.add(selfplay_game(game, adapter, mcts_cfg, rng))
        if args.reanalyze:
            reanalyze(buffer, adapter, mcts_cfg)
        windows = buffer.windows(args.unroll)
        loss = parts = None
        for _ in range(args.train_steps):
            batch = windows.sample(args.batch, gen)
            loss, parts = unrolled_loss(net, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits, _ = net.predict(net.initial(windows.obs)[0])
        amortized = float((logits.argmax(1) == windows.pi.argmax(1)).float().mean())
        logger.log(step=it, games=buffer.total_games, loss=float(loss.detach()),
                   amortized=amortized, **parts)

    ckpt = run_dir / "ckpt.pt"
    extra = {"latent_dim": args.latent_dim, "hidden": args.hidden, "unroll": args.unroll,
             "phase": "selfplay", "env": args.env, "reanalyze": args.reanalyze,
             "games": buffer.total_games}
    save_ckpt(ckpt, net, opt, step=args.iters, extra=extra)
    canonical = CKPT_DIR / (f"muzero_selfplay_{args.tag}.pt" if args.tag else "muzero_selfplay.pt")
    save_ckpt(canonical, net, opt, step=args.iters, extra=extra)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_manifest(run_dir, {
        "run_id": run_dir.name, "module": "nano_muzero", "phase": 1,
        "created": datetime.datetime.now().astimezone().isoformat(),
        "config_hash": config_hash(config), "config": config, "seed": args.seed,
        "gpu_type": "cpu", "modal_run_id": None, "dataset_revision_sha": None,
        "eval_seed_set": "arena seeds 0..19",
        "artifacts": [{"path": str(canonical.relative_to(REPO_ROOT)),
                       "sha256": hashlib.sha256(canonical.read_bytes()).hexdigest()}],
    })
    print(f"saved {ckpt} and {canonical}")
    return net


def load_selfplay_net(ckpt: Path) -> MuZeroNet:
    if ckpt.is_dir():
        ckpt = ckpt / "ckpt.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = MuZeroNet(latent_dim=payload["extra"]["latent_dim"], hidden=payload["extra"]["hidden"])
    net.load_state_dict(payload["model"])
    net.eval()
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
    ap.add_argument("--full", action="store_true", help="M1.e: full self-play loop")
    ap.add_argument("--unroll", type=int, default=5, help="K, the unroll depth")
    ap.add_argument("--recon", choices=["on", "off"], default="off")
    ap.add_argument("--steps", type=int, default=2000, help="offline: gradient steps")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--replay", type=Path, default=baseline.REPLAY_PATH)
    ap.add_argument("--seed", type=int, default=0)
    # --full knobs (M1.e-M1.g)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games-per-iter", type=int, default=25)
    ap.add_argument("--train-steps", type=int, default=150, help="gradient steps per iter")
    ap.add_argument("--sims", type=int, default=50, help="self-play simulations per move")
    ap.add_argument("--buffer-games", type=int, default=400)
    ap.add_argument("--games-cap", type=int, default=10**9,
                    help="M1.f: stop generating new games after this many (constrained-data)")
    ap.add_argument("--reanalyze", action="store_true", help="M1.f: refresh old pi targets")
    ap.add_argument("--env", choices=["clean", "noisy"], default="clean", help="M1.g")
    ap.add_argument("--tag", default="", help="suffix for the canonical checkpoint name")
    ap.add_argument("--ablate-value", action="store_true",
                    help="M1.g: offline value-blind run (policy is the latent's only teacher)")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="--full: warm-start the net from an offline (M1.d) checkpoint")
    args = ap.parse_args(argv)
    if args.full:
        train_full(args)
    elif args.offline:
        train_offline(args)
    else:
        ap.error("pick a phase: --offline (M1.d) or --full (M1.e)")


if __name__ == "__main__":
    main()
