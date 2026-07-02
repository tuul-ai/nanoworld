"""The AlphaZero capstone, re-run: net (p, v) + MCTS over true rules + self-play (M1.a).

This file is the shipped capstone code (implemented from courses/alphazero/capstone/AGENT.md,
the course's source of truth) and the module's baseline: nano_muzero must reach arena parity
with it, and the whole module is the story of deleting exactly one call from this file --
`game.apply(s, a)` inside the search -- and replacing it with learned h/g/f.

D1 evidence (value convention), cited by DECISIONS.md:
  - value head ends in tanh, v in [-1, 1], from the to-play player's view   (AlphaZeroNet)
  - value target z in {-1, 0, +1}: final outcome from each mover's view     (play_game)
  - loss = cross_entropy(policy, pi) + mse(v, z)                            (train)
The AlphaZero deck's [0,1]+BCE presentation is a deck convention; the shipped code is this.

Run it:
  python -m nano_muzero.baseline                      # train, save ckpt + frozen replay
  python -m nano_muzero.baseline --games 200 --assert-never-loses   # the M1.a gate
"""
import argparse
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.tictactoe import TicTacToe
from shared.log import CSVLogger, config_hash, load_ckpt, new_run_dir, save_ckpt, write_manifest
from shared.nets import mlp
from shared.seed import seed_all

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = REPO_ROOT / "data" / "ckpts" / "ttt_capstone.pt"
REPLAY_PATH = REPO_ROOT / "data" / "replays" / "ttt_capstone.npz"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "capstone_net.npz"


# ---------------------------------------------------------------- net: one trunk, two heads
class AlphaZeroNet(nn.Module):
    """encode(s) -> flatten -> MLP trunk -> (policy logits over 9 moves, value in [-1,1])."""

    def __init__(self, obs_dim: int = 18, hidden: int = 64, n_actions: int = 9):
        super().__init__()
        self.trunk = mlp(obs_dim, [hidden], hidden, out_act="relu")
        self.policy_head = mlp(hidden, [], n_actions)
        self.value_head = mlp(hidden, [], 1, out_act="tanh")

    def forward(self, x):
        t = self.trunk(x.flatten(1))
        return self.policy_head(t), self.value_head(t).squeeze(-1)


def net_eval(net, game, s):
    """Evaluate one position: prior over LEGAL moves (illegal masked to -inf before softmax)
    and value, both from the to-play player's view."""
    with torch.no_grad():
        logits, v = net(torch.from_numpy(game.encode(s)).unsqueeze(0))
    logits = logits[0].numpy().astype(np.float64)
    mask = np.full(game.n_actions, -np.inf)
    mask[game.legal_moves(s)] = 0.0
    e = np.exp(logits + mask - (logits + mask).max())
    return e / e.sum(), float(v)


# ------------------------------------------------------------------------- MCTS with PUCT
class Node:
    """One tree node = one REAL board state (this is the luxury MuZero gives up).
    `w` accumulates value from this node's own to-play perspective, so the parent reads
    the edge as Q = -w/n (one sign flip per ply, players alternate)."""

    __slots__ = ("state", "prior", "n", "w", "children")

    def __init__(self, state, prior: float = 0.0):
        self.state = state
        self.prior = prior
        self.n = 0
        self.w = 0.0
        self.children = {}  # action -> Node


def _expand(node, game, net):
    """Attach a child per legal move with the net's priors; return the net's value."""
    p, v = net_eval(net, game, node.state)
    for a in game.legal_moves(node.state):
        node.children[a] = Node(game.apply(node.state, a), prior=float(p[a]))
    return v


def _select_child(node, c_puct: float):
    """argmax PUCT(a) = Q(a) + c_puct * P(a) * sqrt(sum_b N(b)) / (1 + N(a)).
    Iteration in sorted action order makes ties deterministic (lowest action wins)."""
    total = sum(ch.n for ch in node.children.values())
    best_a, best_child, best_score = None, None, -math.inf
    for a in sorted(node.children):
        ch = node.children[a]
        q = 0.0 if ch.n == 0 else -ch.w / ch.n
        score = q + c_puct * ch.prior * math.sqrt(total) / (1 + ch.n)
        if score > best_score:
            best_a, best_child, best_score = a, ch, score
    return best_a, best_child


def run_mcts(game, net, root_state, n_sims: int, c_puct: float = 1.5, noise_rng=None):
    """The capstone search: select by PUCT, expand + evaluate with the net (true result at
    terminals), back the value up with a sign flip per ply. Returns root visit counts."""
    root = Node(root_state)
    _expand(root, game, net)
    if noise_rng is not None:  # self-play exploration: P = 0.75*P + 0.25*Dir(0.3) at the root
        dir_noise = noise_rng.dirichlet([0.3] * len(root.children))
        for eps, ch in zip(dir_noise, root.children.values()):
            ch.prior = 0.75 * ch.prior + 0.25 * float(eps)
    for _ in range(n_sims):
        node, path = root, [root]
        while node.children:  # descend until unexpanded leaf (terminals never get children)
            _, node = _select_child(node, c_puct)
            path.append(node)
        if game.is_terminal(node.state):
            # true result, from the leaf's to-play view (the winner just moved, so a decided
            # game reads -1 for whoever would be next)
            v = game.winner(node.state) * game.to_play(node.state)
        else:
            v = _expand(node, game, net)
        for nd in reversed(path):  # leaf first: v is from the leaf's to-play perspective
            nd.n += 1
            nd.w += v
            v = -v
    counts = np.zeros(game.n_actions, dtype=np.float32)
    for a, ch in root.children.items():
        counts[a] = ch.n
    return counts


# ------------------------------------------------------------------- self-play -> examples
def play_game(game, net, n_sims: int, rng, temp_plies: int = 2):
    """One self-play game. Returns per-position records; z is the final outcome re-signed to
    each position's mover ('label every stored position from that mover's view')."""
    s, ply, records = game.initial_state(), 0, []
    while not game.is_terminal(s):
        counts = run_mcts(game, net, s, n_sims, noise_rng=rng)
        pi = counts / counts.sum()
        if ply < temp_plies:  # tau = 1: sample for variety early
            a = int(rng.choice(game.n_actions, p=pi))
        else:  # tau -> 0: argmax for strength late
            a = int(np.argmax(counts))
        records.append(
            dict(board=s, to_play=game.to_play(s), obs=game.encode(s), pi=pi, action=a)
        )
        s = game.apply(s, a)
        ply += 1
    z = game.winner(s)
    for r in records:
        r["z"] = float(z * r["to_play"])
    return records


def arena_vs_random(game, net, n_games: int, n_sims: int, rng):
    """Net (MCTS, no noise, argmax) vs uniform-random mover; alternate colors. Returns
    (wins, draws, losses) from the net's perspective."""
    w = d = l = 0
    for i in range(n_games):
        net_player = +1 if i % 2 == 0 else -1
        s = game.initial_state()
        while not game.is_terminal(s):
            if game.to_play(s) == net_player:
                a = int(np.argmax(run_mcts(game, net, s, n_sims)))
            else:
                a = int(rng.choice(game.legal_moves(s)))
            s = game.apply(s, a)
        z = game.winner(s) * net_player
        w, d, l = w + (z > 0), d + (z == 0), l + (z < 0)
    return w, d, l


# ----------------------------------------------------------------------------- the replay
REPLAY_SCHEMA = "ttt-replay-v1"
REPLAY_KEYS = ("obs", "board", "to_play", "action", "pi", "z", "game_id", "move_idx")


def save_replay(path: Path, games: list, meta: dict):
    """Flatten a list of play_game() outputs into one npz: full (o, a, pi, z) tuples plus the
    raw boards (so widgets and Reanalyze can reconstruct real positions)."""
    rows = [r for g in games for r in g]
    arrays = dict(
        obs=np.stack([r["obs"] for r in rows]).astype(np.float32),
        board=np.array([r["board"] for r in rows], dtype=np.int8),
        to_play=np.array([r["to_play"] for r in rows], dtype=np.int8),
        action=np.array([r["action"] for r in rows], dtype=np.int8),
        pi=np.stack([r["pi"] for r in rows]).astype(np.float32),
        z=np.array([r["z"] for r in rows], dtype=np.float32),
        game_id=np.concatenate([np.full(len(g), i, dtype=np.int32) for i, g in enumerate(games)]),
        move_idx=np.concatenate([np.arange(len(g), dtype=np.int8) for g in games]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps({"schema": REPLAY_SCHEMA, **meta}), **arrays)
    return path


def validate_replay(path: Path):
    """Schema check for the frozen dataset; raises on any violation, returns the npz dict."""
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    assert meta["schema"] == REPLAY_SCHEMA, f"unknown schema {meta.get('schema')}"
    n = len(data["obs"])
    game = TicTacToe()
    for k in REPLAY_KEYS:
        assert k in data, f"missing key {k}"
        assert len(data[k]) == n, f"misaligned {k}: {len(data[k])} != {n}"
    assert data["obs"].shape == (n, 2, 3, 3) and data["pi"].shape == (n, 9)
    assert np.allclose(data["pi"].sum(1), 1.0, atol=1e-5), "pi rows must sum to 1"
    assert set(np.unique(data["z"])) <= {-1.0, 0.0, 1.0}, "z must be -1/0/+1 from mover's view"
    for i in range(n):
        board = tuple(int(c) for c in data["board"][i])
        assert game.to_play(board) == data["to_play"][i], f"row {i}: to_play mismatch"
        assert board[data["action"][i]] == 0, f"row {i}: stored action was illegal"
        assert np.array_equal(game.encode(board), data["obs"][i]), f"row {i}: obs != encode(board)"
        illegal = [a for a in range(9) if board[a] != 0]
        assert data["pi"][i][illegal].sum() == 0, f"row {i}: pi mass on illegal moves"
    return data


# ------------------------------------------------------------------------------- training
def train(args):
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    game = TicTacToe()
    net = AlphaZeroNet(hidden=args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    run_dir = new_run_dir("baseline")
    logger = CSVLogger(run_dir)

    for it in range(args.iters):
        # 1) self-play: fresh data each iteration, best-net==current-net at TTT scale
        games = [play_game(game, net, args.sims, rng) for _ in range(args.games_per_iter)]
        rows = [r for g in games for r in g]
        obs = torch.from_numpy(np.stack([r["obs"] for r in rows]))
        pi_t = torch.from_numpy(np.stack([r["pi"] for r in rows]))
        z_t = torch.tensor([r["z"] for r in rows], dtype=torch.float32)

        # 2) train: policy CE (soft targets) + value MSE, a few epochs over this iter's data
        for _ in range(args.epochs):
            perm = torch.randperm(len(rows))
            for i in range(0, len(rows), 64):
                idx = perm[i : i + 64]
                logits, v = net(obs[idx])
                loss = F.cross_entropy(logits, pi_t[idx]) + F.mse_loss(v, z_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

        # 3) how you know it learned: arena vs random + "net amortizes the search" agreement
        with torch.no_grad():
            logits, _ = net(obs)
        amortized = float((logits.argmax(1) == pi_t.argmax(1)).float().mean())
        w, d, l = arena_vs_random(game, net, args.arena_games, args.sims, rng)
        logger.log(step=it, loss=float(loss.detach()), amortized=amortized, wins=w, draws=d, losses=l)

    # 4) persist: checkpoint, CI fixture (weights as npz), frozen replay, manifest
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(CKPT_PATH, net, opt, step=args.iters, extra={"hidden": args.hidden})
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        FIXTURE_PATH,
        meta=json.dumps({"hidden": args.hidden, "seed": args.seed}),
        **{k: v.numpy() for k, v in net.state_dict().items()},
    )
    replay_games = [play_game(game, net, args.sims, rng) for _ in range(args.replay_games)]
    save_replay(
        REPLAY_PATH,
        replay_games,
        meta=dict(games=args.replay_games, sims=args.sims, seed=args.seed, hidden=args.hidden),
    )
    validate_replay(REPLAY_PATH)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_manifest(
        run_dir,
        {
            "run_id": run_dir.name,
            "module": "nano_muzero",
            "phase": 1,
            "created": datetime.datetime.now().astimezone().isoformat(),
            "config_hash": config_hash(config),
            "config": config,
            "seed": args.seed,
            "gpu_type": "cpu",
            "modal_run_id": None,
            "dataset_revision_sha": None,
            "artifacts": [
                {"path": str(p.relative_to(REPO_ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in (CKPT_PATH, REPLAY_PATH)
            ],
        },
    )
    print(f"saved: {CKPT_PATH}, {REPLAY_PATH} ({sum(len(g) for g in replay_games)} positions), fixture, {run_dir}/manifest.json")


def load_net(ckpt: Path = CKPT_PATH) -> AlphaZeroNet:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = AlphaZeroNet(hidden=payload["extra"]["hidden"])
    net.load_state_dict(payload["model"])
    net.eval()
    return net


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games-per-iter", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=3, help="epochs over each iter's fresh data")
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--arena-games", type=int, default=30, help="per-iteration arena vs random")
    ap.add_argument("--replay-games", type=int, default=500, help="frozen-replay size (M1.d dataset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    ap.add_argument("--games", type=int, default=0, help="eval mode: arena games vs random")
    ap.add_argument("--eval-sims", type=int, default=200, help="simulations per move in eval mode")
    ap.add_argument("--assert-never-loses", action="store_true")
    args = ap.parse_args(argv)

    if args.games:  # eval mode: the M1.a "never loses" gate
        seed_all(args.seed)
        net = load_net(args.ckpt)
        w, d, l = arena_vs_random(TicTacToe(), net, args.games, args.eval_sims, np.random.default_rng(args.seed))
        print(f"arena vs random over {args.games} games (net = MCTS+{args.ckpt.name}, {args.eval_sims} sims)")
        print("| wins | draws | losses |\n|---|---|---|")
        print(f"| {w} | {d} | {l} |")
        if args.assert_never_loses and l > 0:
            print(f"FAIL: lost {l} games")
            sys.exit(1)
        return
    train(args)


if __name__ == "__main__":
    main()
