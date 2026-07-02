"""The M1.e gates, capstone-style "how you know it learned" -- asserted, not eyeballed.

  python -m nano_muzero.eval --ckpt runs/latest --gates full

Gates (all must pass for exit 0):
  1. never loses to a random mover over 200 games
  2. arena parity with the M1.a AlphaZero capstone over 100 sampled-opening games --
     parity is judged against the capstone's own MIRROR match under the same openings
     (the capstone is not perfect off its argmax line: it goes ~33W/38D/29L against
     itself, so "all draws" was never the honest bar); gate: nano's net score (W - L)
     within 10 points of the mirror baseline
  3. root value of the empty board settles near the draw value
  4. the oracle harness is still green (exact tree equivalence, so this comparison
     is apples to apples: same search, only the simulator differs)
  F. falsifier: latent-MCTS beats the raw policy head on arena seeds 0..19 -- if search
     inside the learned model adds nothing over the policy alone, the module's premise dies
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from envs.tictactoe import TicTacToe
from nano_muzero import baseline
from nano_muzero.mcts import MCTSConfig, NetAdapter, run_mcts
from nano_muzero.oracle import compare, fixed_positions, load_capstone_net
from nano_muzero.train import load_selfplay_net

REPO_ROOT = Path(__file__).resolve().parent.parent


def mcts_mover(adapter, game, n_sims):
    cfg = MCTSConfig(n_sims=n_sims)
    def move(s, counts=False):
        c, _ = run_mcts(adapter, game.encode(s), game.legal_moves(s), cfg)
        return c if counts else int(np.argmax(c))
    return move


def raw_policy_mover(net, game):
    def move(s, counts=False):
        with torch.no_grad():
            _, logits, _ = net.initial(torch.from_numpy(game.encode(s)).unsqueeze(0))
        masked = np.full(9, -np.inf)
        legal = game.legal_moves(s)
        masked[legal] = logits[0].numpy()[legal]
        if counts:  # masked softmax doubles as the opening distribution
            e = np.exp(masked - masked[legal].max())
            e[np.isinf(masked)] = 0.0
            return e
        return int(np.argmax(masked))
    return move


def capstone_mover(net_b, game, n_sims):
    def move(s, counts=False):
        c = baseline.run_mcts(game, net_b, s, n_sims)
        return c if counts else int(np.argmax(c))
    return move


def random_mover(game, rng):
    return lambda s: int(rng.choice(game.legal_moves(s)))


def arena(game, move_a, move_b, n_games, opening_rng=None, opening_plies: int = 2):
    """A plays X in even games, O in odd; returns (a_wins, draws, a_losses).

    When both movers are deterministic, every same-color game is the same game, so a
    100-game table silently reduces to 2 unique games. `opening_rng` fixes that: the first
    `opening_plies` moves are SAMPLED from the mover's own visit distribution (tau = 1,
    exactly the self-play convention), argmax after. The mover still plays its own beliefs;
    the games just stop being copies of each other."""
    w = d = l = 0
    for i in range(n_games):
        a_player = +1 if i % 2 == 0 else -1
        s, ply = game.initial_state(), 0
        while not game.is_terminal(s):
            mover = move_a if game.to_play(s) == a_player else move_b
            if opening_rng is not None and ply < opening_plies:
                counts = mover(s, counts=True)
                a = int(opening_rng.choice(len(counts), p=counts / counts.sum()))
            else:
                a = mover(s)
            s = game.apply(s, a)
            ply += 1
        z = game.winner(s) * a_player
        w, d, l = w + (z > 0), d + (z == 0), l + (z < 0)
    return w, d, l


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, default=REPO_ROOT / "data" / "ckpts" / "muzero_selfplay.pt")
    ap.add_argument("--gates", choices=["full"], default="full")
    ap.add_argument("--eval-sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    game = TicTacToe()
    net = load_selfplay_net(args.ckpt)
    adapter = NetAdapter(net)
    nano = mcts_mover(adapter, game, args.eval_sims)
    results = {}

    # gate 1: never loses to random over 200 games
    w, d, l = arena(game, nano, random_mover(game, np.random.default_rng(args.seed)), 200)
    results["1 never-loses vs random (200 games)"] = (f"{w}W {d}D {l}L", l == 0)

    # gate 2: arena parity with the capstone over 100 games (tau=1 sampled openings so the
    # two deterministic agents actually play 100 different games), judged against the
    # capstone's own mirror match under identical openings
    cap_net = load_capstone_net()
    cap = capstone_mover(cap_net, game, args.eval_sims)
    w, d, l = arena(game, nano, cap, 100, opening_rng=np.random.default_rng(args.seed))
    mw, md, ml = arena(game, cap, cap, 100, opening_rng=np.random.default_rng(args.seed))
    results["2 arena vs capstone (100 games)"] = (
        f"{w}W {d}D {l}L (net {w - l:+d}; capstone mirror {mw}W {md}D {ml}L, net {mw - ml:+d})",
        (w - l) >= (mw - ml) - 10)

    # gate 3: empty-board root value near the draw value
    counts, root_v = run_mcts(adapter, game.encode(game.initial_state()),
                              game.legal_moves(game.initial_state()), MCTSConfig(n_sims=args.eval_sims))
    results["3 empty-board root value"] = (f"{root_v:+.3f} (|v| < 0.25)", abs(root_v) < 0.25)

    # gate 4: oracle harness still green (exact tree equivalence)
    rows = compare(cap_net, fixed_positions(cap_net))
    exact = sum(bool(np.array_equal(c, m)) for _, c, m in rows)
    results["4 oracle harness"] = (f"{exact}/{len(rows)} exact visit vectors", exact == len(rows))

    # falsifier: search must beat the raw policy head AGAINST THE CAPSTONE (seeds 0..19,
    # 20 x 5 sampled-opening games each). Vs a random opponent a fully-amortized policy
    # legitimately ties its own search, so that comparison stops discriminating exactly
    # when training succeeds; against a strong opponent, search advantage still binds.
    net_score = {"mcts": 0, "raw": 0}
    raw = raw_policy_mover(net, game)
    for seed in range(20):
        for name, mover in (("mcts", nano), ("raw", raw)):
            w, d, l = arena(game, mover, cap, 5, opening_rng=np.random.default_rng(seed))
            net_score[name] += w - l
    results["F falsifier: MCTS > raw policy, vs capstone (seeds 0..19)"] = (
        f"mcts net {net_score['mcts']:+d} vs raw net {net_score['raw']:+d}",
        net_score["mcts"] > net_score["raw"])

    print(f"nano_muzero gates ({args.ckpt}, {args.eval_sims} sims)")
    print("| gate | result | pass |\n|---|---|---|")
    for name, (detail, ok) in results.items():
        print(f"| {name} | {detail} | {'PASS' if ok else 'FAIL'} |")
    sys.exit(0 if all(ok for _, ok in results.values()) else 1)


if __name__ == "__main__":
    main()
