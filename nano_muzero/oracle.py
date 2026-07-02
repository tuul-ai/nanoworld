"""The oracle harness: the true environment dressed up as h/g/f (M1.c, the module's best trick).

  h = identity: the "latent" is literally the real board tuple
  g = game.apply() + the true terminal reward, absorbing past the end
  f = the capstone's trained two-headed net

Latent MCTS running on THIS model must reproduce the capstone MCTS move for move on 20 fixed
positions. That proves the search rewrite (mcts.py) is correct before any learning happens,
so every later failure is a MODEL problem, never a search bug. Run the proof:

  python -m nano_muzero.oracle          # prints the 20-position table, exits 1 on mismatch
"""
import sys

import numpy as np
import torch

from envs.tictactoe import TicTacToe
from nano_muzero import baseline
from nano_muzero.baseline import AlphaZeroNet, net_eval
from nano_muzero.mcts import MCTSConfig, capstone_equivalent, run_mcts


def load_capstone_net() -> AlphaZeroNet:
    """The locally-trained checkpoint when present, else the committed CI fixture."""
    if baseline.CKPT_PATH.exists():
        return baseline.load_net()
    import json

    data = np.load(baseline.FIXTURE_PATH)
    meta = json.loads(str(data["meta"]))
    net = AlphaZeroNet(hidden=meta["hidden"])
    net.load_state_dict({k: torch.from_numpy(data[k]) for k in data.files if k != "meta"})
    net.eval()
    return net


class OracleModel:
    """Speaks the mcts.py model protocol; answers with the truth.

    Semantics match what a perfectly-trained nano_muzero would learn under D1 + the
    terminal-reward convention: mid-game reward 0; the transition INTO a terminal board pays
    the outcome from the mover's view; terminal boards are absorbing with value 0 (the game
    is over, there is no future return). Edges the true game forbids (occupied cells) are
    silent dead ends -- reward 0, value 0 -- because the capstone search never takes them and
    the latent search should learn not to either."""

    n_actions = 9

    def __init__(self, net):
        self.game, self.net = TicTacToe(), net

    def _logits(self, board):
        """Masked log-priors: exactly -inf-like on illegal cells (softmax mass exactly 0,
        handled by the gate config's zero-prior skip), exactly log(p) on legal ones so the
        reconstructed priors match the capstone's to the last ulp."""
        p, v = net_eval(self.net, self.game, board)
        legal = np.array(board) == 0
        logits = np.where(legal, np.log(p, where=legal, out=np.zeros(9)), -1e9)
        return logits, v

    def initial(self, board):
        logits, v = self._logits(board)
        return board, logits, v

    def recurrent(self, board, a: int):
        g = self.game
        if g.is_terminal(board) or board[a] != 0:
            return board, 0.0, np.zeros(9), 0.0  # absorbing / impossible: nothing out there
        nxt = g.apply(board, a)
        if g.is_terminal(nxt):
            r = float(g.winner(nxt) * g.to_play(board))  # outcome paid to the mover
            return nxt, r, np.zeros(9), 0.0
        logits, v = self._logits(nxt)
        return nxt, 0.0, logits, v


def fixed_positions(net, n: int = 20, n_sims: int = 200, seed: int = 0):
    """The 20 fixed positions: deterministic random playouts, keeping non-terminal positions
    where the CAPSTONE search itself is decisive (top move holds >= 50% of visits) -- a
    property of the baseline alone, so the harness never selects for agreement."""
    game, rng = TicTacToe(), np.random.default_rng(seed)
    seen, picked = set(), []
    while len(picked) < n:
        s = game.initial_state()
        while not game.is_terminal(s) and len(picked) < n:
            if s not in seen and len(game.legal_moves(s)) >= 3:
                seen.add(s)
                counts = baseline.run_mcts(game, net, s, n_sims)
                if counts.max() / counts.sum() >= 0.5:
                    picked.append(s)
            s = game.apply(s, int(rng.choice(game.legal_moves(s))))
    return picked


def compare(net, positions, n_sims: int = 200, cfg: MCTSConfig = None):
    """Run both searches on every position; return rows of (board, capstone counts, latent
    counts). Default cfg is the GATE config (capstone-equivalent switches), in which the two
    searches are the same algorithm and the root visit-count VECTORS must match exactly, not
    just the argmax -- a much stronger equivalence than same-move. Pass a MuZero-form cfg to
    measure how the normalizer + c1/c2 + parent-count convention shift low-sim decisions."""
    game, oracle = TicTacToe(), OracleModel(net)
    cfg = cfg or capstone_equivalent(n_sims)
    rows = []
    for s in positions:
        cap_counts = baseline.run_mcts(game, net, s, n_sims)
        lat_counts, _ = run_mcts(oracle, s, game.legal_moves(s), cfg)
        rows.append((s, cap_counts, lat_counts))
    return rows


def main():
    net = load_capstone_net()
    positions = fixed_positions(net)
    rows = compare(net, positions)  # the gate: identical algorithm, rules deleted
    muzero_rows = compare(net, positions, cfg=MCTSConfig(n_sims=200))  # informational
    print("oracle harness: capstone MCTS vs latent MCTS over true dynamics (200 sims each)")
    print("| # | board | capstone move | latent move | visit vectors | MuZero-cfg move |")
    print("|---|---|---|---|---|---|")
    ok = mz_ok = 0
    for i, ((s, cap, lat), (_, _, mz)) in enumerate(zip(rows, muzero_rows)):
        board = "".join({1: "X", -1: "O", 0: "."}[c] for c in s)
        exact = bool(np.array_equal(cap, lat))
        ok += exact
        mz_ok += int(np.argmax(cap)) == int(np.argmax(mz))
        print(f"| {i} | {board[:3]} {board[3:6]} {board[6:]} | {int(np.argmax(cap))}"
              f" | {int(np.argmax(lat))} | {'identical' if exact else 'DIFFER'} | {int(np.argmax(mz))} |")
    print(f"gate (capstone-equivalent config): {ok}/{len(rows)} exact visit-vector matches -- must be all")
    print(f"info (MuZero c1/c2/min-max/parent-N): {mz_ok}/{len(rows)} same move at 200 sims")
    sys.exit(0 if ok == len(rows) else 1)


if __name__ == "__main__":
    main()
