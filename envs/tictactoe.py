"""Tic-Tac-Toe with the capstone Game interface, plus the noisy variant scene 1.9 needs.

The interface is ported verbatim from the AlphaZero capstone spec
(courses/alphazero/capstone/AGENT.md): MCTS and self-play call only these eight members and
never care which game they got. nano_muzero's whole premise is deleting `apply()` from the
search; keeping the interface identical is what makes that deletion a one-line diff.

State is a tuple of 9 ints in {+1, -1, 0} (X, O, empty), cells indexed row-major:

    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8

X (+1) always moves first, so whose turn it is falls out of the stone count. `winner` reports
from the fixed X-perspective (+1 X won, -1 O won, 0 draw/ongoing); per-mover value labels are
made by the caller as `z * player`, exactly as the capstone's selfplay does.
"""
import numpy as np

LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
)


class TicTacToe:
    n_actions = 9

    def initial_state(self) -> tuple:
        return (0,) * 9

    def legal_moves(self, s) -> list:
        return [a for a in range(9) if s[a] == 0]

    def apply(self, s, a) -> tuple:
        if self.is_terminal(s):
            raise ValueError("apply() on a terminal state")
        if not (0 <= a < 9) or s[a] != 0:
            raise ValueError(f"illegal move {a} in {s}")
        board = list(s)
        board[a] = self.to_play(s)
        return tuple(board)

    def is_terminal(self, s) -> bool:
        return self._line_winner(s) != 0 or all(c != 0 for c in s)

    def winner(self, s) -> int:
        """+1 / -1 / 0 from the fixed X-perspective (0 also means 'still going')."""
        return self._line_winner(s)

    def to_play(self, s) -> int:
        return +1 if sum(c != 0 for c in s) % 2 == 0 else -1

    def encode(self, s) -> np.ndarray:
        """Two 3x3 binary planes (my stones, their stones), always from the to-play view,
        so the net only ever learns "me vs them", never "X vs O"."""
        me = self.to_play(s)
        planes = np.zeros((2, 3, 3), dtype=np.float32)
        for i, c in enumerate(s):
            if c == me:
                planes[0, i // 3, i % 3] = 1.0
            elif c == -me:
                planes[1, i // 3, i % 3] = 1.0
        return planes

    @staticmethod
    def _line_winner(s) -> int:
        for i, j, k in LINES:
            if s[i] != 0 and s[i] == s[j] == s[k]:
                return s[i]
        return 0


class NoisyTicTacToe(TicTacToe):
    """Stochastic variant for scene 1.9's determinism failure mode: with probability `p` the
    chosen move lands on a uniformly random empty cell instead (possibly the intended one).
    A deterministic g trained on this data learns a blur; the value calibration shows it.
    """

    def __init__(self, p: float = 0.1, seed: int = 0):
        self.p = p
        self.rng = np.random.default_rng(seed)

    def apply(self, s, a) -> tuple:
        if self.is_terminal(s):
            raise ValueError("apply() on a terminal state")
        if not (0 <= a < 9) or s[a] != 0:
            raise ValueError(f"illegal move {a} in {s}")
        if self.rng.random() < self.p:
            a = int(self.rng.choice(self.legal_moves(s)))
        return super().apply(s, a)
