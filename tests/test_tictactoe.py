"""Capstone-ported checks for envs/tictactoe.py: a known line ends with the right winner,
illegal moves are rejected, encoding is always from the to-play view, and the noisy variant
redirects at the specified rate while staying legal."""
import numpy as np
import pytest

from envs.tictactoe import NoisyTicTacToe, TicTacToe


def play(game, moves, s=None):
    s = game.initial_state() if s is None else s
    for a in moves:
        s = game.apply(s, a)
    return s


def test_known_line_x_wins_diagonal():
    g = TicTacToe()
    s = play(g, [0, 3, 4, 5, 8])  # X: 0,4,8 (diagonal); O: 3,5
    assert g.is_terminal(s)
    assert g.winner(s) == +1


def test_known_line_o_wins_column():
    g = TicTacToe()
    s = play(g, [0, 1, 3, 4, 8, 7])  # O: 1,4,7 (middle column)
    assert g.is_terminal(s)
    assert g.winner(s) == -1


def test_known_draw_line():
    g = TicTacToe()
    s = play(g, [0, 4, 8, 1, 7, 6, 2, 5, 3])
    assert g.is_terminal(s)
    assert g.winner(s) == 0
    assert g.legal_moves(s) == []


def test_illegal_moves_rejected():
    g = TicTacToe()
    s = g.apply(g.initial_state(), 4)
    with pytest.raises(ValueError):
        g.apply(s, 4)  # occupied
    with pytest.raises(ValueError):
        g.apply(s, 9)  # out of range
    won = play(g, [0, 3, 4, 5, 8])
    with pytest.raises(ValueError):
        g.apply(won, 1)  # terminal


def test_to_play_alternates_from_x():
    g = TicTacToe()
    s = g.initial_state()
    assert g.to_play(s) == +1
    s = g.apply(s, 0)
    assert g.to_play(s) == -1
    s = g.apply(s, 1)
    assert g.to_play(s) == +1


def test_encode_is_always_to_play_view():
    g = TicTacToe()
    s = g.apply(g.initial_state(), 0)  # X on cell 0, O to play
    enc = g.encode(s)
    assert enc.shape == (2, 3, 3) and enc.dtype == np.float32
    assert enc[0].sum() == 0  # "my stones" = O's: none yet
    assert enc[1, 0, 0] == 1 and enc[1].sum() == 1  # "their stones" = X on cell 0
    s = g.apply(s, 4)  # O on center, X to play: perspective flips back
    enc = g.encode(s)
    assert enc[0, 0, 0] == 1 and enc[0].sum() == 1
    assert enc[1, 1, 1] == 1 and enc[1].sum() == 1


def test_noisy_redirect_rate_and_legality():
    g = NoisyTicTacToe(p=0.1, seed=0)
    redirected, total = 0, 4000
    for _ in range(total):
        s = g.apply(g.initial_state(), 4)  # intend the center on an empty board
        assert sum(c != 0 for c in s) == 1  # exactly one stone landed, on an empty cell
        if s[4] == 0:
            redirected += 1
    # P(land off the intended cell) = p * 8/9; expect ~0.089 of 4000 = ~356
    rate = redirected / total
    assert 0.06 < rate < 0.12, f"off-target rate {rate:.3f} not near 0.089"


def test_noisy_p0_is_deterministic_tictactoe():
    g = NoisyTicTacToe(p=0.0, seed=0)
    s = play(g, [0, 3, 4, 5, 8])
    assert g.winner(s) == +1


def test_noisy_still_rejects_illegal_intent():
    g = NoisyTicTacToe(p=1.0, seed=0)
    s = (+1, 0, 0, 0, 0, 0, 0, 0, 0)  # X already on cell 0
    with pytest.raises(ValueError):
        g.apply(s, 0)  # occupied intent is illegal even though noise would re-land it
