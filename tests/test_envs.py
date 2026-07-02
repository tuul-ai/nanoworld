"""Game-interface contract (both tic-tac-toe variants) + replay schema checks (M1.a)."""
import numpy as np
import pytest

from envs.tictactoe import NoisyTicTacToe, TicTacToe
from nano_muzero.baseline import (
    REPLAY_PATH,
    AlphaZeroNet,
    play_game,
    save_replay,
    validate_replay,
)


@pytest.mark.parametrize("game", [TicTacToe(), NoisyTicTacToe(p=0.1, seed=0)])
def test_game_interface_contract(game):
    """MCTS and self-play touch exactly these members; both variants must honor them."""
    assert game.n_actions == 9
    s = game.initial_state()
    assert game.legal_moves(s) == list(range(9))
    assert not game.is_terminal(s) and game.winner(s) == 0 and game.to_play(s) == +1
    enc = game.encode(s)
    assert enc.shape == (2, 3, 3) and enc.dtype == np.float32 and enc.sum() == 0

    rng = np.random.default_rng(1)
    for _ in range(20):  # random playouts terminate within 9 plies with a sane result
        s, plies = game.initial_state(), 0
        while not game.is_terminal(s):
            before = sum(c != 0 for c in s)
            s = game.apply(s, int(rng.choice(game.legal_moves(s))))
            assert sum(c != 0 for c in s) == before + 1  # exactly one stone lands
            plies += 1
        assert plies <= 9 and game.winner(s) in (-1, 0, 1)


def test_replay_schema_roundtrip(tmp_path):
    """A replay written by save_replay passes validate_replay; a corrupted one does not."""
    game, net = TicTacToe(), AlphaZeroNet()
    rng = np.random.default_rng(0)
    games = [play_game(game, net, n_sims=8, rng=rng) for _ in range(3)]
    path = save_replay(tmp_path / "replay.npz", games, meta=dict(games=3, sims=8, seed=0))

    data = validate_replay(path)
    assert data["game_id"].max() == 2
    assert (data["move_idx"][data["game_id"] == 0] == np.arange((data["game_id"] == 0).sum())).all()

    corrupt = {k: data[k].copy() for k in data.files if k != "meta"}
    corrupt["z"][0] = 0.5  # not a {-1, 0, +1} outcome
    np.savez(tmp_path / "bad.npz", meta=data["meta"], **corrupt)
    with pytest.raises(AssertionError):
        validate_replay(tmp_path / "bad.npz")


@pytest.mark.skipif(not REPLAY_PATH.exists(), reason="canonical replay not built on this machine")
def test_canonical_replay_valid():
    """The frozen M1.d dataset on this machine conforms to the schema."""
    data = validate_replay(REPLAY_PATH)
    assert len(data["obs"]) > 1000  # 500 games' worth of positions
