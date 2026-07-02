"""Smoke-train tests: tiny end-to-end runs that must pass before anything bigger."""
import argparse

import numpy as np
import pytest
import torch

from envs.tictactoe import TicTacToe
from nano_muzero.baseline import AlphaZeroNet, play_game, save_replay
from nano_muzero.train import ReplayWindows, ReconDecoder, MuZeroNet, unrolled_loss


@pytest.fixture
def tiny_replay(tmp_path):
    game, net = TicTacToe(), AlphaZeroNet()
    rng = np.random.default_rng(0)
    games = [play_game(game, net, n_sims=8, rng=rng) for _ in range(4)]
    return save_replay(tmp_path / "tiny.npz", games, meta=dict(games=4, sims=8, seed=0))


@pytest.mark.smoke
@pytest.mark.parametrize("recon", [None, "on"])
def test_muzero_offline_smoke(tiny_replay, recon):
    """20 steps of the offline unrolled loss on a 4-game replay: loss falls, stays finite."""
    torch.manual_seed(0)
    gen = torch.Generator()
    gen.manual_seed(0)
    windows = ReplayWindows(tiny_replay, unroll=3)
    net = MuZeroNet(latent_dim=8, hidden=32)
    dec = ReconDecoder(8, 32) if recon else None
    params = list(net.parameters()) + (list(dec.parameters()) if dec else [])
    opt = torch.optim.Adam(params, lr=1e-2)
    losses = []
    for _ in range(20):
        batch = windows.sample(16, gen)
        loss, parts = unrolled_loss(net, batch, dec)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0], f"loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"


@pytest.mark.smoke
def test_muzero_window_targets_aligned(tiny_replay):
    """The off-by-one killer (README: '90% of nano_muzero bugs'): reward target for step k
    must be the reward of the k-th teacher-forced action, and only game-ending transitions
    pay. Recompute one batch's targets by hand from the raw replay and compare."""
    data = np.load(tiny_replay)
    windows = ReplayWindows(tiny_replay, unroll=3)
    gen = torch.Generator()
    gen.manual_seed(1)
    batch = windows.sample(32, gen)
    ends = np.searchsorted(data["game_id"], data["game_id"], side="right")
    for b in range(32):
        j = int(batch["rows"][b])
        assert np.array_equal(batch["obs"][b].numpy(), data["obs"][j])
        L = ends[j]
        for k in range(3):
            row = j + k
            if row < L:
                assert batch["actions"][b, k] == data["action"][row]
                expected_u = data["z"][row] if row == L - 1 else 0.0
                assert batch["u"][b, k] == expected_u, f"reward target misaligned at k={k}"
        for k in range(4):
            if j + k < L:
                assert batch["pi_mask"][b, k] == 1
                assert np.allclose(batch["pi"][b, k], data["pi"][j + k])
                assert batch["z"][b, k] == data["z"][j + k]
            else:
                assert batch["pi_mask"][b, k] == 0
                assert batch["z"][b, k] == 0
