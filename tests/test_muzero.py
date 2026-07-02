"""nano_muzero tests. -k model: shapes, latent range, NaN-free unrolls, gradient plumbing
(the 'unroll gradient reaches g at all K steps' check from M1.b)."""
import numpy as np
import pytest
import torch

from envs.tictactoe import TicTacToe
from nano_muzero.model import MuZeroNet, rescale_latent, scale_gradient


@pytest.fixture
def net():
    torch.manual_seed(0)
    return MuZeroNet()


def batch_obs(n=4):
    game, rng = TicTacToe(), np.random.default_rng(0)
    boards = []
    for _ in range(n):
        s = game.initial_state()
        for _ in range(int(rng.integers(0, 5))):
            if game.is_terminal(s):
                break
            s = game.apply(s, int(rng.choice(game.legal_moves(s))))
        boards.append(game.encode(s))
    return torch.from_numpy(np.stack(boards))


def test_model_shapes(net):
    obs = batch_obs(4)
    s0, p, v = net.initial(obs)
    assert s0.shape == (4, 16) and p.shape == (4, 9) and v.shape == (4,)
    s1, r, p1, v1 = net.recurrent(s0, torch.tensor([0, 4, 8, 2]))
    assert s1.shape == (4, 16) and r.shape == (4,) and p1.shape == (4, 9) and v1.shape == (4,)


def test_model_latent_rescaled_to_unit_range(net):
    obs = batch_obs(4)
    s0, _, _ = net.initial(obs)
    assert torch.allclose(s0.min(-1).values, torch.zeros(4), atol=1e-6)
    assert torch.allclose(s0.max(-1).values, torch.ones(4), atol=1e-6)
    s1, _, _, _ = net.recurrent(s0, torch.tensor([0, 1, 2, 3]))
    assert s1.min() >= 0 and s1.max() <= 1


def test_model_values_and_rewards_bounded(net):
    obs = batch_obs(4)
    s0, _, v = net.initial(obs)
    _, r, _, v1 = net.recurrent(s0, torch.tensor([0, 1, 2, 3]))
    for t in (v, r, v1):
        assert t.abs().max() <= 1.0  # tanh heads, D1 convention


def test_model_unroll_5_steps_no_nans(net):
    obs = batch_obs(8)
    actions = torch.randint(0, 9, (8, 5))
    policies, values, rewards = net.unroll(obs, actions)
    assert len(policies) == 6 and len(values) == 6 and len(rewards) == 5
    for t in [*policies, *values, *rewards]:
        assert torch.isfinite(t).all()


@pytest.mark.parametrize("k_loss", [1, 3, 5])
def test_model_unroll_gradient_reaches_g_at_all_k(net, k_loss):
    """A loss taken ONLY at unroll step k must send gradient back through every g application
    on the path -- and all the way into h. This is what 'trained through the composition'
    means; if it breaks, deep unroll steps silently stop teaching g."""
    obs = batch_obs(4)
    actions = torch.randint(0, 9, (4, 5))
    _, values, _ = net.unroll(obs, actions)
    net.zero_grad()
    values[k_loss].sum().backward()
    g_grad = net.g_core[0].weight.grad
    h_grad = net.h[0].weight.grad
    assert g_grad is not None and g_grad.abs().sum() > 0, f"no gradient into g from step {k_loss}"
    assert h_grad is not None and h_grad.abs().sum() > 0, f"no gradient into h from step {k_loss}"


def test_model_scale_gradient_halves_backward_only():
    x = torch.ones(3, requires_grad=True)
    y = scale_gradient(x, 0.5)
    assert torch.equal(y.detach(), x.detach())  # forward is the identity
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), 0.5))


def test_model_rescale_latent_handles_constant_rows():
    s = torch.zeros(2, 16)  # degenerate: max == min must not divide by zero
    out = rescale_latent(s)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------- mcts.py + oracle harness
from nano_muzero.mcts import MCTSConfig, NetAdapter, capstone_equivalent, run_mcts
from nano_muzero.oracle import OracleModel, compare, fixed_positions, load_capstone_net


class RiggedModel:
    """Hand-built dynamics for pinning the search's plumbing: no learning anywhere."""

    n_actions = 9

    def __init__(self, win_action=None, good_child_action=None):
        self.win_action = win_action          # this edge pays reward +1 (mover's view)
        self.good_child_action = good_child_action  # this child evaluates to -1 (child's view)

    def initial(self, obs):
        return 0, np.zeros(9), 0.0

    def recurrent(self, s, a):
        r = 1.0 if a == self.win_action else 0.0
        v = -0.9 if a == self.good_child_action else 0.0  # child view: bad for THEM
        return s + 1, r, np.zeros(9), v


def test_mcts_rigged_dynamics_finds_winning_reward():
    counts, root_v = run_mcts(RiggedModel(win_action=7), None, list(range(9)), MCTSConfig(n_sims=100))
    assert int(np.argmax(counts)) == 7
    assert counts.sum() == 100  # every simulation lands on exactly one root edge
    # root value is the visit-weighted mean, so it approximates the winning edge's visit
    # share rather than 1.0; positive and substantial is what a correct backup produces
    assert root_v > 0.2


def test_mcts_backup_sign_flip_prefers_bad_for_opponent():
    counts, _ = run_mcts(RiggedModel(good_child_action=3), None, list(range(9)), MCTSConfig(n_sims=100))
    assert int(np.argmax(counts)) == 3  # v=-0.9 from the child's view means +0.9 for the mover


def test_mcts_legality_masked_only_at_root():
    counts, _ = run_mcts(RiggedModel(), None, [0, 4], MCTSConfig(n_sims=60))
    assert counts[[1, 2, 3, 5, 6, 7, 8]].sum() == 0  # root: only legal edges get visits
    assert counts[0] + counts[4] == 60


def test_mcts_net_adapter_smoke():
    from nano_muzero.model import MuZeroNet

    torch.manual_seed(0)
    model = NetAdapter(MuZeroNet())
    game = TicTacToe()
    obs = game.encode(game.initial_state())
    counts, root_v = run_mcts(model, obs, list(range(9)), MCTSConfig(n_sims=30))
    assert counts.sum() == 30 and np.isfinite(root_v)


def test_oracle_harness_exact_tree_equivalence():
    """M1.c gate: latent MCTS over true dynamics == capstone MCTS on the 20 fixed positions
    in the capstone-equivalent config -- EXACT root visit-count vectors, not just the same
    argmax move. Both searches are deterministic, so this either always holds or is a bug."""
    net = load_capstone_net()
    rows = compare(net, fixed_positions(net))
    assert len(rows) == 20
    for s, cap, lat in rows:
        assert np.array_equal(cap, lat), f"visit vectors differ at {s}: {cap} vs {lat}"


def test_oracle_terminal_edges_pay_the_mover():
    net = load_capstone_net()
    oracle = OracleModel(net)
    board = (1, 1, 0, -1, -1, 0, 0, 0, 0)  # X to move; 2 wins for X, 5 wins for O next
    _, r, _, v = oracle.recurrent(board, 2)
    assert r == 1.0 and v == 0.0  # win pays +1 to the mover; terminal value is 0 (absorbing)
    nxt, r, _, _ = oracle.recurrent((1, 1, 1, -1, -1, 0, 0, 0, 0), 5)
    assert r == 0.0 and nxt == (1, 1, 1, -1, -1, 0, 0, 0, 0)  # past terminal: absorbing
