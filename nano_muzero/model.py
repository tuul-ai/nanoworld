"""The three networks that replace the rulebook: h, g, f (M1.b).

  h (representation): observation -> latent s^0            runs once per real move
  g (dynamics):       (latent, action) -> (r, latent')     runs once per tree edge
  f (prediction):     latent -> (policy logits, value)     runs once per tree node

f is the capstone's two-headed net retargeted from boards to latents; h and g are the new
machinery. The latent has NO fixed meaning: nothing in training says "s must contain the
board" -- it is shaped only by what the three heads must predict (value equivalence, scene
1.4). Two stability tricks from the MuZero paper (Appendix G), both visible here, both
exercised by the tests:

  1. the latent is min-max rescaled to [0, 1] at every step, so it lives in the same range
     as the one-hot action that gets concatenated onto it;
  2. the gradient is halved at the start of each dynamics application, so a K-step unroll
     puts roughly constant total gradient pressure on g regardless of K.

The reward head is dormant machinery on Tic-Tac-Toe (u = 0 everywhere except the terminal
transition, where the env's outcome arrives as a reward); we keep it because Module 2 needs
it and the deck teaches it as the general form.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.nets import mlp


def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity in the forward pass; multiplies the gradient by `scale` in the backward.
    The 2-line idiom from the paper: the detached copy carries the rest of the value."""
    return x * scale + x.detach() * (1.0 - scale)


def rescale_latent(s: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max rescale to [0, 1] (paper Appendix G). Keeps every unroll step's
    latent in the same range as the one-hot action input; also stops latents drifting off
    to arbitrary scales as g composes with itself."""
    lo = s.min(dim=-1, keepdim=True).values
    hi = s.max(dim=-1, keepdim=True).values
    return (s - lo) / (hi - lo).clamp_min(1e-8)


class MuZeroNet(nn.Module):
    """h/g/f as small MLPs over the capstone's flattened 2-plane encoding."""

    def __init__(self, obs_dim: int = 18, latent_dim: int = 16, hidden: int = 64, n_actions: int = 9):
        super().__init__()
        self.latent_dim, self.n_actions = latent_dim, n_actions
        # h: encoder. obs -> s^0
        self.h = mlp(obs_dim, [hidden], latent_dim)
        # g: the learned rules. (s, one-hot a) -> shared core -> (next latent, reward)
        self.g_core = mlp(latent_dim + n_actions, [hidden], hidden, out_act="relu")
        self.g_state = mlp(hidden, [], latent_dim)
        self.g_reward = mlp(hidden, [], 1, out_act="tanh")  # r in [-1,1], mover's view
        # f: the old friend -- trunk + policy + value(tanh), same shape as AlphaZeroNet,
        # reading a latent instead of a board (D1: value in [-1,1], mover's view)
        self.f_trunk = mlp(latent_dim, [hidden], hidden, out_act="relu")
        self.f_policy = mlp(hidden, [], n_actions)
        self.f_value = mlp(hidden, [], 1, out_act="tanh")

    def predict(self, s):
        """f: latent -> (policy logits, value)."""
        t = self.f_trunk(s)
        return self.f_policy(t), self.f_value(t).squeeze(-1)

    def initial(self, obs):
        """h then f: the once-per-real-move call. obs (B,2,3,3) or (B,18)."""
        s0 = rescale_latent(self.h(obs.flatten(1)))
        p, v = self.predict(s0)
        return s0, p, v

    def recurrent(self, s, a):
        """g then f: the once-per-tree-edge call. `a` is a LongTensor of action ids (B,).
        Gradient is halved at the entrance of g (trick 2); latent rescaled on exit (trick 1)."""
        a_onehot = F.one_hot(a, self.n_actions).float()
        core = self.g_core(torch.cat([scale_gradient(s, 0.5), a_onehot], dim=-1))
        s_next = rescale_latent(self.g_state(core))
        r = self.g_reward(core).squeeze(-1)
        p, v = self.predict(s_next)
        return s_next, r, p, v

    def unroll(self, obs, actions):
        """Training-shape forward: encode once, then push K teacher-forced actions through g.
        actions: LongTensor (B, K). Returns lists of per-step heads:
        policies/values have K+1 entries (k = 0..K), rewards have K (k = 1..K, no r^0)."""
        s, p, v = self.initial(obs)
        policies, values, rewards = [p], [v], []
        for k in range(actions.shape[1]):
            s, r, p, v = self.recurrent(s, actions[:, k])
            policies.append(p)
            values.append(v)
            rewards.append(r)
        return policies, values, rewards
