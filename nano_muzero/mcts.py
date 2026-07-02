"""Latent MCTS: the capstone's search with `game.apply()` deleted (M1.c).

The tree algorithm is unchanged -- select by PUCT, expand, evaluate, back up with a sign
flip per ply. Three honest differences, each straight from the MuZero paper (Appendix B),
each a teaching point in scene 1.3:

  1. legality lives only at the root: the real state is known there, so illegal actions are
     masked; deeper in the tree there is no board, so every action gets an edge;
  2. no terminal detection inside the tree: the search can imagine right past the end of a
     game (the model is trained to make terminals absorbing: reward 0, value 0);
  3. rewards + discount enter the backup, and Q is min-max normalized by the extremes seen
     in THIS tree so PUCT's exploration term has a [0,1]-scale Q to argue with.

The search never touches the environment. It talks to a model through two calls:

  initial(root)   -> (s, policy logits, value)          # h + f, once per real move
  recurrent(s, a) -> (s', reward, policy logits, value) # g + f, once per tree edge

Anything that speaks this protocol can sit in the model seat: the trained MuZeroNet (via
NetAdapter below), or the true rules dressed up as h/g/f (oracle.py) -- which is how this
file is proven correct before any learning happens.

Sign conventions (two-player, alternating plies): a node's accumulated value `w` is from
that node's own to-play perspective; the edge's Q from the parent's view is
r + gamma * (-w/n), with rewards from the mover's (parent's) view.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MCTSConfig:
    n_sims: int = 50
    c1: float = 1.25         # PUCT constants, MuZero form (Appendix B)
    c2: float = 19652.0      # math.inf drops the log growth term entirely
    discount: float = 1.0    # board games: gamma = 1
    dirichlet_alpha: float = 0.3
    noise_frac: float = 0.25
    use_minmax: bool = True        # False = raw Q in selection, exactly the capstone
    parent_visits_from_node: bool = True  # paper: sqrt(parent.N); capstone: sqrt(sum_b N(b))
    skip_zero_prior: bool = False  # gate-only: P(a)=0 edges are never selected


def capstone_equivalent(n_sims: int) -> "MCTSConfig":
    """The oracle-harness gate config: min-max normalization off, constant exploration
    coefficient 1.5, parent count = sum of child visits, and hard-masked (exactly-zero
    prior) edges never selected. Under these four switches this search IS the capstone's
    algorithm -- the only thing deleted is game.apply(). Any difference in the search
    trees over true dynamics is a bug."""
    return MCTSConfig(n_sims=n_sims, c1=1.5, c2=math.inf, use_minmax=False,
                      parent_visits_from_node=False, skip_zero_prior=True)


class MinMaxStats:
    """Q normalizer: track the min/max edge-Q seen anywhere in this tree, rescale to [0,1].
    Before two distinct bounds exist, Q passes through raw (paper behavior)."""

    def __init__(self):
        self.lo, self.hi = math.inf, -math.inf

    def update(self, q: float):
        self.lo, self.hi = min(self.lo, q), max(self.hi, q)

    def normalize(self, q: float) -> float:
        if self.hi > self.lo:
            return (q - self.lo) / (self.hi - self.lo)
        return q


class LatentNode:
    """One tree node = one latent state. No board anywhere below the root."""

    __slots__ = ("s", "reward", "edge_priors", "n", "w", "children")

    def __init__(self, s, reward: float, edge_priors: dict):
        self.s = s
        self.reward = reward            # r on the edge INTO this node, mover's view
        self.edge_priors = edge_priors  # action -> prior for edges OUT of this node
        self.n = 0
        self.w = 0.0
        self.children = {}              # action -> LatentNode, created on first traversal

    def q_from_parent(self, discount: float) -> float:
        return self.reward + discount * (-self.w / self.n)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _select_action(node: LatentNode, cfg: MCTSConfig, minmax: MinMaxStats) -> int:
    """argmax over PUCT(a) = Q_bar(a) + P(a) * sqrt(parent N) / (1+N(a)) * (c1 + log(...)).
    Sorted iteration makes ties deterministic (lowest action wins). Unvisited edges score
    with Q = 0, i.e. the tree's own average-ish pessimism, exactly as the paper.

    parent N is the node's own visit count (paper Appendix B) by default, so a freshly
    expanded node's first revisit is guided by priors instead of tie-breaking to the lowest
    action; the capstone-equivalent config counts child visits, as the capstone does."""
    parent_n = node.n if cfg.parent_visits_from_node else sum(ch.n for ch in node.children.values())
    growth = cfg.c1 if math.isinf(cfg.c2) else cfg.c1 + math.log((parent_n + cfg.c2 + 1) / cfg.c2)
    best_a, best_score = None, -math.inf
    for a in sorted(node.edge_priors):
        child = node.children.get(a)
        n = child.n if child else 0
        if cfg.skip_zero_prior and node.edge_priors[a] == 0.0 and n == 0:
            continue  # gate config only: a hard-masked edge (P exactly 0) must not outscore
            # a losing-but-real one just because its Q sits at the unvisited default of 0.
            # The capstone never creates these edges; skipping keeps the trees identical.
        q = 0.0
        if child and child.n > 0:
            q = child.q_from_parent(cfg.discount)
            if cfg.use_minmax:
                q = minmax.normalize(q)
        score = q + node.edge_priors[a] * math.sqrt(parent_n) / (1 + n) * growth
        if score > best_score:
            best_a, best_score = a, score
    return best_a


def run_mcts(model, root_input, legal_actions, cfg: MCTSConfig, noise_rng=None):
    """Search from one real observation. Returns (visit counts, root value estimate).

    `legal_actions` is the only fact about the real environment the search may use, and
    only at the root (difference 1). Terminals are never checked (difference 2)."""
    s0, logits, _ = model.initial(root_input)
    p = _softmax(logits)
    legal_actions = sorted(legal_actions)  # canonical order: Dirichlet noise attaches per
    # action identically however the caller ordered its legal-move list (capstone parity)
    priors = {a: float(p[a]) for a in legal_actions}
    norm = sum(priors.values()) or 1.0
    priors = {a: pr / norm for a, pr in priors.items()}  # root legality mask + renorm
    if noise_rng is not None:  # self-play exploration, identical to the capstone
        noise = noise_rng.dirichlet([cfg.dirichlet_alpha] * len(priors))
        priors = {a: (1 - cfg.noise_frac) * pr + cfg.noise_frac * float(e)
                  for (a, pr), e in zip(priors.items(), noise)}
    root = LatentNode(s0, reward=0.0, edge_priors=priors)
    minmax = MinMaxStats()

    for _ in range(cfg.n_sims):
        # 1) SELECT: walk down by PUCT until an edge that has never been traversed
        node, path = root, [root]
        a = _select_action(node, cfg, minmax)
        while a in node.children:
            node = node.children[a]
            path.append(node)
            a = _select_action(node, cfg, minmax)

        # 2) EXPAND + EVALUATE: one g call (the chip that replaced game.apply), one f call.
        # The new node gets priors over ALL actions -- legality does not exist down here.
        s_next, r, child_logits, v = model.recurrent(node.s, a)
        pri = _softmax(child_logits)
        child = LatentNode(s_next, reward=float(r), edge_priors={i: float(pri[i]) for i in range(len(pri))})
        node.children[a] = child
        path.append(child)

        # 3) BACKUP: leaf's view first; one sign flip + edge reward per ply on the way up
        v = float(v)
        for nd in reversed(path):
            nd.n += 1
            nd.w += v
            if nd is not root:
                minmax.update(nd.q_from_parent(cfg.discount))
            v = nd.reward + cfg.discount * (-v)

    counts = np.zeros(model.n_actions, dtype=np.float32)
    for a, ch in root.children.items():
        counts[a] = ch.n
    return counts, root.w / root.n


class NetAdapter:
    """Wraps a MuZeroNet (batched tensors) into the single-position protocol the search
    speaks. Everything is detached to numpy: the search plans, it never backprops."""

    def __init__(self, net):
        self.net = net
        self.n_actions = net.n_actions

    def initial(self, obs: np.ndarray):
        with torch.no_grad():
            s, p, v = self.net.initial(torch.from_numpy(obs).unsqueeze(0))
        return s[0], p[0].numpy().astype(np.float64), float(v[0])

    def recurrent(self, s, a: int):
        with torch.no_grad():
            s2, r, p, v = self.net.recurrent(s.unsqueeze(0), torch.tensor([a]))
        return s2[0], float(r[0]), p[0].numpy().astype(np.float64), float(v[0])
