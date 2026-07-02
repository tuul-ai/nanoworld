"""ReplayBuffer (uniform transitions) + SequenceBuffer (contiguous windows out of episodes).

One episode schema everywhere: dict of arrays with matching first dim (obs, action, reward,
done, ...). The buffers never care whether data came from tic-tac-toe self-play, the YAM
recorder, or HIW shards.
"""
import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions; uniform sampling under a fixed Generator."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._store = {}
        self._n = 0
        self._ptr = 0

    def __len__(self):
        return self._n

    def add(self, **transition):
        for k, v in transition.items():
            v = np.asarray(v, dtype=np.float32)
            if k not in self._store:
                self._store[k] = np.zeros((self.capacity, *v.shape), dtype=np.float32)
            self._store[k][self._ptr] = v
        self._ptr = (self._ptr + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def sample(self, batch: int, gen: torch.Generator) -> dict:
        idx = torch.randint(0, self._n, (batch,), generator=gen).numpy()
        return {k: torch.from_numpy(v[idx]) for k, v in self._store.items()}


class SequenceBuffer:
    """Stores whole episodes; samples contiguous windows of length `seq_len`."""

    def __init__(self, capacity_episodes: int, seq_len: int):
        self.capacity = capacity_episodes
        self.seq_len = seq_len
        self._episodes = []

    def __len__(self):
        return len(self._episodes)

    def add_episode(self, episode: dict):
        lengths = {k: len(v) for k, v in episode.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"episode arrays misaligned: {lengths}")
        if next(iter(lengths.values())) < self.seq_len:
            raise ValueError(f"episode shorter than seq_len={self.seq_len}: {lengths}")
        self._episodes.append({k: np.asarray(v, dtype=np.float32) for k, v in episode.items()})
        if len(self._episodes) > self.capacity:
            self._episodes.pop(0)

    def sample(self, batch: int, gen: torch.Generator) -> dict:
        eps = torch.randint(0, len(self._episodes), (batch,), generator=gen).numpy()
        out = None
        for i, e in enumerate(eps):
            ep = self._episodes[e]
            T = len(next(iter(ep.values())))
            start = int(torch.randint(0, T - self.seq_len + 1, (1,), generator=gen))
            window = {k: v[start : start + self.seq_len] for k, v in ep.items()}
            if out is None:
                out = {k: np.zeros((batch, *v.shape), dtype=np.float32) for k, v in window.items()}
            for k, v in window.items():
                out[k][i] = v
        return {k: torch.from_numpy(v) for k, v in out.items()}
