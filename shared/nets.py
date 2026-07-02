"""The only net primitives in the repo: mlp, conv_encoder, conv_decoder, ResidualBlock.

Rule (see export/js/forward.js contract): this file may not gain a primitive without its JS
twin. Plain conv/linear/norm/activation only; no exotic ops (MPS coverage + browser export).
"""
import torch
import torch.nn as nn

ACTS = {"relu": nn.ReLU, "elu": nn.ELU, "silu": nn.SiLU, "tanh": nn.Tanh}


def mlp(in_dim: int, hidden: list, out_dim: int, act: str = "relu", out_act: str = None) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), ACTS[act]()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    if out_act:
        layers.append(ACTS[out_act]())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, act: str = "relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = ACTS[act]()

    def forward(self, x):
        h = self.act(self.conv1(x))
        return self.act(x + self.conv2(h))


def conv_encoder(in_ch: int = 3, base: int = 32, depth: int = 4, act: str = "relu") -> nn.Sequential:
    """Stride-2 conv stack: 64x64 input -> (base*2^(depth-1)) x 4 x 4 at depth 4."""
    layers, ch = [], in_ch
    for i in range(depth):
        out = base * (2**i)
        layers += [nn.Conv2d(ch, out, 4, stride=2, padding=1), ACTS[act]()]
        ch = out
    return nn.Sequential(*layers)


def conv_decoder(in_dim: int, out_ch: int = 3, base: int = 32, depth: int = 4, act: str = "relu") -> nn.Sequential:
    """Mirror of conv_encoder: latent vector -> (out_ch, 64, 64) at depth 4."""
    ch0 = base * (2 ** (depth - 1))
    layers = [nn.Linear(in_dim, ch0 * 4 * 4), nn.Unflatten(1, (ch0, 4, 4))]
    ch = ch0
    for i in range(depth - 1, 0, -1):
        out = base * (2 ** (i - 1))
        layers += [nn.ConvTranspose2d(ch, out, 4, stride=2, padding=1), ACTS[act]()]
        ch = out
    layers.append(nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1))
    return nn.Sequential(*layers)
