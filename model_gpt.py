from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
from typing import Callable, cast
import numpy as np
import tinygrad.nn as nn

import math


class Embedding:
    vocab_size: int
    embed_size: int
    weight: Tensor

    def __init__(self, vocab_size: int, embed_size: int):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.weight = Tensor.scaled_uniform(vocab_size, embed_size)

    def __call__(self, idx: Tensor) -> Tensor:
        idxnp = idx.numpy().astype(np.int32)
        onehot = np.zeros(
            (idx.shape[0], idx.shape[1], self.vocab_size), dtype=np.float32
        )
        for i in range(idx.shape[0]):
            onehot[i, np.arange(idx.shape[1]), idxnp[i]] = 1

        return Tensor(onehot) @ self.weight


class ChannelMix:
    time_mix_k: Tensor
    time_mix_r: Tensor

    key: nn.Linear
    receptance: nn.Linear
    value: nn.Linear

    def __init__(self, i: int, embed_size: int, layers: int):
        ratio_1_to_almost_0 = 1.0 - (i / layers)

        x = np.ones((1, 1, embed_size), dtype=np.float32)
        for i in range(embed_size):
            x[0, 0, i] = i / embed_size

        self.time_mix_k = Tensor(pow(x, ratio_1_to_almost_0))
        self.time_mix_r = Tensor(pow(x, ratio_1_to_almost_0))

        hidden_sz = 4 * embed_size
        self.key = nn.Linear(embed_size, hidden_sz, bias=False)
        self.receptance = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(hidden_sz, embed_size, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        # time shift
        xx = (
            x[:, :-1, :]
            .reshape((1, x.shape[0], x.shape[1] - 1, x.shape[2]))
            .pad2d((0, 0, 1, 0))
            .reshape((x.shape[0], x.shape[1], x.shape[2]))
        )
        xk = self.time_mix_k * (x - xx) + xx
        xr = self.time_mix_r * (x - xx) + xx

        k = self.key(xk)
        k = k.relu().square()
        kv = self.value(k)

        rkv = self.receptance(xr).sigmoid() * kv
        return rkv


class WKV:
    def __call__(
        self,
        T: int,
        C: int,
        time_first: Tensor,
        time_decay: Tensor,
        key: Tensor,
        value: Tensor,
        time_curve: Tensor,
    ) -> Tensor:
        ek = key.clip(-60, 60).transpose(1, 2).exp()
        ekv = ek * value.transpose(1, 2)

        time_w = (time_first.exp().unsqueeze(1) * time_curve).cat(
            time_decay.unsqueeze(1), dim=-1
        )
        w = time_w.exp().unsqueeze(1)
        w = w.reshape(w.shape[0], w.shape[1], w.shape[2], 1)

        ekv = (
            ekv.reshape(1, *ekv.shape)
            .pad2d((T - 1, 0, 0, 0))
            .reshape(ekv.shape[0], ekv.shape[1], ekv.shape[2] + T - 1, 1)
        )
        wkv = ekv.conv2d(w, groups=C).reshape(
            ekv.shape[0], ekv.shape[1], ekv.shape[2] - T + 1
        )
        ek = (
            ek.reshape(1, *ek.shape)
            .pad2d((T - 1, 0, 0, 0))
            .reshape(ek.shape[0], ek.shape[1], ek.shape[2] + T - 1, 1)
        )
        wk = (
            ek.conv2d(w, groups=C).reshape(
                ek.shape[0], ek.shape[1], ek.shape[2] - T + 1
            )
            + 1e-8
        )

        wkv = (wkv / wk).transpose(1, 2)

        return wkv


class TimeMix:
    embed_size: int

    time_decay: Tensor

    time_first: Tensor

    time_mix_k: Tensor
    time_mix_v: Tensor
    time_mix_r: Tensor

    key: nn.Linear
    receptance: nn.Linear
    value: nn.Linear

    wkv: WKV

    output: nn.Linear

    def __init__(self, i: int, embed_size: int, layers: int):
        self.embed_size = embed_size

        ratio_0_to_1 = i / (layers - 1)
        ratio_1_to_almost_0 = 1.0 - (i / layers)

        decay_speed = np.array([0.0] * embed_size, dtype=np.float32)
        for h in range(embed_size):
            decay_speed[h] = -5 + 8 * (h / (embed_size - 1)) ** (
                0.7 + 1.3 * ratio_0_to_1
            )
        self.time_decay = Tensor(decay_speed)

        zigzag = np.array(
            [((i + 1) % 3 - 1) * 0.5 for i in range(embed_size)], dtype=np.float32
        )
        self.time_first = Tensor(
            np.array([math.log(0.3)] * embed_size, dtype=np.float32) + zigzag
        )

        x = np.ones((1, 1, embed_size), dtype=np.float32)
        for i in range(embed_size):
            x[0, 0, i] = i / embed_size
        self.time_mix_k = Tensor(pow(x, ratio_1_to_almost_0))
        self.time_mix_v = Tensor(pow(x, ratio_1_to_almost_0) + 0.3 * ratio_0_to_1)
        self.time_mix_r = Tensor(pow(x, 0.5 * ratio_1_to_almost_0))

        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.receptance = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)

        self.wkv = WKV()

        self.output = nn.Linear(embed_size, embed_size, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        # time shift
        xx = (
            x[:, :-1, :]
            .reshape((1, x.shape[0], x.shape[1] - 1, x.shape[2]))
            .pad2d((0, 0, 1, 0))
            .reshape((x.shape[0], x.shape[1], x.shape[2]))
        )
        xk = self.time_mix_k * (x - xx) + xx
        xv = self.time_mix_v * (x - xx) + xx
        xr = self.time_mix_r * (x - xx) + xx

        k = self.key(xk)
        r = self.receptance(xr).sigmoid()
        v = self.value(xv)

        _, T, C = x.shape
        time_curve = Tensor([-(T - 2 - i) for i in range(T - 1)], requires_grad=False)
        rwkv = r * self.wkv(T, C, self.time_decay, self.time_first, k, v, time_curve)

        rwkv = self.output(rwkv)
        return rwkv


class Block:
    i: int

    ln0: nn.LayerNorm | None = None

    ln1: nn.LayerNorm
    ln2: nn.LayerNorm

    att: TimeMix
    ffn: ChannelMix

    def __init__(self, i: int, embed_size: int, layers: int):
        self.i = i

        if i == 0:
            self.ln0 = nn.LayerNorm(embed_size)

        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

        self.att = TimeMix(i, embed_size, layers)
        self.ffn = ChannelMix(i, embed_size, layers)

    def __call__(self, x: Tensor) -> Tensor:
        if self.i == 0:
            x = cast(nn.LayerNorm, self.ln0)(x)

        ln1x = self.ln1(x)
        x = x + self.att(ln1x)

        ln2x = self.ln2(x)
        x = x + self.ffn(ln2x)

        return x.realize()


class RWKV_GPT:
    ctx_size: int
    vocab_size: int
    embed_size: int

    emb: Embedding

    blocks: list[Block]

    ln_out: nn.LayerNorm
    head: nn.Linear

    def __init__(self, ctx_size: int, vocab_size: int, embed_size: int, layers: int):
        self.ctx_size = ctx_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.emb = Embedding(vocab_size, embed_size)

        self.blocks = []
        for i in range(layers):
            self.blocks.append(Block(i, embed_size, layers))

        self.ln_out = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        x = self.emb(idx)

        x = x.sequential(cast(list[Callable[[Tensor], Tensor]], self.blocks))

        x = self.ln_out(x)

        x = self.head(x)

        return x.realize()
