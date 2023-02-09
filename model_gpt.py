from typing import Callable, cast
import tinygrad.nn as nn
from tinygrad.tensor import Tensor
import numpy as np

import math

from utils import matvec, elemmax


class Embedding:
    vocab_size: int
    embed_size: int
    weight: Tensor

    def __init__(self, vocab_size: int, embed_size: int):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.weight = Tensor.uniform(vocab_size, embed_size)

    def __call__(self, idx: Tensor) -> Tensor:
        idxnp = idx.numpy()
        x = self.weight[int(idxnp[0, 0])].reshape((1, 1, self.embed_size))
        for i in range(1, idx.shape[1]):
            y = self.weight[int(idxnp[0, i])].reshape((1, 1, self.embed_size))
            x = x.cat(y, dim=1)
        return x


class ChannelMix:
    time_mix_k: Tensor
    time_mix_r: Tensor

    key: nn.Linear
    receptance: nn.Linear
    value: nn.Linear

    def __init__(self, i: int, embed_size: int, layers: int):
        ratio_1_to_almost_0 = 1.0 - (i / layers)

        x = np.ones((1, 1, embed_size))
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
        B: int,
        T: int,
        C: int,
        time_first: Tensor,
        time_decay: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        time_first = -(time_first.exp())
        k = k.transpose((1, 0, 2))
        v = v.transpose((1, 0, 2))
        sl = []
        s = 2
        while s <= T:
            sl += [(s, (s >> 1) - 1, s - 1, T - T % s)]
            s = s << 1
        s = s >> 1
        while s >= 2:
            sl += [(s, s - 1, (s >> 1) * 3 - 1, T - (T % s < (s >> 1)) * (s >> 1))]
            s = s >> 1

        # only section that is still numpy
        oo = k.detach().numpy()
        pp = v.detach().numpy()
        qq = np.ones((T, B, C))
        dd = np.ones((T, 1, 1))
        for ss, sa, sb, sz in sl:
            p = pp[sb:sz:ss]
            q = qq[sb:sz:ss]
            d = dd[sb:sz:ss]
            o = oo[sb:sz:ss]

            e = oo[sa:sz:ss] + d * time_first.numpy()

            x = np.maximum(e, o)
            a = np.exp(e - x)
            b = np.exp(o - x)

            p[:] = a * pp[sa:sz:ss] + b * p
            q[:] = a * qq[sa:sz:ss] + b * q
            d[:] = dd[sa:sz:ss] + d
            o[:] = x

        pt = Tensor(pp)
        p = pt[-1:, :, :].cat(pt[:-1, :, :], dim=0)
        pq = Tensor(qq)
        q = pq[-1:, :, :].cat(pq[:-1, :, :], dim=0)
        po = Tensor(oo)
        o = po[-1:, :, :].cat(po[:-1, :, :], dim=0)

        x = elemmax(o, k + time_decay)
        a = (o - x).exp()
        b = (k + time_decay - x).exp()
        y = (a * p + b * v) / (a * q + b)
        y = v[:1, :, :].cat(y[1:, :, :])
        y = y.transpose((1, 0, 2))
        return y


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

        decay_speed = np.array([0.0] * embed_size)
        for h in range(embed_size):
            decay_speed[h] = -5 + 8 * (h / (embed_size - 1)) ** (
                0.7 + 1.3 * ratio_0_to_1
            )
        self.time_decay = Tensor(decay_speed)

        zigzag = np.array([((i + 1) % 3 - 1) * 0.5 for i in range(embed_size)])
        self.time_first = Tensor(np.array([math.log(0.3)] * embed_size) + zigzag)

        x = np.ones((1, 1, embed_size))
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

        B, T, C = x.shape
        rwkv = r * self.wkv(B, T, C, self.time_decay, self.time_first, k, v)

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
        x += self.att(ln1x)

        ln2x = self.ln2(x)
        x += self.ffn(ln2x)

        return x


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
