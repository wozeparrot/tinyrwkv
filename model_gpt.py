import tinygrad.nn as nn
from tinygrad.tensor import Tensor
import numpy as np

import math


class Embedding:
    def __init__(self, vocab_size, embed_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.weight = Tensor.uniform(vocab_size, embed_size)

    def __call__(self, idx):
        idxnp = idx.numpy()
        x = Tensor.cat(
            *[
                self.weight[int(idxnp[0, i])].reshape((1, self.embed_size))
                for i in range(idx.shape[1])
            ],
        )
        return x.reshape((1, idx.shape[1], self.embed_size))


class RWKV_GPT:
    def __init__(self, ctx_size, vocab_size, embed_size, layers):
        self.ctx_size = ctx_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.emb = Embedding(vocab_size, embed_size)

        self.blocks = []
        for i in range(layers):
            self.blocks.append(Block(i, ctx_size, embed_size, layers))

        self.ln_out = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, idx):
        x = self.emb(idx)

        x = x.sequential(self.blocks)

        x = self.ln_out(x)

        x = self.head(x)

        return x


class Block:
    def __init__(self, i, ctx_size, embed_size, layers):
        self.i = i

        if i == 0:
            self.ln0 = nn.LayerNorm(embed_size)

        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

        self.att = TimeMix(i, embed_size, layers)
        self.ffn = ChannelMix(i, embed_size, layers)

    def __call__(self, x):
        if self.i == 0:
            x = self.ln0(x)

        ln1x = self.ln1(x)
        x += self.att(ln1x)

        ln2x = self.ln2(x)
        x += self.ffn(ln2x)

        return x


class ChannelMix:
    def __init__(self, i, embed_size, layers):
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

    def __call__(self, x):
        # time shift
        xx = x[:, -1:-1, :]
        xk = self.time_mix_k * (x - xx) + xx
        xr = self.time_mix_r * (x - xx) + xx

        k = self.key(xk)
        k = k.relu().square()
        kv = self.value(k)

        rkv = self.receptance(xr).sigmoid() * kv
        return rkv


class WKV:
    def __init__(self):
        pass

    # really need to rewrite this with only tinygrad ops
    def __call__(self, B, T, C, w, u, k, ov):
        w = w.numpy()
        u = u.numpy()
        k = k.numpy()
        v = ov.numpy()

        w = -np.exp(w)
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

        oo = np.copy(k)
        pp = np.copy(v)
        qq = np.ones((T, B, C))
        dd = np.ones((T, 1, 1))
        for ss, sa, sb, sz in sl:
            p = pp[sb:sz:ss]
            q = qq[sb:sz:ss]
            d = dd[sb:sz:ss]
            o = oo[sb:sz:ss]
            e = oo[sa:sz:ss] + d * w
            x = np.maximum(e, o)
            a = np.exp(e - x)
            b = np.exp(o - x)
            p[:] = a * pp[sa:sz:ss] + b * p
            q[:] = a * qq[sa:sz:ss] + b * q
            d[:] = dd[sa:sz:ss] + d
            o[:] = x

        p = np.roll(pp, 1, axis=0)
        q = np.roll(qq, 1, axis=0)
        o = np.roll(oo, 1, axis=0)

        x = np.maximum(o, k + u)
        a = np.exp(o - x)
        b = np.exp(k + u - x)
        y = (a * p + b * v) / (a * q + b)
        y = Tensor.cat(Tensor(v[:1, :, :]), Tensor(y[1:, :, :]))
        y = y.transpose((1, 0, 2))

        return y


class TimeMix:
    def __init__(self, i, embed_size, layers):
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

    def __call__(self, x):
        # time shift
        xx = x[:, -1:-1, :]
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
