from tinygrad.tensor import Tensor
from tqdm import tqdm
from typing import Callable, cast
import numpy as np
import tinygrad.nn as nn

import json
import math
import pickle

from utils.misc import get_child
from wkv import WKV, ConvWKV, OpenCLWKV


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
        xx = x.slice([None, (-1, x.shape[1] - 1), None])
        xk = self.time_mix_k * (x - xx) + xx
        xr = self.time_mix_r * (x - xx) + xx

        k = self.key(xk)
        k = k.relu().square()
        kv = self.value(k)

        rkv = self.receptance(xr).sigmoid() * kv
        return rkv


class TimeMix:
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

        # self.wkv = ConvWKV()
        self.wkv = OpenCLWKV()

        self.output = nn.Linear(embed_size, embed_size, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        # time shift
        xx = x.slice([None, (-1, x.shape[1] - 1), None])
        xk = self.time_mix_k * (x - xx) + xx
        xv = self.time_mix_v * (x - xx) + xx
        xr = self.time_mix_r * (x - xx) + xx

        k = self.key(xk)
        r = self.receptance(xr).sigmoid()
        v = self.value(xv)

        B, T, C = x.shape
        rwkv = r * self.wkv(B, T, C, self.time_first, self.time_decay, k, v)

        rwkv = self.output(rwkv)
        return rwkv


class Block:
    i: int

    ln1: nn.LayerNorm
    ln2: nn.LayerNorm

    att: TimeMix
    ffn: ChannelMix

    def __init__(self, i: int, embed_size: int, layers: int):
        self.i = i

        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

        self.att = TimeMix(i, embed_size, layers)
        self.ffn = ChannelMix(i, embed_size, layers)

    def __call__(self, x: Tensor) -> Tensor:
        ln1x = self.ln1(x)
        x = x + self.att(ln1x)

        ln2x = self.ln2(x)
        x = x + self.ffn(ln2x)

        return x.realize()


class RWKV_GPT:
    vocab_size: int
    embed_size: int
    dtype: str

    emb: nn.Embedding

    blocks: list[Block]

    ln_out: nn.LayerNorm
    head: nn.Linear

    def __init__(self, path: str):
        # load info file
        with open(path + ".json", "r") as f:
            info = json.load(f)

        self.vocab_size = info["vocab_size"]
        self.embed_size = info["embed_size"]
        self.layers = info["layers"]
        self.dtype = info["dtype"]

        # setup model
        self.emb = nn.Embedding(self.vocab_size, self.embed_size)

        self.ln0 = nn.LayerNorm(self.embed_size)

        self.blocks = []
        for i in range(self.layers):
            self.blocks.append(Block(i, self.embed_size, self.layers))

        self.ln_out = nn.LayerNorm(self.embed_size)
        self.head = nn.Linear(self.embed_size, self.vocab_size, bias=False)

        # load weights
        with open(path, "rb") as f:
            weights = pickle.load(f)

            for k, v in tqdm(weights.items()):
                if "ln0" in k:
                    if "weight" in k:
                        cast(Tensor, self.ln0.weight).assign(v)
                    elif "bias" in k:
                        cast(Tensor, self.ln0.bias).assign(v)

                try:
                    w = cast(Tensor, get_child(self, k))
                except:
                    w = None
                if w is not None:
                    assert w.shape == v.shape
                    w.assign(v)

            del weights

    def forward(self, idx: Tensor) -> Tensor:
        x = self.emb(idx)

        x = self.ln0(x)

        x = x.sequential(cast(list[Callable[[Tensor], Tensor]], self.blocks))

        x = self.ln_out(x)

        x = self.head(x)

        return x.realize()
