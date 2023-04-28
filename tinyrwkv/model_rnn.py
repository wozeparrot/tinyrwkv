from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
import tinygrad.nn as nn
from tqdm import tqdm

import gc
import json
import pickle

from utils.tensor import matvec, elemmax


class Att:
    time_mix_k: Tensor
    time_mix_v: Tensor
    time_mix_r: Tensor
    key: Tensor
    value: Tensor
    receptance: Tensor
    time_first: Tensor
    time_decay: Tensor
    output: Tensor

    def __init__(
        self,
        time_mix_k: Tensor,
        time_mix_v: Tensor,
        time_mix_r: Tensor,
        key: Tensor,
        value: Tensor,
        receptance: Tensor,
        time_first: Tensor,
        time_decay: Tensor,
        output: Tensor,
    ):
        self.time_mix_k = time_mix_k
        self.time_mix_v = time_mix_v
        self.time_mix_r = time_mix_r
        self.key = key
        self.value = value
        self.receptance = receptance
        self.time_first = time_first
        self.time_decay = time_decay
        self.output = output

    def __call__(
        self, x: Tensor, att_xx, att_aa, att_bb, att_pp
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        xk = self.time_mix_k * (x - att_xx) + att_xx
        xv = self.time_mix_v * (x - att_xx) + att_xx
        xr = self.time_mix_r * (x - att_xx) + att_xx

        k = matvec(self.key, xk)
        v = matvec(self.value, xv)
        r = matvec(self.receptance, xr).sigmoid()

        # calculate output
        ww = k + self.time_first

        p = elemmax(att_pp, ww)
        e1 = (att_pp - p).exp()
        e2 = (ww - p).exp()
        a = (e1 * att_aa) + (e2 * v)
        b = (e1 * att_bb) + e2
        rwkv = r * (a / b)

        # update state
        ww = att_pp + self.time_decay
        p = elemmax(ww, k)
        e1 = (ww - p).exp()
        e2 = (k - p).exp()

        return (
            matvec(self.output, rwkv),
            x,
            ((e1 * att_aa) + (e2 * v)),
            ((e1 * att_bb) + e2),
            p,
        )


class Ffn:
    time_mix_k: Tensor
    time_mix_r: Tensor
    key: Tensor
    value: Tensor
    receptance: Tensor

    def __init__(
        self,
        time_mix_k: Tensor,
        time_mix_r: Tensor,
        key: Tensor,
        value: Tensor,
        receptance: Tensor,
    ):
        self.time_mix_k = time_mix_k
        self.time_mix_r = time_mix_r
        self.key = key
        self.value = value
        self.receptance = receptance

    def __call__(self, x: Tensor, ffn_xx: Tensor) -> tuple[Tensor, Tensor]:
        xk = self.time_mix_k * (x - ffn_xx) + ffn_xx
        xr = self.time_mix_r * (x - ffn_xx) + ffn_xx

        k = matvec(self.key, xk).relu().square()
        kv = matvec(self.value, k)
        r = matvec(self.receptance, xr).sigmoid()
        rkv = r * kv

        return rkv, x


class Block:
    embed_size: int

    att_ln: nn.LayerNorm
    att: Att
    ffn_ln: nn.LayerNorm
    ffn: Ffn

    def __init__(
        self,
        embed_size: int,
        att_ln_weight: Tensor,
        att_ln_bias: Tensor,
        att_time_mix_k: Tensor,
        att_time_mix_v: Tensor,
        att_time_mix_r: Tensor,
        att_key: Tensor,
        att_value: Tensor,
        att_receptance: Tensor,
        att_time_first: Tensor,
        att_time_decay: Tensor,
        att_output: Tensor,
        ffn_ln_weight: Tensor,
        ffn_ln_bias: Tensor,
        ffn_time_mix_k: Tensor,
        ffn_time_mix_r: Tensor,
        ffn_key: Tensor,
        ffn_value: Tensor,
        ffn_receptance: Tensor,
    ):
        self.embed_size = embed_size

        self.att_ln = nn.LayerNorm(embed_size)
        self.att_ln.weight.assign(att_ln_weight)
        self.att_ln.bias.assign(att_ln_bias)
        self.att = Att(
            att_time_mix_k,
            att_time_mix_v,
            att_time_mix_r,
            att_key,
            att_value,
            att_receptance,
            att_time_first,
            att_time_decay,
            att_output,
        )

        self.ffn_ln = nn.LayerNorm(embed_size)
        self.ffn_ln.weight.assign(ffn_ln_weight)
        self.ffn_ln.bias.assign(ffn_ln_bias)
        self.ffn = Ffn(
            ffn_time_mix_k,
            ffn_time_mix_r,
            ffn_key,
            ffn_value,
            ffn_receptance,
        )

    def __call__(
        self,
        x: Tensor,
        att_xx: Tensor,
        att_aa: Tensor,
        att_bb: Tensor,
        att_pp: Tensor,
        ffn_xx: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        ln1 = self.att_ln(x)
        att, att_xx, att_aa, att_bb, att_pp = self.att(
            ln1, att_xx, att_aa, att_bb, att_pp
        )
        x = x + att
        ln2 = self.ffn_ln(x)
        ffn, ffn_xx = self.ffn(ln2, ffn_xx)
        x = x + ffn

        return (
            x,
            att_xx.realize(),
            att_aa.realize(),
            att_bb.realize(),
            att_pp.realize(),
            ffn_xx.realize(),
        )


class RWKV_RNN:
    vocab_size: int
    embed_size: int
    layers: int
    dtype: str

    emb: Tensor
    blocks: list[Block]
    ln_out_weight: Tensor
    ln_out_bias: Tensor
    head: Tensor

    def __init__(self, path: str):
        # load info file
        with open(path + ".json", "r") as f:
            info = json.load(f)

        self.vocab_size = info["vocab_size"]
        self.embed_size = info["embed_size"]
        self.layers = info["layers"]
        self.dtype = info["dtype"]

        with open(path, "rb") as f:
            weights = pickle.load(f)
        tg_weights = {}
        for k, v in tqdm(weights.items()):
            tg_weights[k] = Tensor(v)

        self.emb = tg_weights["emb.weight"]

        self.blocks = []
        for i in range(self.layers):
            self.blocks.append(
                Block(
                    self.embed_size,
                    tg_weights[f"blocks.{i}.ln1.weight"],
                    tg_weights[f"blocks.{i}.ln1.bias"],
                    tg_weights[f"blocks.{i}.att.time_mix_k"],
                    tg_weights[f"blocks.{i}.att.time_mix_v"],
                    tg_weights[f"blocks.{i}.att.time_mix_r"],
                    tg_weights[f"blocks.{i}.att.key.weight"],
                    tg_weights[f"blocks.{i}.att.value.weight"],
                    tg_weights[f"blocks.{i}.att.receptance.weight"],
                    tg_weights[f"blocks.{i}.att.time_first"],
                    tg_weights[f"blocks.{i}.att.time_decay"],
                    tg_weights[f"blocks.{i}.att.output.weight"],
                    tg_weights[f"blocks.{i}.ln2.weight"],
                    tg_weights[f"blocks.{i}.ln2.bias"],
                    tg_weights[f"blocks.{i}.ffn.time_mix_k"],
                    tg_weights[f"blocks.{i}.ffn.time_mix_r"],
                    tg_weights[f"blocks.{i}.ffn.key.weight"],
                    tg_weights[f"blocks.{i}.ffn.value.weight"],
                    tg_weights[f"blocks.{i}.ffn.receptance.weight"],
                )
            )

        self.ln_out_weight = tg_weights["ln_out.weight"]
        self.ln_out_bias = tg_weights["ln_out.bias"]

        self.head = tg_weights["head.weight"]

        gc.collect()

    def init_state(self) -> Tensor:
        states = []
        for _ in range(self.layers):
            states.append(
                Tensor.cat(
                    Tensor([0.0] * self.embed_size),
                    Tensor([0.0] * self.embed_size),
                    Tensor([0.0] * self.embed_size),
                    Tensor([-1e30] * self.embed_size),
                    Tensor([0.0] * self.embed_size),
                )
            )
        return Tensor.cat(*states).realize()

    def index_embed(self, ctx: int) -> Tensor:
        return self.emb[ctx]

    def build_input(self, ctx: Tensor, state: Tensor) -> Tensor:
        return Tensor.cat(ctx, state).realize()

    @TinyJit
    def forward(
        self,
        ctx: Tensor,
    ) -> Tensor:
        x = ctx[: self.embed_size]
        state = ctx[self.embed_size :]

        new_state = []
        for i, block in enumerate(self.blocks):
            x, att_xx, att_aa, att_bb, att_pp, ffn_xx = block(
                x,
                state[
                    i * 5 * self.embed_size
                    + 0 * self.embed_size : i * 5 * self.embed_size
                    + 1 * self.embed_size
                ],
                state[
                    i * 5 * self.embed_size
                    + 1 * self.embed_size : i * 5 * self.embed_size
                    + 2 * self.embed_size
                ],
                state[
                    i * 5 * self.embed_size
                    + 2 * self.embed_size : i * 5 * self.embed_size
                    + 3 * self.embed_size
                ],
                state[
                    i * 5 * self.embed_size
                    + 3 * self.embed_size : i * 5 * self.embed_size
                    + 4 * self.embed_size
                ],
                state[
                    i * 5 * self.embed_size
                    + 4 * self.embed_size : i * 5 * self.embed_size
                    + 5 * self.embed_size
                ],
            )

            new_state.append(Tensor.cat(att_xx, att_aa, att_bb, att_pp, ffn_xx))

        state = Tensor.cat(*new_state)

        x = x.layernorm().linear(self.ln_out_weight, self.ln_out_bias)
        x = matvec(self.head, x)

        return Tensor.cat(x, state).realize()
