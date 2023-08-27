from tinygrad.jit import TinyJit
from tinygrad.lazy import Device
from tinygrad.nn.state import safe_load
from tinygrad.tensor import Tensor
import tinygrad.nn as nn

from typing import cast
import gc
import json


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

        k = xk @ self.key.T
        v = xv @ self.value.T
        r = (xr @ self.receptance.T).sigmoid()

        # calculate output
        ww = k + self.time_first
        p = att_pp.maximum(ww)
        e1 = (att_pp - p).exp()
        e2 = (ww - p).exp()
        a = (e1 * att_aa) + (e2 * v)
        b = (e1 * att_bb) + e2
        rwkv = r * (a / b)

        # update state
        ww = att_pp + self.time_decay
        p = ww.maximum(k)
        e1 = (ww - p).exp()
        e2 = (k - p).exp()

        return (
            rwkv @ self.output.T,
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

        k = (xk @ self.key.T).relu().square()
        kv = k @ self.value.T
        r = (xr @ self.receptance.T).sigmoid()
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
        cast(Tensor, self.att_ln.weight).assign(att_ln_weight)
        cast(Tensor, self.att_ln.bias).assign(att_ln_bias)
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
        cast(Tensor, self.ffn_ln.weight).assign(ffn_ln_weight)
        cast(Tensor, self.ffn_ln.bias).assign(ffn_ln_bias)
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
    model_type: str

    emb: Tensor
    blocks: list[Block]
    ln_out_weight: Tensor
    ln_out_bias: Tensor
    head: Tensor

    def __init__(self, path: str):
        # load info file
        with open(path + ".json", "r") as f:
            info = json.load(f)
        assert info["version"] == "v4", "model version mismatch"

        self.vocab_size = info["vocab_size"]
        self.embed_size = info["embed_size"]
        self.layers = info["layers"]
        self.dtype = info["dtype"]
        self.model_type = info["model_type"]

        # load weights
        weights = safe_load(path)
        for k, v in weights.items():
            weights[k] = v.to(Device.DEFAULT).realize()

        self.emb = weights["emb.weight"]

        self.blocks = []
        for i in range(self.layers):
            self.blocks.append(
                Block(
                    self.embed_size,
                    weights[f"blocks.{i}.ln1.weight"],
                    weights[f"blocks.{i}.ln1.bias"],
                    weights[f"blocks.{i}.att.time_mix_k"],
                    weights[f"blocks.{i}.att.time_mix_v"],
                    weights[f"blocks.{i}.att.time_mix_r"],
                    weights[f"blocks.{i}.att.key.weight"],
                    weights[f"blocks.{i}.att.value.weight"],
                    weights[f"blocks.{i}.att.receptance.weight"],
                    weights[f"blocks.{i}.att.time_first"],
                    weights[f"blocks.{i}.att.time_decay"],
                    weights[f"blocks.{i}.att.output.weight"],
                    weights[f"blocks.{i}.ln2.weight"],
                    weights[f"blocks.{i}.ln2.bias"],
                    weights[f"blocks.{i}.ffn.time_mix_k"],
                    weights[f"blocks.{i}.ffn.time_mix_r"],
                    weights[f"blocks.{i}.ffn.key.weight"],
                    weights[f"blocks.{i}.ffn.value.weight"],
                    weights[f"blocks.{i}.ffn.receptance.weight"],
                )
            )

        self.ln_out_weight = weights["ln_out.weight"]
        self.ln_out_bias = weights["ln_out.bias"]

        self.head = weights["head.weight"]

        gc.collect()

    def init_state(self) -> Tensor:
        states = []
        for _ in range(self.layers):
            states.extend(
                [
                    Tensor([0.0] * self.embed_size),
                    Tensor([0.0] * self.embed_size),
                    Tensor([0.0] * self.embed_size),
                    Tensor([-1e30] * self.embed_size),
                    Tensor([0.0] * self.embed_size),
                ]
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
            state_index = i * 5 * self.embed_size
            state_xx_aa = state_index + 1 * self.embed_size
            state_aa_bb = state_index + 2 * self.embed_size
            state_bb_pp = state_index + 3 * self.embed_size
            state_pp_xx = state_index + 4 * self.embed_size
            state_end = state_index + 5 * self.embed_size
            x, att_xx, att_aa, att_bb, att_pp, ffn_xx = block(
                x,
                state[state_index:state_xx_aa],
                state[state_xx_aa:state_aa_bb],
                state[state_aa_bb:state_bb_pp],
                state[state_bb_pp:state_pp_xx],
                state[state_pp_xx:state_end],
            )

            new_state.extend([att_xx, att_aa, att_bb, att_pp, ffn_xx])

        x = x.layernorm().linear(self.ln_out_weight, self.ln_out_bias)
        x = x @ self.head.T

        return Tensor.cat(x, *new_state).realize()
