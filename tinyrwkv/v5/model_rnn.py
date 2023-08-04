from tinygrad.jit import TinyJit
from tinygrad.lazy import Device
from tinygrad.state import safe_load
from tinygrad.tensor import Tensor
import tinygrad.nn as nn

from typing import cast
import gc
import json


class Att:
    n_heads: int
    head_dim: int
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
        n_heads: int,
        head_dim: int,
        time_mix_k: Tensor,
        time_mix_v: Tensor,
        time_mix_r: Tensor,
        key: Tensor,
        value: Tensor,
        receptance: Tensor,
        time_first: Tensor,
        time_decay: Tensor,
        gn_weight: Tensor,
        gn_bias: Tensor,
        output: Tensor,
    ):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.time_mix_k = time_mix_k
        self.time_mix_v = time_mix_v
        self.time_mix_r = time_mix_r
        self.key = key
        self.value = value
        self.receptance = receptance
        self.time_first = time_first
        self.time_decay = time_decay
        self.output = output

        self.output_group_norm = nn.GroupNorm(self.n_heads, self.head_dim)
        self.output_group_norm.weight = gn_weight
        self.output_group_norm.bias = gn_bias

    def __call__(
        self,
        x: Tensor,
        att_xx,
        att_ss,
    ) -> tuple[Tensor, Tensor, Tensor]:
        xk = self.time_mix_k * (x - att_xx) + att_xx
        xv = self.time_mix_v * (x - att_xx) + att_xx
        xr = self.time_mix_r * (x - att_xx) + att_xx

        k = (self.key @ xk).reshape(self.n_heads, self.head_dim, 1)
        v = (self.value @ xv).reshape(self.n_heads, 1, self.head_dim)
        r = (self.receptance @ xr).reshape(self.n_heads, 1, self.head_dim)

        ss = att_ss.reshape(self.n_heads, self.head_dim, self.head_dim)

        a = k @ v
        o = r @ (self.time_first * a + ss)
        o = self.output_group_norm(o.flatten().unsqueeze(0)).squeeze(0)

        return (
            self.output @ o,
            x,
            (a + self.time_decay * ss).flatten(),
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

        k = (self.key @ xk).relu().square()
        kv = self.value @ k
        r = (self.receptance @ xr).sigmoid()
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
        n_heads: int,
        head_dim: int,
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
        att_gn_weight: Tensor,
        att_gn_bias: Tensor,
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
            n_heads,
            head_dim,
            att_time_mix_k,
            att_time_mix_v,
            att_time_mix_r,
            att_key,
            att_value,
            att_receptance,
            att_time_first,
            att_time_decay,
            att_gn_weight,
            att_gn_bias,
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
        att_ss: Tensor,
        ffn_xx: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ln1 = self.att_ln(x)
        att, att_xx, att_ss = self.att(ln1, att_xx, att_ss)
        x = x + att
        ln2 = self.ffn_ln(x)
        ffn, ffn_xx = self.ffn(ln2, ffn_xx)
        x = x + ffn

        return (
            x,
            att_xx.realize(),
            att_ss.realize(),
            ffn_xx.realize(),
        )


class RWKV_RNN:
    vocab_size: int
    embed_size: int
    n_heads: int
    head_dim: int
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
        assert info["version"] == "v5", "model version mismatch"

        self.vocab_size = info["vocab_size"]
        self.embed_size = info["embed_size"]
        self.n_heads = info["n_heads"]
        self.head_dim = info["head_dim"]
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
                    self.n_heads,
                    self.head_dim,
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
                    weights[f"blocks.{i}.att.ln_x.weight"],
                    weights[f"blocks.{i}.att.ln_x.bias"],
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
                ]
                * (2 + self.head_dim)
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
            state_index = i * (2 + self.head_dim) * self.embed_size
            state_xx_ss = state_index + 1 * self.embed_size
            state_ss_xx = state_index + (1 + self.head_dim) * self.embed_size
            state_end = state_index + (2 + self.head_dim) * self.embed_size
            x, att_xx, att_ss, ffn_xx = block(
                x,
                state[state_index:state_xx_ss],
                state[state_xx_ss:state_ss_xx],
                state[state_ss_xx:state_end],
            )

            new_state.extend([att_xx, att_ss, ffn_xx])

        x = x.layernorm().linear(self.ln_out_weight, self.ln_out_bias)
        x = self.head @ x

        return Tensor.cat(x, *new_state).realize()
