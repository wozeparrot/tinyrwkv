from tinygrad.tensor import Tensor
import tinygrad.nn as nn
import pickle
from tqdm import tqdm

import gc

from utils import matvec, elemmax


class LayerState:
    ffn_xx: Tensor
    att_xx: Tensor
    att_aa: Tensor
    att_bb: Tensor
    att_pp: Tensor

    def __init__(self, embed_size: int):
        self.ffn_xx = Tensor([0.0] * embed_size)
        self.att_xx = Tensor([0.0] * embed_size)
        self.att_aa = Tensor([0.0] * embed_size)
        self.att_bb = Tensor([0.0] * embed_size)
        self.att_pp = Tensor([-1e30] * embed_size)


class State:
    state: list[LayerState]

    def __init__(self, embed_size: int, layers: int):
        self.states = [LayerState(embed_size) for _ in range(layers)]

    def __getitem__(self, i: int) -> LayerState:
        return self.states[i]

    def __setitem__(self, i: int, v: LayerState) -> None:
        self.states[i] = v


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
        self, x: Tensor, state: LayerState
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        xk = self.time_mix_k * (x - state.att_xx) + state.att_xx
        xv = self.time_mix_v * (x - state.att_xx) + state.att_xx
        xr = self.time_mix_r * (x - state.att_xx) + state.att_xx

        k = matvec(self.key, xk)
        v = matvec(self.value, xv)
        r = matvec(self.receptance, xr).sigmoid()

        # calculate output
        ww = k + self.time_first
        eww = ww.exp()
        epp = state.att_pp.exp()
        rwkv = r * ((eww * v + epp * state.att_aa) / (eww + epp * state.att_bb))

        # update state
        ww = state.att_pp + self.time_decay
        p = elemmax(ww, k)
        e1 = (ww - p).exp()
        e2 = (k - p).exp()

        return (
            matvec(self.output, rwkv),
            x.realize(),
            ((e1 * state.att_aa) + (e2 * v)).realize(),
            ((e1 * state.att_bb) + e2).realize(),
            p.realize(),
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

    def __call__(self, x: Tensor, state: LayerState) -> tuple[Tensor, Tensor]:
        xk = self.time_mix_k * (x - state.ffn_xx) + state.ffn_xx
        xr = self.time_mix_r * (x - state.ffn_xx) + state.ffn_xx

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

    def __call__(self, x: Tensor, state: LayerState) -> tuple[Tensor, LayerState]:
        new_state = LayerState(self.embed_size)

        ln1 = self.att_ln(x)
        (
            att,
            new_state.att_xx,
            new_state.att_aa,
            new_state.att_bb,
            new_state.att_pp,
        ) = self.att(ln1, state)
        x = x + att
        ln2 = self.ffn_ln(x)
        ffn, new_state.ffn_xx = self.ffn(ln2, state)
        x = x + ffn

        return x, new_state


class RWKV_RNN:
    ctx_size: int
    vocab_size: int
    embed_size: int
    layers: int

    emb: Tensor
    blocks: list[Block]
    ln_out_weight: Tensor
    ln_out_bias: Tensor
    head: Tensor

    def __init__(
        self, ctx_size: int, vocab_size: int, embed_size: int, layers: int, path: str
    ):
        self.ctx_size = ctx_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.layers = layers

        weights = pickle.load(open(path, "rb"))
        tg_weights = {}
        for k, v in tqdm(weights.items()):
            tg_weights[k] = Tensor(v)

        self.emb = tg_weights["emb.weight"]

        self.blocks = []
        for i in range(layers):
            self.blocks.append(
                Block(
                    embed_size,
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

    def forward(
        self, ctx: Tensor | int, state: State | None, preprocess: bool = False
    ) -> tuple[Tensor, State] | State:
        if state is None:
            state = State(self.embed_size, self.layers)

        if isinstance(ctx, int):
            x = self.emb[ctx]
        else:
            x = self.emb[int(ctx.numpy()[-1])]

        for i, block in enumerate(self.blocks):
            x, state[i] = block(x, state[i])

        if not preprocess:
            x = x.layernorm().linear(self.ln_out_weight, self.ln_out_bias)
            x = matvec(self.head, x)

            return x, state
        return state
