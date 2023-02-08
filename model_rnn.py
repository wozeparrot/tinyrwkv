from tinygrad.tensor import Tensor
import tinygrad.nn as nn
import pickle
from tqdm import tqdm

import types
import gc


class LayerState:
    def __init__(self, embed_size):
        self.ffn_xx = Tensor([0.0] * embed_size)
        self.att_xx = Tensor([0.0] * embed_size)
        self.att_aa = Tensor([0.0] * embed_size)
        self.att_bb = Tensor([0.0] * embed_size)
        self.att_pp = Tensor([-1e30] * embed_size)


class State:
    def __init__(self, embed_size, layers):
        self.states = [LayerState(embed_size) for _ in range(layers)]

    def __getitem__(self, i):
        return self.states[i]


class Att:
    def __init__(
        self,
        time_mix_k,
        time_mix_v,
        time_mix_r,
        key,
        value,
        receptance,
        time_first,
        time_decay,
        output,
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

    def __call__(self, x, state):
        xx = state.att_xx
        xk = self.time_mix_k * (x - xx) + xx
        xv = self.time_mix_v * (x - xx) + xx
        xr = self.time_mix_r * (x - xx) + xx

        k = matvec(self.key, xk)
        v = matvec(self.value, xv)
        r = matvec(self.receptance, xr).sigmoid()

        aa = state.att_aa
        bb = state.att_bb
        pp = state.att_pp

        # calculate output
        ww = k + self.time_first
        eww = ww.exp()
        epp = pp.exp()
        rwkv = r * ((eww * v + epp * aa) / (eww + epp * bb))

        # update state
        ww = pp + self.time_decay
        p = ww.elemmax(k)
        e1 = (ww - p).exp()
        e2 = (k - p).exp()

        state.att_xx = x.realize()
        state.att_aa = ((e1 * aa) + (e2 * v)).realize()
        state.att_bb = ((e1 * bb) + e2).realize()
        state.att_pp = p.realize()

        return matvec(self.output, rwkv)


class Ffn:
    def __init__(
        self,
        time_mix_k,
        time_mix_r,
        key,
        value,
        receptance,
    ):
        self.time_mix_k = time_mix_k
        self.time_mix_r = time_mix_r
        self.key = key
        self.value = value
        self.receptance = receptance

    def __call__(self, x, state):
        xx = state.ffn_xx
        xk = self.time_mix_k * (x - xx) + xx
        xr = self.time_mix_r * (x - xx) + xx
        state.ffn_xx = x.realize()

        k = matvec(self.key, xk).relu().square()
        kv = matvec(self.value, k)
        r = matvec(self.receptance, xr).sigmoid()
        rkv = r * kv

        return rkv


class Block:
    def __init__(
        self,
        embed_size,
        att_ln_weight,
        att_ln_bias,
        att_time_mix_k,
        att_time_mix_v,
        att_time_mix_r,
        att_key,
        att_value,
        att_receptance,
        att_time_first,
        att_time_decay,
        att_output,
        ffn_ln_weight,
        ffn_ln_bias,
        ffn_time_mix_k,
        ffn_time_mix_r,
        ffn_key,
        ffn_value,
        ffn_receptance,
    ):
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

    def __call__(self, x, state):
        ln1 = self.att_ln(x)
        x += self.att(ln1, state)
        ln2 = self.ffn_ln(x)
        x += self.ffn(ln2, state)
        return x


class RWKV_RNN:
    def __init__(self, ctx_size, vocab_size, embed_size, layers, path):
        self.ctx_size = ctx_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.layers = []

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

        self.state = State(embed_size, layers)

    def forward(self, ctx: Tensor | int, preprocess: bool = False) -> Tensor | None:
        if isinstance(ctx, int):
            x = self.emb[ctx]
        else:
            x = self.emb[int(ctx.numpy()[-1])]

        for i, block in enumerate(self.blocks):
            x = block(x, self.state[i]).realize()

        if not preprocess:
            x = x.layernorm().linear(self.ln_out_weight, self.ln_out_bias)
            x = matvec(self.head, x)

            return x.realize()


def matvec(mat: Tensor, vec: Tensor) -> Tensor:
    return (mat * vec).sum(axis=1)
