from dataclasses import dataclass
import warnings
from typing import Callable
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad import nn
from tinygrad.helpers import round_up
from tinygrad.nn.state import get_parameters
from tinygrad.engine.jit import TinyJit

@dataclass
class BlockState:
  tm: Tensor
  kv: Tensor
  cm: Tensor

@dataclass
class State:
  blocks: list[BlockState]

  @TinyJit
  def assign(self, other):
    for i in range(len(self.blocks)):
      self.blocks[i].tm.assign(other.blocks[i].tm)
      self.blocks[i].kv.assign(other.blocks[i].kv)
      self.blocks[i].cm.assign(other.blocks[i].cm)
    self.realize()

  def realize(self):
    Tensor.realize(*get_parameters(self))
    return self

class Model:
  def __init__(self, n_blocks, dim, n_vocab, n_heads, *, rescale=0, dropout=0.01, linear:Callable=nn.Linear):
    self.n_blocks, self.dim, self.n_heads, self.head_dim, self.rescale, self.dropout = n_blocks, dim, n_heads, dim // n_heads, rescale, dropout

    self.emb = nn.Embedding(n_vocab, dim)
    self.emb_norm = nn.LayerNorm(dim)

    self.blocks = [Block(i, n_blocks, dim, n_heads, dropout=dropout, linear=linear) for i in range(n_blocks)]

    self.ln_out = nn.LayerNorm(dim)
    self.head = nn.Linear(dim, n_vocab, bias=False)

  def init_state(self, bs:int) -> State:
    return State([BlockState(
      tm=Tensor.zeros(bs, 1, self.dim),
      kv=Tensor.zeros(bs, self.n_heads, self.dim // self.n_heads, self.dim // self.n_heads),
      cm=Tensor.zeros(bs, 1, self.dim),
    ) for _ in range(self.n_blocks)])

  def __call__(self, x:Tensor, state:State) -> tuple[Tensor, State]:
    assert x.shape[0] == 1, "only batch size 1 supported"

    x = self.emb_norm(self.emb(x))

    new_state = []
    for i, block in enumerate(self.blocks):
      x, block_state = block(x, state.blocks[i])
      if self.rescale != 0 and (i + 1) % self.rescale == 0: x = x / 2
      new_state.append(block_state)

    logits = self.head(self.ln_out(x))[:, -1, :]

    # sampling
    return logits, State(new_state)

  def forward(self, x:Tensor) -> Tensor:
    x = self.emb_norm(self.emb(x)).dropout(self.dropout)
    for block in self.blocks: x = block.forward(x)
    return self.head(self.ln_out(x))

class Block:
  def __init__(self, i, n_blocks, dim, n_heads, *, dropout=0.01, linear:Callable=nn.Linear):
    self.dropout = dropout

    self.ln1 = nn.LayerNorm(dim)
    self.ln2 = nn.LayerNorm(dim)

    self.att = TimeMix(i, n_blocks, dim, n_heads, linear=linear)
    self.ffn = ChannelMix(i, n_blocks, dim, linear=linear)

  def __call__(self, x:Tensor, state:BlockState) -> tuple[Tensor, BlockState]:
    tm, tm_state, kv_state = self.att(self.ln1(x), state.tm, state.kv)
    x = x + tm
    cm, cm_state = self.ffn(self.ln2(x), state.cm)
    return x + cm, BlockState(tm_state, kv_state, cm_state)

  def forward(self, x):
    x = (x + self.att.forward(self.ln1(x))).dropout(self.dropout)
    return (x + self.ffn.forward(self.ln2(x))).dropout(self.dropout)

class TimeMix:
  def __init__(self, i, n_blocks, dim, n_heads, *, linear:Callable=nn.Linear):
    self.dim, self.n_heads, self.head_dim, self.head_divisor = dim, n_heads, dim // n_heads, 8

    ratio_0_to_1 = i / (n_blocks - 1)
    ratio_1_to_almost_0 = 1 - (i / n_blocks)
    self.time_mix_k = Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(ratio_1_to_almost_0)
    self.time_mix_v = Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(ratio_1_to_almost_0) + 0.3 * ratio_0_to_1
    self.time_mix_r = Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(0.5 * ratio_1_to_almost_0)
    self.time_mix_g = Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(0.5 * ratio_1_to_almost_0)

    self.time_decay = (-6 + 5 * Tensor.arange(dim).div(dim - 1).pow(0.7 + 1.3 * ratio_0_to_1)).reshape(n_heads, self.head_dim).float()
    self.time_faaaa = Tensor([ratio_0_to_1 * (1 - (n / (dim - 1))) + (((n + 1) % 3 - 1) * 0.1) for n in range(dim)]).reshape(n_heads, self.head_dim)

    self.receptance = linear(dim, dim, bias=False)
    self.key = linear(dim, dim, bias=False)
    self.value = linear(dim, dim, bias=False)
    self.output = linear(dim, dim, bias=False)
    self.gate = linear(dim, dim, bias=False)
    self.ln_x = nn.GroupNorm(n_heads, dim, eps=1e-5 * (self.head_divisor ** 2))

  @staticmethod
  def wkv(r:Tensor, k:Tensor, v:Tensor, u:Tensor, w:Tensor, kv_state:Tensor, C:int=32):
    B, H, T, X = r.shape
    if T % C != 0:
      warnings.warn(f"T % C != 0, T={T}, C={C}, T % C={T % C}", RuntimeWarning)
      # try to find the nearest C
      if T % 2 != 0: C = 1
      else:
        while T % C != 0: C -= 2
      warnings.warn(f"new C={C}, N={T // C}", RuntimeWarning)
    N = T // C

    if T == 1:
      y = kv_state + (kv := k.transpose(-2, -1) @ v) * u.transpose(-2, -1)
      kv_state = kv_state * w.transpose(-2, -1) + kv
      return r @ y, kv_state
    else:
      w_log = w.maximum(0.005).float().log()
      wc_log = w_log.reshape(w.shape[0], H, N, C, X)
      wc_log_cumsum = wc_log.cumsum(axis=-2)

      shifted_wc_log_cumsum = wc_log_cumsum.pad2d((0, 0, 1, -1))

      ws = wc_log.sum(axis=-2, keepdim=True)
      w_inter = ws - wc_log_cumsum
      w_intra = wc_log_cumsum - wc_log

      ws = list(map(lambda x: x.squeeze(-3), ws.transpose(-2, -1).exp().split(1, dim=-3)))
      w_inter = w_inter.exp()
      w_intra = w_intra.exp()

      r, k, v = r.reshape(B, H, N, C, X), k.reshape(B, H, N, C, X), v.reshape(B, H, N, C, X)
      u = u.unsqueeze(2).float()

      wc_log_offset = shifted_wc_log_cumsum[..., C//2:C//2 + 1, :]
      r_decay = (shifted_wc_log_cumsum - wc_log_offset).exp()
      k_inv_decay = (wc_log_offset - wc_log_cumsum).exp()
      a = ((r * r_decay) @ (k * k_inv_decay).transpose(-2, -1)).tril(-1)
      a = a + diag_embed(Tensor.einsum('bhncx,bhncx->bhnc', r, u * k))
      out = a @ v

      wkv = (k * w_inter).transpose(-2, -1) @ v
      wkv = list(map(lambda x: x.squeeze(-3), wkv.split(1, dim=-3)))

      states = []
      for i in range(T // C):
        states.append(kv_state)
        kv_state = kv_state * ws[i] + wkv[i]
      states = Tensor.stack(*states, dim=2)

      out = out + (r * w_intra) @ states
      out = out.reshape(B, H, T, X)
      return out.cast(dtypes.default_float), kv_state.cast(dtypes.default_float)

  def __call__(self, x:Tensor, tm_state:Tensor, kv_state:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # token shift
    xx = tm_state
    xr, xk, xv, xg = xx.lerp(x, self.time_mix_r), xx.lerp(x, self.time_mix_k), xx.lerp(x, self.time_mix_v), xx.lerp(x, self.time_mix_g)

    # projection
    r, k, v = self.receptance(xr), self.key(xk), self.value(xv)
    (B, T, D), H, X = x.shape, self.n_heads, self.head_dim
    r, k, v = r.reshape(B, T, H, X), k.reshape(B, T, H, X), v.reshape(B, T, H, X)
    r, k, v = r.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    # force the decay to be 0 to 1
    w = self.time_decay.exp().neg().exp().reshape(1, H, 1, X).expand(1, H, T, X)
    u = self.time_faaaa.reshape(1, H, 1, X)

    # wkv
    kv_state = kv_state.reshape(B, H, X, X)
    out, kv_state = TimeMix.wkv(r, k, v, u, w, kv_state)

    # project and gate
    out = self.ln_x(out.transpose(1, 2).reshape(B * T, D).div(self.head_divisor)).reshape(B, T, D)
    out = self.output(out * self.gate(xg).silu())
    return out, x, kv_state.reshape(B, H, X, X)

  def forward(self, x:Tensor) -> Tensor:
    tm_state = x.pad((None, (1, 0), None)).shrink((None, (0, x.shape[1]), None))
    kv_state = Tensor.zeros(x.shape[0], self.n_heads, self.head_dim, self.head_dim, dtype=x.dtype, device=x.device)
    return self(x, tm_state, kv_state)[0]

class ChannelMix:
  def __init__(self, i, n_blocks, dim, *, linear:Callable=nn.Linear):
    ratio_1_to_almost_0 = 1 - (i / n_blocks)
    self.time_mix_k = Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(ratio_1_to_almost_0)
    self.time_mix_r = Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(ratio_1_to_almost_0)

    self.receptance = linear(dim, dim, bias=False)
    self.key = linear(dim, round_up(int(dim * 3.5), 32), bias=False)
    self.value = linear(round_up(int(dim * 3.5), 32), dim, bias=False)

  def __call__(self, x:Tensor, cm_state:Tensor):
    # token shift
    xx = cm_state
    xr, xk = xx.lerp(x, self.time_mix_r), xx.lerp(x, self.time_mix_k)

    # projection and activation
    k = self.key(xk).relu().square()
    kv = self.value(k)

    # gate
    out = self.receptance(xr).sigmoid() * kv

    return out, x

  def forward(self, x:Tensor) -> Tensor:
    cm_state = x.pad((None, (1, 0), None)).shrink((None, (0, x.shape[1]), None))
    return self(x, cm_state)[0]

def diag_embed(x:Tensor, offset:int=0, dim1:int=-2, dim2:int=-1) -> Tensor:
  assert offset == 0, "only offset 0 supported"

  dim1 = (x.ndim + 1) + dim1 if dim1 < 0 else dim1
  dim2 = (x.ndim + 1) + dim2 if dim2 < 0 else dim2

  x = x.unsqueeze(dim1).transpose(-1, dim2)

  last_dim = x.shape[-1]

  # generate ranges for shifting indices based on offset
  a_range = Tensor.arange(last_dim, device=x.device)
  b_range = Tensor.arange(offset, last_dim + offset, device=x.device)

  # broadcast
  cond = a_range == b_range.unsqueeze(-1)
  cond_shape = [last_dim if i in (dim1, dim2) else 1 for i in range(x.ndim)]
  cond = cond.reshape(cond_shape)

  return cond.where(x, 0)

if __name__ == "__main__":
  Tensor.manual_seed(0)

  T = 32
  B = 1
  H = 1
  K,V = 4,4
  r = Tensor.rand(B,H,T,K)
  k = Tensor.rand(B,H,T,K)
  v = Tensor.rand(B,H,T,V)
  w = Tensor.rand(B,H,T,K)
  u = Tensor.rand(1,H,1,K)
  kv_state = Tensor.zeros(B,H,K,V)

  w = w.maximum(0.005)

  # sequential
  out = []
  kv_state_ = Tensor(kv_state.numpy())
  for t in range(T):
    out_, kv_state_ = TimeMix.wkv(r[...,t:t+1,:], k[...,t:t+1,:], v[...,t:t+1,:], u, w[...,t:t+1,:], kv_state_)
    out.append(out_)
  out_s = Tensor.cat(*out, dim=-2)
  print(out_s.numpy())

  # parallel
  out_p, _ = TimeMix.wkv(r,k,v,u,w,kv_state,C=8)
  print(out_p.numpy())

  assert (diff := (out_s - out_p).abs().max().item()) < 1e-6, f"{diff=}"
