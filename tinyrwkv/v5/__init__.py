import math
from typing import Callable
from tinygrad import dtypes, nn, Tensor, TinyJit
from tinygrad.helpers import round_up

from ..utils import sample

class Model:
  def __init__(self, n_blocks, dim, n_vocab, n_heads, *, rescale=6, dropout=0.01, linear:Callable=nn.Linear):
    self.n_blocks, self.dim, self.n_heads, self.head_dim, self.rescale, self.dropout = n_blocks, dim, n_heads, dim // n_heads, rescale, dropout
    self.state_size = dim + n_heads * self.head_dim * self.head_dim + dim

    self.emb = nn.Embedding(n_vocab, dim)
    self.emb_norm = nn.LayerNorm(dim)

    self.blocks = [Block(dim, n_heads, dropout=dropout, linear=linear) for _ in range(n_blocks)]

    self.ln_out = nn.LayerNorm(dim)
    self.head = nn.Linear(dim, n_vocab, bias=False)

  def init_state(self, bs: int) -> Tensor:
    return Tensor.cat(*[
      Tensor.zeros(bs, 1, self.dim, requires_grad=False),
      Tensor.zeros(bs, 1, self.n_heads * self.dim // self.n_heads * self.dim // self.n_heads, dtype=dtypes.float32, requires_grad=False),
      Tensor.zeros(bs, 1, self.dim, requires_grad=False)
    ] * self.n_blocks, dim=2)

  @TinyJit
  def __call__(self, x: Tensor, state: Tensor, *, temperature:float=0, top_k:int=0, top_p:float=0, alpha_presence:float=0, alpha_frequency:float=0) -> tuple[Tensor, Tensor]:
    assert x.shape[0] == 1, "only batch size 1 supported"

    x = self.emb_norm(self.emb(x))
    new_state = []
    for i, block in enumerate(self.blocks):
      tm_state = state[:, :, i*self.state_size:i*self.state_size + self.dim]
      kv_state = state[:, :, i*self.state_size + self.dim:i*self.state_size + self.dim + self.n_heads*self.head_dim*self.head_dim]
      cm_state = state[:, :, i*self.state_size + self.dim + self.n_heads * self.head_dim * self.head_dim:i*self.state_size+self.state_size]
      x, tm_state, kv_state, cm_state = block(x, [tm_state, kv_state, cm_state])
      new_state += [tm_state, kv_state, cm_state]
      if self.rescale != 0 and (i + 1) % self.rescale == 0: x = x / 2
    logits = self.head(self.ln_out(x))[:, -1, :]
    new_state = Tensor.cat(*new_state, dim=2)

    # sampling
    return sample(logits.flatten(), temperature, top_k, top_p, alpha_presence, alpha_frequency).realize(), new_state.realize()

  def forward(self, x: Tensor) -> Tensor:
    x = self.emb_norm(self.emb(x)).dropout(self.dropout)
    for block in self.blocks: x = block.forward(x)
    return self.head(self.ln_out(x))

class Block:
  def __init__(self, dim, n_heads, *, dropout=0.01, linear:Callable=nn.Linear):
    self.dropout = dropout

    self.ln1 = nn.LayerNorm(dim)
    self.ln2 = nn.LayerNorm(dim)

    self.att = TimeMix(dim, n_heads, linear=linear)
    self.ffn = ChannelMix(dim, linear=linear)

  def __call__(self, x, state):
    tm, tm_state, kv_state = self.att(self.ln1(x), state[0:2])
    cm, cm_state = self.ffn(self.ln2(x := x + tm), state[2])
    return x + cm, tm_state, kv_state, cm_state

  def forward(self, x):
    x = (x + self.att.forward(self.ln1(x))).dropout(self.dropout)
    return (x + self.ffn.forward(self.ln2(x))).dropout(self.dropout)

class TimeMix:
  def __init__(self, dim, n_heads, *, linear:Callable=nn.Linear):
    self.n_heads, self.head_dim = n_heads, dim // n_heads

    self.time_mix_k = Tensor.kaiming_uniform(1, 1, dim, a=math.sqrt(5))
    self.time_mix_v = Tensor.kaiming_uniform(1, 1, dim, a=math.sqrt(5))
    self.time_mix_r = Tensor.kaiming_uniform(1, 1, dim, a=math.sqrt(5))
    self.time_mix_g = Tensor.kaiming_uniform(1, 1, dim, a=math.sqrt(5))

    self.time_decay = Tensor.ones(n_heads, self.head_dim)
    self.time_faaaa = Tensor.zeros(n_heads, self.head_dim)

    self.receptance = linear(dim, dim, bias=False)
    self.key = linear(dim, dim, bias=False)
    self.value = linear(dim, dim, bias=False)
    self.output = linear(dim, dim, bias=False)
    self.gate = linear(dim, dim, bias=False)
    self.ln_x = nn.GroupNorm(n_heads, dim, eps=64e-5)

  @staticmethod
  def wkv(r:Tensor, k:Tensor, v:Tensor, u:Tensor, w:Tensor, kv_state:Tensor):
    (B, H, T, X), C = r.shape, 32
    if T % C != 0: C = 1

    if T == 1:
      y = kv_state + (kv := k.transpose(-2, -1) @ v) * u.transpose(-2, -1)
      kv_state = kv_state * w.transpose(-2, -1) + kv
      return r @ y, kv_state
    else:
      w_log = w.log()
      wc_log = w_log.reshape(-1, H, T // C, C, X)
      wc_log_cumsum = wc_log.cumsum(axis=-2)

      shifted_wc_log_cumsum = wc_log_cumsum.pad2d((0, 0, 1, -1))

      ws = wc_log.sum(axis=-1, keepdim=True)
      w_inter = ws - wc_log_cumsum
      w_intra = wc_log_cumsum - wc_log

      ws = list(map(lambda x: x.squeeze(-3), ws.transpose(-2, -1).exp().split(1, dim=-3)))
      w_inter = w_inter.exp()
      w_intra = w_intra.exp()

      r, k, v = r.reshape(B, H, T // C, C, X), k.reshape(B, H, T // C, C, X), v.reshape(B, H, T // C, C, X)
      u = u.unsqueeze(2)

      wc_log_offset = shifted_wc_log_cumsum[..., C//2:C//2 + 1, :]
      r_decay = (shifted_wc_log_cumsum - wc_log_offset).exp()
      k_inv_decay = (wc_log_offset - wc_log_cumsum).exp()
      a = ((r * r_decay) @ (k * k_inv_decay).transpose(-2, -1)).tril(-1)
      out = a + Tensor.einsum("bhncx,bhncx,bhncx->bhncx", r, u * k, v)

      wkv = (k * w_inter).transpose(-2, -1) @ v
      wkv = list(map(lambda x: x.squeeze(-3), wkv.split(1, dim=-3)))

      states = []
      for i in range(T // C):
        states.append(kv_state)
        kv_state = (kv_state * ws[i] + wkv[i])
      states = Tensor.stack(*states, dim=2)

      out = out + (r * w_intra) @ states
      out = out.reshape(B, H, T, X)
      return out, kv_state

  def __call__(self, x:Tensor, state:Tensor | None):
    # token shift
    xx = x.pad((None, (0, 1), None)).shrink((None, (1, x.shape[1] + 1), None)) if state is None else state[0]
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
    out, kv_state = [], Tensor.zeros(B, H, X, X, dtype=x.dtype, device=x.device) if state is None else state[1].reshape(B, H, X, X)
    out, kv_state = TimeMix.wkv(r, k, v, u, w, kv_state)

    # project and gate
    out = self.ln_x(out.reshape(B * T, D)).reshape(B, T, D)
    out = self.output(out * self.gate(xg).silu())
    return out if state is None else (out, x, kv_state.reshape(B, 1, H * X * X))
  def forward(self, x): return self(x, None)

class ChannelMix:
  def __init__(self, dim, *, linear:Callable=nn.Linear):
    self.time_mix_k = Tensor.kaiming_uniform(1, 1, dim, a=math.sqrt(5))
    self.time_mix_r = Tensor.kaiming_uniform(1, 1, dim, a=math.sqrt(5))

    self.receptance = linear(dim, dim, bias=False)
    self.key = linear(dim, round_up(int(dim * 3.5), 32), bias=False)
    self.value = linear(round_up(int(dim * 3.5), 32), dim, bias=False)

  def __call__(self, x:Tensor, state:Tensor | None):
    # token shift
    xx = x.pad((None, (0, 1), None)).shrink((None, (1, x.shape[1] + 1), None)) if state is None else state[0]
    xr, xk = xx.lerp(x, self.time_mix_r), xx.lerp(x, self.time_mix_k)

    # projection and activation
    k = self.key(xk).relu().square()
    kv = self.value(k)

    # gate
    out = self.receptance(xr).sigmoid() * kv

    return out if state is None else (out, x)
  def forward(self, x): return self(x, None)
