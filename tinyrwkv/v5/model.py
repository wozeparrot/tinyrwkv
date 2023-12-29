import math
from tinygrad import dtypes, nn, Tensor, TinyJit

class Model:
  def __init__(self, n_blocks, dim, n_vocab, n_heads, *, rescale=6, dropout=0.01, linear=nn.Linear):
    self.n_blocks, self.dim, self.n_heads, self.rescale, self.dropout = n_blocks, dim, n_heads, rescale, dropout
    self.state_size = dim + n_heads * dim // n_heads * dim // n_heads + dim

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
  def __call__(self, x: Tensor, state: Tensor, *, temperature:float=0) -> tuple[Tensor, Tensor]:
    x = self.emb_norm(self.emb(x))
    new_state = []
    for i, block in enumerate(self.blocks):
      tm_state = state[:, :, i*self.state_size:i*self.state_size+self.dim]
      kv_state = state[:, :, i*self.state_size+self.dim:i*self.state_size+self.dim+self.n_heads * self.dim // self.n_heads * self.dim // self.n_heads]
      cm_state = state[:, :, i*self.state_size+self.dim+self.n_heads * self.dim // self.n_heads * self.dim // self.n_heads:i*self.state_size+self.state_size]
      x, tm_state, kv_state, cm_state = block(x, [tm_state, kv_state, cm_state])
      new_state += [tm_state, kv_state, cm_state]
      if self.rescale != 0 and (i + 1) % self.rescale == 0: x = x / 2
    logits = self.head(self.ln_out(x))[:, -1, :]
    new_state = Tensor.cat(*new_state, dim=2)

    # sampling
    if temperature > 0: logits = logits / temperature
    else: return logits.argmax().realize(), new_state.realize()
    probs = logits.softmax()
    return probs.multinomial().realize(), new_state.realize()

  @TinyJit
  def forward(self, x: Tensor) -> Tensor:
    x = self.emb_norm(self.emb(x)).dropout(self.dropout)
    for block in self.blocks: x = block.forward(x)
    return self.head(self.ln_out(x)).realize()

class Block:
  def __init__(self, dim, n_heads, *, dropout=0.01, linear=nn.Linear):
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
  def __init__(self, dim, n_heads, *, linear=nn.Linear):
    self.n_heads = n_heads

    self.time_mix_k = Tensor.kaiming_uniform(1, 1, dim, a=math.sqrt(5))
    self.time_mix_v = Tensor.kaiming_uniform(1, 1, dim, a=math.sqrt(5))
    self.time_mix_r = Tensor.kaiming_uniform(1, 1, dim, a=math.sqrt(5))
    self.time_mix_g = Tensor.kaiming_uniform(1, 1, dim, a=math.sqrt(5))

    self.time_decay = Tensor.ones(n_heads, dim // n_heads)
    self.time_faaaa = Tensor.zeros(n_heads, dim // n_heads)

    self.receptance = linear(dim, dim, bias=False)
    self.key = linear(dim, dim, bias=False)
    self.value = linear(dim, dim, bias=False)
    self.output = linear(dim, dim, bias=False)
    self.gate = linear(dim, dim, bias=False)
    self.ln_x = nn.GroupNorm(n_heads, dim, eps=64e-5)

  @staticmethod
  def wkv(r, k, v, u, w, kv_state):
    y = kv_state + (kv := k @ v) * u
    kv_state = kv_state * w + kv
    return (r @ y)[:, :, 0], kv_state

  def __call__(self, x, state):
    # token shift
    xx = x.slice([None, (-1, x.shape[1] - 1), None]) if state is None else state[0]
    xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
    xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
    xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
    xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

    # projection
    r, k, v = self.receptance(xr), self.key(xk), self.value(xv)
    (B, T, C), H = x.shape, self.n_heads
    r, k, v = r.reshape(B, T, H, 1, C // H), k.reshape(B, T, H, C // H, 1), v.reshape(B, T, H, 1, C // H)

    # force the decay to be 0 to 1
    w = self.time_decay.exp().neg().exp().unsqueeze(-1)
    u = self.time_faaaa.unsqueeze(-1)

    # wkv for each timestep
    out, kv_state = [], Tensor.zeros(B, H, C // H, C // H) if state is None else state[1].reshape(B, H, C // H, C // H)
    for i in range(T):
      ou, kv_state = TimeMix.wkv(r[:, i], k[:, i], v[:, i], u, w, kv_state)
      out.append(ou)
    out = out[0].cat(*out[1:], dim=1) if T > 1 else out[0]

    # project and gate
    out = self.ln_x(out.reshape(B * T, C)).reshape(B, T, C)
    out = self.output(out * self.gate(xg).silu())
    return out if state is None else (out, x, kv_state.reshape(B, 1, H * C // H * C // H))
  def forward(self, x): return self(x, None)

class ChannelMix:
  def __init__(self, dim, *, linear=nn.Linear):
    self.time_mix_k = Tensor.kaiming_uniform(1, 1, dim, a=math.sqrt(5))
    self.time_mix_r = Tensor.kaiming_uniform(1, 1, dim, a=math.sqrt(5))

    self.receptance = linear(dim, dim, bias=False)
    self.key = linear(dim, int(dim * 3.5), bias=False)
    self.value = linear(int(dim * 3.5), dim, bias=False)

  def __call__(self, x, state):
    # token shift
    xx = x.slice([None, (-1, x.shape[1] - 1), None]) if state is None else state
    xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
    xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

    # projection and activation
    k = self.key(xk).relu().square()
    kv = self.value(k)

    # gate
    out = self.receptance(xr).sigmoid() * kv

    return out if state is None else (out, x)
  def forward(self, x): return self(x, None)

class Int8Linear:
  def __init__(self, in_features, out_features, bias=False):
    assert not bias, "bias not supported"
    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.int8)
    self.scale = Tensor.empty(out_features, dtype=dtypes.float16)
  def __call__(self, x: Tensor) -> Tensor: return x.linear(self.weight.cast(dtypes.float16).T * self.scale)

  @staticmethod
  def quantize(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    new_state_dict = {}
    for k, v in state_dict.items():
      if ".key" in k or ".value" in k or ".receptance" in k or ".gate" in k or ".output" in k or ".value" in k:
        v = v.to("CLANG")
        scale = (v.abs().max(axis=1) / 127.0).cast(dtypes.float16)
        new_state_dict[k] = (v.T / scale).T.cast(dtypes.int8)
        new_state_dict[k.replace(".weight", ".scale")] = scale
      else:
        new_state_dict[k] = v
    return new_state_dict

def NF4Linear(block_size):
  CODE = Tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
  ], dtype=dtypes.float32)
  class _NF4Linear:
    def __init__(self, in_features, out_features, bias=False):
      assert not bias, "bias not supported"
      self.in_features, self.out_features = in_features, out_features
      self.weight = Tensor.empty(int(out_features * in_features / 2), dtype=dtypes.uint8)
      self.scale = Tensor.empty(int(out_features * in_features / block_size), 1, dtype=dtypes.float32)

    def __call__(self, x: Tensor) -> Tensor:
      high_bits, low_bits = self.weight / 2 ** 4, (self.weight * 2 ** 4).contiguous() / 2 ** 4
      unpacked = Tensor.stack([high_bits, low_bits], dim=-1).flatten()
      unscaled = CODE[unpacked].reshape(-1, block_size) * self.scale
      return x.linear(unscaled.reshape(self.out_features, self.in_features).T)

    @staticmethod
    def quantize(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
      new_state_dict = {}
      for k, v in state_dict.items():
        if ".key" in k or ".value" in k or ".receptance" in k or ".gate" in k or ".output" in k or ".value" in k:
          grouped = v.to(CODE.device).reshape(-1, block_size)
          scale = (grouped.abs().max(axis=1, keepdim=True))
          coded = ((grouped / scale).unsqueeze(-1) - CODE).abs().argmin(axis=-1).cast(dtypes.uint8).flatten()
          new_state_dict[k] = coded[::2] * 2 ** 4 + coded[1::2]
          new_state_dict[k.replace(".weight", ".scale")] = scale.cast(dtypes.float32)
        else:
          new_state_dict[k] = v
      return new_state_dict
  return _NF4Linear
