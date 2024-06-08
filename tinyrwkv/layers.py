from tinygrad import dtypes, Tensor, nn

class Int8Linear:
  def __init__(self, in_features: int, out_features: int, bias=False):
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
    def __init__(self, in_features: int, out_features: int, bias=False):
      assert not bias, "bias not supported"
      self.in_features, self.out_features = in_features, out_features
      self.weight = Tensor.empty(int(out_features * in_features / 2), dtype=dtypes.uint8)
      self.scale = Tensor.empty(int(out_features * in_features / block_size), 1, dtype=dtypes.float32)

    def __call__(self, x: Tensor) -> Tensor:
      low_bits = (self.weight * 2 ** 4).contiguous()
      unpacked = Tensor.stack([self.weight, low_bits], dim=-1).div(2 ** 4, upcast=False)
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

class LayerNorm(nn.LayerNorm):
  def __init__(self, dim:int, eps=1e-5, affine=True): super().__init__(dim, eps, affine)
  def __call__(self, x:Tensor) -> Tensor:
    if dtypes.default_float != dtypes.bfloat16: return super().__call__(x.float()).cast(dtypes.default_float)
    else: return super().__call__(x)

class GroupNorm(nn.GroupNorm):
  def __init__(self, num_groups:int, num_channels:int, eps=1e-5, affine=True): super().__init__(num_groups, num_channels, eps, affine)
  def __call__(self, x:Tensor) -> Tensor:
    if dtypes.default_float != dtypes.bfloat16: return super().__call__(x.float()).cast(dtypes.default_float)
    else: return super().__call__(x)
