from tinygrad.helpers import Context, GlobalCounters
from tinygrad import Tensor, TinyJit, nn, dtypes
from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Optimizer
from tqdm import tqdm

from tinyrwkv.tokenizer import Tokenizer
from tinyrwkv.v5 import Model

class Lion(Optimizer):
  def __init__(self, params: list[Tensor], lr: float = 1e-4, b1: float = 0.9, b2: float = 0.999, weight_decay: float = 0.0):
    super().__init__(params, lr)
    self.b1, self.b2, self.wd = b1, b2, weight_decay
    self.ea = [Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device, requires_grad=False).contiguous() for t in self.params]

  def _step(self):
    for i, t in enumerate(self.params):
      assert t.grad is not None
      g = t.grad

      update = self.ea[i] * self.b1 + g * (1 - self.b1)
      t.assign((t.detach() * (1 - self.lr * self.wd)) + (update.sign() * (-self.lr)))
      self.ea[i].assign(self.ea[i] * self.b2 + g * (1 - self.b2))
    return self.ea

def z_loss(logits: Tensor): return 1e-4 * logits.logsumexp(axis=-1).mean()

if __name__ == "__main__":
  model = Model(4, 128, 65536, 4, dropout=0, rescale=0)
  print(f"{sum(p.numel() for p in get_parameters(model)) / 1e6}M parameters")
  tokenizer = Tokenizer()

  # optim = nn.optim.AdamW(nn.state.get_parameters(model), lr=1e-3, weight_decay=1e-5)
  optim = Lion(get_parameters(model), lr=1e-4, weight_decay=1e-5)

  @TinyJit
  def train_step(model: Model, x, y):
    y_hat = model.forward(x)
    mloss = y_hat.sparse_categorical_crossentropy(y)
    zloss = z_loss(y_hat)
    loss = mloss + zloss
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.realize(), mloss.realize(), zloss.realize()

  train_data = tokenizer.encode("hello world!")
  print(len(train_data))

  with Context(BEAM=2):
    Tensor.no_grad = False
    Tensor.training = True
    for i in (t := tqdm(range(1000))):
      GlobalCounters.reset()
      loss, mloss, zloss = train_step(model, Tensor([train_data[:-1]]).expand(16, -1), Tensor([train_data[1:]]).expand(16, -1))
      t.set_description(f"loss: {loss.item():6.6f}, mloss: {mloss.item():6.6f}, zloss: {zloss.item():6.6f}")
