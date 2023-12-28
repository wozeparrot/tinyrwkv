from pathlib import Path
from typing import NamedTuple

with open(Path(__file__).parent / "vocab" / "world.txt", "r") as f:
  VOCAB = f.readlines()

class Tokenizer:
  table: list[list[list[bytes]]]
  good: list[set[int]]
  wlen: list[int]

  def __init__(self):
    self.idx2token, sorted = {}, []
    for l in VOCAB:
      idx = int(l[: l.index(" ")])
      x = eval(l[l.index(" ") : l.rindex(" ")])
      x = x.encode("utf-8") if isinstance(x, str) else x
      assert isinstance(x, bytes)
      assert len(x) == int(l[l.rindex(" ") :])
      sorted += [x]
      self.idx2token[idx] = x
    self.token2idx = {v: int(k) for k, v in self.idx2token.items()}

    # precompute some tables for fast matching
    self.table = [[[] for _ in range(256)] for _ in range(256)]
    self.good = [set() for _ in range(256)]
    self.wlen = [0 for _ in range(256)]

    # reverse order - match longer tokens first
    for i in reversed(range(len(sorted))):
      s = sorted[i]
      if len(s) >= 2:
        s0, s1 = int(s[0]), int(s[1])
        self.table[s0][s1] += [s]
        self.wlen[s0] = max(self.wlen[s0], len(s))
        self.good[s0].add(s1)

  def encodeBytes(self, src: bytes) -> list[int]:
    src_len, tokens, i = len(src), [], 0
    while i < src_len:
      s = src[i : i + 1]
      if i < src_len - 1:
        s0, s1 = int(src[i]), int(src[i + 1])
        if s1 in self.good[s0]:
          sss: bytes = src[i : i + self.wlen[s0]]
          try: s = next(filter(sss.startswith, self.table[s0][s1]))
          except: pass
      tokens.append(self.token2idx[s])
      i += len(s)
    return tokens
  def decodeBytes(self, tokens): return b"".join(map(lambda i: self.idx2token[i], tokens))

  def encode(self, src: str): return self.encodeBytes(src.encode("utf-8"))
  def decode(self, tokens): return self.decodeBytes(tokens).decode("utf-8")

  def get_vocab_size(self): return len(self.idx2token)
