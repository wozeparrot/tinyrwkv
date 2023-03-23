# tinyrwkv

A port of the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) large language model to the [tinygrad](https://tinygrad.org/) framework.

## Usage

Currently requires tinygrad from git.

Weights must be preprocessed.

```
Usage: python main.py [pre|gen|gra|cmp|gpt|tra]
  pre: preprocess weights
  gen: generate text with the rnn mode
  gra: use with GRAPH=1 to generate a graph of the rnn mode
  cmp: attempt to compile the rnn mode to c (broken)
  gpt: generate text with the gpt mode
  tra: train with gpt mode
```

## License

See the [LICENSE](./LICENSE) and [NOTICE](./NOTICE) files.
