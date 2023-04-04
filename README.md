# tinyrwkv

A port of the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) large language model to the [tinygrad](https://tinygrad.org/) framework.

## Usage

Currently requires tinygrad from git.

Weights must be preprocessed.

```
Usage: python main.py [pre|gen|gra|cmp|gpt|tra]
  pre: preprocess weights
    `python main.py pre <.pth> <outfile>`
  gen: generate text with the rnn mode
    `python main.py gen <.pkl> [prompt]`
    Run with GPU=1 for much faster inference on gpu
  gra: use with GRAPH=1 to generate a graph of the rnn mode
    `GRAPH=1 python main.py gra`
  cmp: attempt to compile the rnn mode to c (broken)
    `python main.py cmp > out.c`
  gpt: generate text with the gpt mode
    `python main.py gpt`
  tra: train with gpt mode (broken)
    `python main.py tra`
```

## License

See the [LICENSE](./LICENSE) and [NOTICE](./NOTICE) files.
