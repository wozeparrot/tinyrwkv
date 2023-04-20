# tinyrwkv: A tinier port of RWKV-LM

[![Build Status](https://travis-ci.com/BlinkDL/tinyrwkv.svg?branch=master)](https://travis-ci.com/BlinkDL/tinyrwkv)

A port of the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) large language model to the [tinygrad](https://tinygrad.org/) framework.

## Usage

Currently, requires tinygrad from git.

Weights must be preprocessed.

```
Usage: python main.py [pre|gen|gra|cmp|gpt|ptr|tra]
  pre: preprocess weights from pytorch or from training subcommand
    `python main.py pre <.pth | .pkl> <out.pkl> <float | half>`
  gen: generate text with the rnn mode
    `python main.py gen <.pkl> [prompt]`
    Run with JIT=1 OPTLOCAL=1 GPU=1 for much faster inference on gpu
  gra: use with GRAPH=1 to generate a graph of the rnn mode
    `GRAPH=1 python main.py gra <.pkl>`
  cmp: attempt to compile the rnn mode to c (must use float32 weights)
       outputs the compiled code to `out.c`
    `python main.py cmp <.pkl>`
  gpt: generate text with the gpt mode
    `python main.py gpt`
  ptr: preprocess pytorch weights into compatible format for training
    `python main.py ptr <.pth> <out.pkl>`
  tra: train with gpt mode
    `python3 run.py tra <start_lr> <end_lr> <b1> <b2> <wd> <start_epoch> <epochs> <steps> <batch_size> <ctx_size> <ckpt_name>`
```

## License

See the [LICENSE](./LICENSE) and [NOTICE](./NOTICE) files.
