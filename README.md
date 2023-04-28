# tinyrwkv: A tinier port of RWKV-LM

[![Build Status](https://travis-ci.com/BlinkDL/tinyrwkv.svg?branch=master)](https://travis-ci.com/BlinkDL/tinyrwkv)

A port of the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) large language model to the [tinygrad](https://tinygrad.org/) framework.

## Usage

Currently, requires tinygrad from git.

Run the cli with `python -m cli`

```
usage: tinyrwkv-cli [-h] [--seed SEED] {pre,gen,cmp,ptr,gpt} ...

CLI for tinyrwkv

positional arguments:
  {pre,gen,cmp,ptr,gpt}
    pre                 preprocess either tinyrwkv trained weights or pytorch trained weights into RNN form
    gen                 freeform generation using the RNN mode (requires a preprocessed model using `pre`)
    cmp                 compile a RNN model into c source code and a compiled executable (need to run with CLANG=1)
    ptr                 preprocess pytorch weights weights into GPT form for training or inference
    gpt                 freeform generation using the GPT mode (requires a preprocessed model using `ptr`)

options:
  -h, --help            show this help message and exit
  --seed SEED           seed for random```

## License

See the [LICENSE](./LICENSE) and [NOTICE](./NOTICE) files.
