# tinyrwkv: A tinier port of RWKV-LM

A port of the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) family of large language models to the [tinygrad](https://tinygrad.org/) framework.

## Roadmap

- [x] Implement the WKV kernel as a custom function
- [ ] Implement the backwards of the WKV kernel as a custom function
- [x] Add support for the world model and tokenizer
- [x] Add support for the MIDI models

## Dependencies

Currently, requires tinygrad from git or just use the nix flake.

### Python
```
numpy
pydot (only for GRAPH=1)
tinygrad
tokenizers
torch (only for loading pytorch weights)
tqdm
wandb (optional during training)
```

### System
```
rust (only for compiling)
clang (only for compiling)
graphviz (only for GRAPH=1)
```

## Usage

Run the CLI with `python -m cli`.

Also, usable as a python package to embed in other projects. It's also possible to compile the model to portable C code and embed it that way.

```
usage: tinyrwkv-cli [-h] [--seed SEED] {pre,gen,cht,cmp,bch,ptr,gpt,tra,bpt,wkv,mus} ...

CLI for tinyrwkv

positional arguments:
  {pre,gen,cht,cmp,bch,ptr,gpt,tra,bpt,wkv,mus}
    pre                 preprocess either tinyrwkv trained weights or pytorch trained weights into RNN form
    gen                 freeform generation using the RNN mode (requires a preprocessed model using `pre`)
    cht                 chat with a model in RNN mode (requires a preprocessed model using `pre`)
    cmp                 compile a RNN model into c source code and a compiled executable (need to run with CLANG=1)
    bch                 benchmark the rnn mode
    ptr                 preprocess pytorch weights weights into GPT form for training or inference
    gpt                 freeform generation using the GPT mode (requires a preprocessed model using `ptr`)
    tra                 pretrain or finetune a model (if finetuning the model needs to be preprocessed with `ptr`)
    bpt                 benchmark the gpt mode
    wkv                 benchmark/test each wkv module
    mus                 music generation using the RNN mode (requires a preprocessed model using `pre`)

options:
  -h, --help            show this help message and exit
  --seed SEED           seed for random
```

## License

See the [LICENSE](./LICENSE) and [NOTICE](./NOTICE) files.
