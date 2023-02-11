from model_gpt import RWKV_GPT
from model_rnn import RWKV_RNN
from utils import sample_logits

from tinygrad.nn.optim import get_parameters
from tinygrad.tensor import Tensor
from tokenizers import Tokenizer
import numpy as np

from tqdm import tqdm
from typing import cast
import gc
import pickle
import sys


np.set_printoptions(precision=4, suppress=True, linewidth=200)


def get_child(parent, key):
    obj = parent
    for k in key.split("."):
        if k.isnumeric():
            obj = obj[int(k)]
        elif isinstance(obj, dict):
            obj = obj[k]
        else:
            obj = getattr(obj, k)
    return obj


def count_parameters(model):
    params = get_parameters(model)
    count = 0
    for p in params:
        param_count = 1
        for s in p.shape:
            param_count *= s
        count += param_count
    return count


if len(sys.argv) < 2:
    print("Usage: python main.py [pre|gen|gra|gpt|tra]")
    print("  pre: preprocess weights")
    print("  gen: generate text with the rnn mode")
    print("  gra: use with GRAPH=1 to generate a graph of the rnn mode")
    print("  gpt: generate text with the gpt mode")
    print("  tra: train with gpt mode")
    sys.exit(1)

if sys.argv[1] == "pre":
    # load weights
    import torch

    weights = torch.load("./RWKV-4-Pile-430M-20220808-8066.pth", map_location="cpu")

    # refine weights
    for k, v in tqdm(weights.items()):
        v = v.half().numpy()
        if ".time_" in k:
            v = v.squeeze()
        if ".time_decay" in k:
            v = -np.exp(v)

        weights[k] = v

    # precompute ln0 with emb.weight
    print("Precomputing emb.weight with ln0...")
    weights["emb.weight"] = (
        Tensor(weights["emb.weight"])
        .layernorm()
        .linear(
            Tensor(weights["blocks.0.ln0.weight"]), Tensor(weights["blocks.0.ln0.bias"])
        )
        .numpy()
    )

    # write weights
    import pickle

    print("Writing weights...")
    pickle.dump(weights, open("weights.pkl", "wb"))

elif sys.argv[1] == "gen":
    Tensor.no_grad = True

    # load tokenizer
    tokenizer = Tokenizer.from_file("tokenizer.json")

    # load model
    model = RWKV_RNN(1024, 50277, 1024, 24, "./weights.pkl")

    # run model once to move weights to correct device
    model.forward(187)

    # encode initial context
    ctx_str = "The quick brown"
    ctx = tokenizer.encode(ctx_str).ids

    # encode separator
    sep = tokenizer.encode("\n\n").ids

    print("Preprocessing...")
    for i in tqdm(range(len(ctx))):
        x = np.concatenate([sep, ctx[:i]])
        model.forward(int(x[-1]), True)
    last_token = np.concatenate([sep, ctx])[-1]

    print()
    print(ctx_str, end="", flush=True)

    gc.collect()

    tokens = []
    alpha_counter = np.zeros(50277)
    out = ""
    while True:
        logits = model.forward(int(last_token))
        logits = logits.numpy()

        # disable <|endoftext|> token
        logits[0] = float("-Inf")

        # sample
        sampled = sample_logits(
            logits,
            alpha_counter=alpha_counter,
            alpha_presence=0.1,
            alpha_frequency=0.1,
            temperature=0.8,
            top_p=0.9,
            top_k=35,
        )

        last_token = sampled
        tokens.append(last_token)
        alpha_counter[last_token] += 1

        last_decoded = tokenizer.decode([last_token])
        print(last_decoded, end="", flush=True)
        out += last_decoded

        # break if we reach the "end" of text
        if tokens[-1] == 0:
            break
        if out.endswith(("<|endoftext|>", "\n\n")):
            break
elif sys.argv[1] == "gra":
    Tensor.no_grad = True

    # seed random
    np.random.seed(42)

    # load model
    model = RWKV_RNN(1024, 50277, 1024, 1, "./weights.pkl")

    print(model.forward(187))
elif sys.argv[1] == "gpt":
    np.random.seed(42)

    # load tokenizer
    tokenizer = Tokenizer.from_file("tokenizer.json")

    # load model
    model = RWKV_GPT(1024, 50277, 1024, 24)
    print(f"model has ~{count_parameters(model) / 1000 / 1000}M parameters")

    # load weights
    import torch

    weights = torch.load("./RWKV-4-Pile-430M-20220808-8066.pth", map_location="cpu")
    # convert to tinygrad
    tg_weights = {}
    for k, v in tqdm(weights.items()):
        tg_weights[k] = v.float().numpy()

    loaded = 0
    skipped = 0
    for k, v in tqdm(tg_weights.items()):
        try:
            w = get_child(model, k)
            loaded += 1
        except:
            w = None
            skipped += 1
        if w is not None:
            assert w.shape == v.shape
            w.assign(v.astype(np.float32))

    print(f"loaded {loaded} weights, skipped {skipped} weights")

    # cleanup extra memory
    gc.collect()

    # make fast
    model.forward(Tensor(np.array([[187, 510]])))

    # encode initial context
    ctx_str = "The quick brown"
    ctx = tokenizer.encode(ctx_str).ids

    # run model
    print(ctx_str, end="", flush=True)
    alpha_counter = np.zeros(50277)
    for i in range(10):
        out = model.forward(Tensor([ctx]))
        sampled = sample_logits(
            out.numpy()[-1][-1],
            alpha_counter=alpha_counter,
            alpha_presence=0.1,
            alpha_frequency=0.1,
            temperature=0.8,
            top_p=0.9,
            top_k=35,
        )
        alpha_counter[sampled] += 1
        txt = tokenizer.decode([sampled])
        print(txt, end="", flush=True)
        ctx = np.concatenate((ctx, [sampled]), axis=0)
elif sys.argv[1] == "tra":
    # load tokenizer
    tokenizer = Tokenizer.from_file("tokenizer.json")

    # load model
    model = RWKV_GPT(128, 50277, 768, 12)
    print(f"model has ~{count_parameters(model) / 1000 / 1000}M parameters")

    # load weights
    import torch

    weights = torch.load("./RWKV-4-Pile-169M-20220807-8023.pth", map_location="cpu")
    # convert to tinygrad
    tg_weights = {}
    for k, v in tqdm(weights.items()):
        tg_weights[k] = v.float().numpy()
    del weights

    loaded = 0
    skipped = 0
    for k, v in tqdm(tg_weights.items()):
        try:
            w = get_child(model, k)
            loaded += 1
        except:
            w = None
            skipped += 1
        if w is not None:
            assert w.shape == v.shape
            w.assign(v.astype(np.float32))

    print(f"loaded {loaded} weights, skipped {skipped} weights")
    gc.collect()

    from tinygrad.nn.optim import Adam

    print("starting optimizer...")
    params = get_parameters(model)
    optimizer = Adam(params, lr=1e-5, b1=0.9, b2=0.999)
    print("done starting optimizer")
    gc.collect()

    print("loading training data...")
    train_data = np.load("train.npy").astype(int)
    print("done loading training data")
    gc.collect()

    Tensor.training = True
    for epoch in range(10):
        for i in tqdm(range(0, len(train_data) - 129)):
            x = train_data[i : i + 128]
            y = train_data[i + 1 : i + 1 + 128]

            out = model.forward(Tensor([x]))
            sampled = sample_logits(
                out.numpy()[-1][-1],
                temperature=0.8,
                top_p=0.9,
                top_k=35,
            )
            txt = tokenizer.decode([sampled])
            print(tokenizer.decode(x) + "`" + txt + "`")

            optimizer.zero_grad()

            # calculate cross entropy loss with numpy
            out = out.clip(1e-8, 1 - 1e-8)[-1]
            outnp = out.numpy()
            loss = out[0][int(y[0])]
            for j in range(1, y.shape[0]):
                outy = out[j][int(y[j])]
                loss = loss.cat(outy, dim=0)
            loss = -loss.log()
            loss = loss.mean()
            gc.collect()

            loss.backward()
            gc.collect()

            optimizer.step()
            gc.collect()

            loss = loss.numpy()
            print(f"epoch {epoch}, step {i}, loss {loss}")
