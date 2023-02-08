from model_gpt import RWKV_GPT
from model_rnn import RWKV_RNN

from scipy.special import softmax
from tinygrad.nn.optim import get_parameters
from tinygrad.tensor import Tensor
from transformers import PreTrainedTokenizerFast
import numpy as np

from tqdm import tqdm
from typing import cast
import gc
import pickle
import sys
import types


np.set_printoptions(precision=4, suppress=True, linewidth=200)


def sample_logits(
    logits,
    *,
    alpha_counter=None,
    alpha_presence=0.0,
    alpha_frequency=0.0,
    temperature=0.8,
    top_p=0.9,
    top_k=35,
):
    # alpha sampling
    if alpha_counter is not None and alpha_presence > 0.0 and alpha_frequency > 0.0:
        for i in range(logits.shape[0]):
            logits[i] -= (alpha_counter[i] * alpha_frequency) + (
                float(alpha_counter[i] > 0) * alpha_presence
            )

    # top-k sampling
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = float("-Inf")

    # top-p sampling
    probs = softmax(logits, axis=-1)
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = pow(probs, 1.0 / temperature)
    probs = probs / np.sum(probs, axis=0)
    out = np.random.choice(a=len(probs), p=probs)
    return out


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
    print("Usage: python main.py [pre|gen|gra|gpt]")
    print("  pre: preprocess weights")
    print("  gen: generate text with the rnn mode")
    print("  gra: use with GRAPH=1 to generate a graph of the rnn mode")
    print("  gpt: generate text with the gpt mode")
    sys.exit(1)

if sys.argv[1] == "pre":
    # load weights
    import torch

    weights = torch.load(
        "./RWKV-4-Pile-1B5-Instruct-test1-20230124.pth", map_location="cpu"
    )

    # refine weights
    for k, v in tqdm(weights.items()):
        v = v.half().numpy()
        if ".time_" in k:
            v = v.squeeze()
        if ".time_decay" in k:
            v = -np.exp(v)

        weights[k] = v

    # precompute ln0 with emb.weight
    from model_run import layernorm

    ln0 = types.SimpleNamespace()
    setattr(ln0, "weight", Tensor(weights["blocks.0.ln0.weight"]))
    setattr(ln0, "bias", Tensor(weights["blocks.0.ln0.bias"]))
    weights["emb.weight"] = layernorm(Tensor(weights["emb.weight"]), ln0).numpy()

    # write weights
    import pickle

    pickle.dump(weights, open("weights_1b5.pkl", "wb"))

elif sys.argv[1] == "gen":
    Tensor.no_grad = True

    # load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

    # load model
    model = RWKV_RNN(1024, 50277, 1024, 24, "./weights.pkl")

    # run model once to move weights to correct device
    model.forward(187)

    # encode initial context
    ctx_str = "The quick brown"
    ctx = cast(np.ndarray, tokenizer.encode(ctx_str, return_tensors="np")[0])

    # encode separator
    sep = tokenizer.encode("\n\n", return_tensors="np")[0]

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

        out = tokenizer.decode(tokens)
        print(tokenizer.decode(last_token), end="", flush=True)

        # break if we reach the "end" of text
        if tokens[-1] == 0:
            break
        if out.endswith(("<|endoftext|>", "\n\n")):
            break
elif sys.argv[1] == "gra":
    Tensor.no_grad = True

    # seed random
    np.random.seed(42)

    # load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

    # load model
    model = RWKV_RNN(1024, 50277, 1024, 1, "./weights.pkl")

    print(model.forward(187))
elif sys.argv[1] == "gpt":
    np.random.seed(42)

    # load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

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
    model.forward(Tensor(np.array([[187, 187]])))

    # encode initial context
    ctx_str = "The quick brown"
    ctx = tokenizer.encode(ctx_str, return_tensors="np")

    # run model
    print(ctx_str, end="", flush=True)
    alpha_counter = np.zeros(50277)
    for i in range(100):
        out = model.forward(Tensor(ctx))
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
        txt = tokenizer.decode(sampled)
        print(txt, end="", flush=True)
        ctx = np.concatenate((ctx, [[sampled]]), axis=1)
