from model_gpt import RWKV_GPT
from model_rnn import RWKV_RNN
from utils import sample_logits

from tinygrad.nn.optim import get_parameters
from tinygrad.tensor import Tensor
from tokenizers import Tokenizer
import numpy as np

from tqdm import tqdm
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
    print("  cmp: attempt to compile the rnn mode to c (broken)")
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

    # jit
    from tinygrad.jit import TinyJit

    @TinyJit
    def run(x):
        ret = model.forward(x, jit=True)
        return ret.realize()

    embed = Tensor(model.index_embed(510).numpy())
    state = model.init_state()
    the_input = model.build_jit_input(embed, state)

    # run model twice to initialize the jit
    the_output = run(the_input)
    the_output = run(the_input)

    # encode initial context
    ctx_str = """
This is a test of the emergency broadcast system. This is only a test. If this had been an actual emergency, you would have been instructed to do something. This concludes this test of the emergency broadcast system.
"""
    ctx = tokenizer.encode(ctx_str).ids

    # encode separator
    sep = tokenizer.encode("\n\n").ids

    print("Preprocessing...")
    state = model.init_state()
    for i in tqdm(range(len(ctx))):
        x = np.concatenate([sep, ctx[:i]])
        embed = model.index_embed(int(x[-1]))
        the_input = model.build_jit_input(embed, state)
        out = run(the_input)
        state = out[50277:]
    last_token = np.concatenate([sep, ctx])[-1]

    print()
    print(ctx_str, end="", flush=True)

    gc.collect()

    tokens = []
    alpha_counter = np.zeros(50277)
    out = ""
    while True:
        embed = model.index_embed(int(last_token))
        the_input = model.build_jit_input(embed, state)
        the_output = run(the_input)
        logits = the_output[:50277]
        state = the_output[50277:]
        # logits to cpu
        logits = logits.cpu().numpy()
        logits.flags.writeable = True

        # disable <|endoftext|> token
        logits[0] = float("-Inf")

        # sample
        sampled = sample_logits(
            logits,
            alpha_counter=alpha_counter,
            alpha_presence=0.1,
            alpha_frequency=0.2,
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
        # if tokens[-1] == 0:
        #     break
        # if out.endswith(("<|endoftext|>", "\n\n")):
        #     break
elif sys.argv[1] == "gra":
    Tensor.no_grad = True

    # seed random
    np.random.seed(42)

    # load model
    model = RWKV_RNN(1024, 50277, 1024, 1, "./weights.pkl")

    print(model.forward(187, None))
elif sys.argv[1] == "cmp":
    Tensor.no_grad = True

    # seed random
    np.random.seed(42)

    # load model
    model = RWKV_RNN(1024, 50277, 768, 12, "./weights-169m.pkl")

    from tinygrad.jit import TinyJit

    @TinyJit
    def run(x):
        ret = model.forward(x, jit=True)
        return ret.realize()

    embed = Tensor(model.index_embed(510).numpy())
    init_state = model.init_state()
    the_input = model.build_jit_input(embed, init_state)
    out = run(the_input)
    out = run(the_input)

    for (j, i), idx in run.input_replace.items():
        run.jit_cache[j][1][i] = the_input.lazydata.realized.raw()

    special_names = {
        id(the_input.lazydata.realized.raw()): "input",
        id(out.lazydata.realized.raw()): "output",
    }

    from utils import compile_net

    with open("out.c", "w") as f:
        functions, statements, bufs, bufs_to_save = compile_net(run, special_names)

        cprog = [
            "#include <stdio.h>",
            "#include <string.h>",
            "#include <math.h>",
            "#define max(x,y) ((x>y)?x:y)",
        ]
        f.write("\n".join(cprog) + "\n")

        # write init state
        f.write(f'unsigned char state_data[] = "')
        for i, x in enumerate(tqdm(bytes(init_state.lazydata.realized.raw()._buf))):
            if i % 32 == 0 and i > 0:
                f.write('"\n"')
            f.write(f"\\x%02X" % x)
        f.write('";\n')
        f.write(f"float *state = (float *)state_data;\n")

        # write embedding weights
        f.write(f'unsigned char emb_data[] = "')
        for i, x in enumerate(tqdm(bytes(model.emb.lazydata.realized.raw()._buf))):
            if i % 32 == 0 and i > 0:
                f.write('"\n"')
            f.write(f"\\x%02X" % x)
        f.write('";\n')
        f.write(f"float *emb = (float *)emb_data;\n")

        for name, cl in tqdm(bufs_to_save.items()):
            f.write(f'unsigned char {name}_data[] = "')
            for i, x in enumerate(bytes(cl._buf)):
                if i % 32 == 0 and i > 0:
                    f.write('"\n"')
                f.write(f"\\x%02X" % x)
            f.write('";\n')

        cprog = [
            f"float {name}[{len}];"
            if name not in bufs_to_save
            else f"float *{name} = (float *){name}_data;"
            for name, len in bufs.values()
        ]
        f.write("\n".join(cprog) + "\n")

        cprog = list(functions.values())
        f.write("\n".join(cprog) + "\n")

        cprog = ["void net() {"] + statements + ["}"]
        f.write("\n".join(cprog) + "\n")

        cprog = [
            """
int main(int argc, char *argv[]) {{
  // load input
  memcpy(input, emb + 510, sizeof(float) * 768);
  memcpy(input + 768, state, sizeof(float) * 12 * 768);
  net();
  memcpy(state, output + 50277, sizeof(float) * 12 * 768);
  memcpy(input, emb + 3158, sizeof(float) * 768);
  memcpy(input + 768, state, sizeof(float) * 12 * 768);
  net();
  memcpy(state, output + 50277, sizeof(float) * 12 * 768);
  memcpy(input, emb + 8516, sizeof(float) * 768);
  memcpy(input + 768, state, sizeof(float) * 12 * 768);
  net();
  memcpy(state, output + 50277, sizeof(float) * 12 * 768);

  int idx = 8516;
  for (int i = 0; i < 100; i++) {{
    memcpy(input, emb + idx, sizeof(float) * {});
    memcpy(input + {}, state, sizeof(float) * {} * {});

    net();

    memcpy(state, output + 50277, sizeof(float) * {} * {});

    float best = -INFINITY;
    int best_idx = -1;
    for (int j = 0; j < 50277; j++) {{
      if (output[j] > best) {{
        best = output[j];
        best_idx = j;
      }}
    }}
    printf("Best: %d (%f)\\n", best_idx, best);

    idx = best_idx;
  }}
}}
""".format(
                model.embed_size,
                model.embed_size,
                model.layers,
                model.embed_size,
                model.layers,
                model.embed_size,
            )
        ]
        f.write("\n".join(cprog) + "\n")

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
    np.random.seed(42)

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
                temperature=0.0,
            )
            txt = tokenizer.decode([sampled])
            print(tokenizer.decode(x) + f"({txt}|{tokenizer.decode([y[-1]])})")

            optimizer.zero_grad()

            out = out.clip(1e-8, 1 - 1e-8)[-1]
            outnp = out.numpy()
            loss = out[0][int(y[0])]
            for j in range(1, y.shape[0]):
                outy = out[j][int(y[j])]
                loss = loss.cat(outy, dim=0)
            loss = -loss.log()
            loss = loss.mean()

            loss.backward()

            optimizer.step()

            loss = loss.numpy()
            print(f"epoch {epoch}, step {i}, loss {loss}")
