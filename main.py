from model_gpt import RWKV_GPT
from model_rnn import RWKV_RNN
from utils import sample_logits, get_child, count_parameters

from tinygrad.nn.optim import get_parameters, get_state_dict
from tinygrad.tensor import Tensor
from tokenizers import Tokenizer
import numpy as np

from tqdm import tqdm, trange
import gc
import pickle
import json
import sys
import os


np.set_printoptions(precision=4, suppress=True, linewidth=200)


if len(sys.argv) < 2:
    print("Usage: python main.py [pre|gen|gra|cmp|gpt|ptr|tra]")
    print("  pre: preprocess weights from pytorch or from training subcommand")
    print("    `python main.py pre <.pth | .pkl> <out.pkl> <float | half>`")
    print("  gen: generate text with the rnn mode")
    print("    `python main.py gen <.pkl> [prompt]`")
    print("    Run with JIT=1 OPTLOCAL=1 GPU=1 for much faster inference on gpu")
    print("  gra: use with GRAPH=1 to generate a graph of the rnn mode")
    print("    `GRAPH=1 python main.py gra <.pkl>`")
    print("  cmp: attempt to compile the rnn mode to c (must use float32 weights)")
    print("       outputs the compiled code to `out.c`")
    print("    `python main.py cmp <.pkl>`")
    print("  gpt: generate text with the gpt mode")
    print("    `python main.py gpt`")
    print("  ptr: preprocess pytorch weights into compatible format for training")
    print("    `python main.py ptr <.pth> <out.pkl>`")
    print("  tra: train with gpt mode")
    print(
        "    `python3 run.py tra <start_lr> <end_lr> <b1> <b2> <wd> <start_epoch> <epochs> <steps> <batch_size> <ctx_size> <ckpt_name>`"
    )
    sys.exit(1)

if sys.argv[1] == "pre":
    if len(sys.argv) < 5:
        print("Usage: python main.py pre <.pth | .pkl> <outfile> <float | half>")
        sys.exit(1)

    # load weights
    if sys.argv[2].endswith(".pth"):
        import torch

        weights = torch.load(sys.argv[2], map_location="cpu")
    elif sys.argv[2].endswith(".pkl"):
        import pickle

        with open(sys.argv[2], "rb") as f:
            weights = pickle.load(f)
    else:
        print("Unknown file type")
        sys.exit(1)

    # refine weights
    for k, v in tqdm(weights.items()):
        if sys.argv[4] == "half":
            if isinstance(v, np.ndarray):
                v = v.astype(np.float16)
            else:
                v = v.half().numpy()
        else:
            if isinstance(v, np.ndarray):
                v = v.astype(np.float32)
            else:
                v = v.float().numpy()

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

    print("Writing weights...")
    with open(sys.argv[3], "wb") as f:
        pickle.dump(weights, f)

    print("Writing info...")
    info = {
        "vocab_size": 50277,
        "embed_size": weights["emb.weight"].shape[1],
        "layers": sum("ln1.weight" in k for k in weights.keys()),
        "dtype": sys.argv[4],
    }
    with open(sys.argv[3] + ".json", "w") as f:
        json.dump(info, f)


elif sys.argv[1] == "gen":
    if len(sys.argv) < 3:
        print("Usage: python main.py gen <.pkl> [prompt]")
        sys.exit(1)

    Tensor.no_grad = True

    # load tokenizer
    tokenizer = Tokenizer.from_file("tokenizer.json")

    # load model
    model = RWKV_RNN(sys.argv[2])

    # jit
    from tinygrad.jit import TinyJit

    @TinyJit
    def run(x):
        ret = model.forward(x)
        return ret.realize()

    embed = Tensor(model.index_embed(510).numpy())
    state = model.init_state()
    the_input = model.build_input(embed, state)

    # run model twice to initialize the jit
    the_output = run(the_input)
    the_output = run(the_input)

    # encode initial context
    ctx_str = (
        """
This is a test of the emergency broadcast system. This is only a test. If this had been an actual emergency, you would have been instructed to do something. This concludes this test of the emergency broadcast system.
"""
        if len(sys.argv) < 4
        else sys.argv[3]
    )
    # convert \n in prompt to newline
    ctx_str = ctx_str.replace("\\n", "\n")
    ctx = tokenizer.encode(ctx_str).ids

    # encode separator
    sep = tokenizer.encode("\n\n").ids

    print("Preprocessing...")
    state = model.init_state()
    for i in tqdm(range(len(ctx))):
        x = np.concatenate([sep, ctx[:i]])
        embed = model.index_embed(int(x[-1]))
        the_input = model.build_input(embed, state)
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
        the_input = model.build_input(embed, state)
        the_output = run(the_input)
        logits = the_output[:50277]
        state = the_output[50277:]
        # logits to cpu
        logits = logits.cpu().numpy()

        # sample
        sampled = sample_logits(
            logits,
            alpha_counter=alpha_counter,
            alpha_presence=0.2,
            alpha_frequency=0.2,
            temperature=1.0,
            top_p=0.0,
            typical_p=0.2,
            top_k=50,
        )

        last_token = sampled
        tokens.append(last_token)
        alpha_counter[last_token] += 1

        last_decoded = tokenizer.decode([last_token])
        print(last_decoded, end="", flush=True)
        out += last_decoded
elif sys.argv[1] == "gra":
    if len(sys.argv) < 3:
        print("Usage: python main.py gra <.pkl>")
        sys.exit(1)

    Tensor.no_grad = True

    # seed random
    np.random.seed(42)

    # load model
    model = RWKV_RNN(sys.argv[3])

    embed = Tensor(model.index_embed(187).numpy())
    state = model.init_state()
    the_input = model.build_input(embed, state)

    print(model.forward(the_input))
elif sys.argv[1] == "cmp":
    if len(sys.argv) < 3:
        print("Usage: python main.py cmp <.pkl>")
        sys.exit(1)

    Tensor.no_grad = True

    # seed random
    np.random.seed(42)

    # load model
    model = RWKV_RNN(sys.argv[2])

    from tinygrad.jit import TinyJit

    @TinyJit
    def run(x):
        ret = model.forward(x)
        return ret.realize()

    embed = Tensor(model.index_embed(510).numpy())
    state = model.init_state()
    the_input = model.build_input(embed, state)

    # run model twice to initialize the jit
    the_output = run(the_input)
    the_output = run(the_input)

    for (j, i), idx in run.input_replace.items():
        run.jit_cache[j][1][i] = the_input.lazydata.realized

    special_names = {
        id(the_input.lazydata.realized): "input",
        id(the_output.lazydata.realized): "output",
    }

    from utils import compile_net

    with open("out.c", "w") as f:
        functions, statements, bufs, bufs_to_save = compile_net(run, special_names)

        cprog = [
            "#include <stdlib.h>",
            "#include <stdio.h>",
            "#include <string.h>",
            "#include <math.h>",
            "#define max(x,y) ((x>y)?x:y)",
            "#define half __fp16",
        ]
        f.write("\n".join(cprog) + "\n")

        # write init state
        cprog = [
            f"float *state[{len(state.lazydata.realized._buf)}];",
            "void init_state(float *state_data) {",
        ]
        with open("out.state.bin", "wb") as fe:
            cprog.append(
                f"memcpy(state, state_data, sizeof(float) * {len(state.lazydata.realized._buf)});"
            )
            fe.write(bytes(memoryview(state.lazydata.realized._buf)))
        cprog += ["}"]
        f.write("\n".join(cprog) + "\n")

        # buffers
        cprog = [
            f"{str(dtype)[7:]} {name}[{len}];"
            for name, len, dtype in bufs.values()
            if name != "input"
        ]
        f.write("\n".join(cprog) + "\n")

        # write weights
        cprog = ["void init_weights(float *weight_data) {"]
        weights_written = 0
        with open("out.weights.bin", "wb") as fw:
            for name, cl in tqdm(bufs_to_save.items()):
                cprog.append(
                    f"memcpy({name}, weight_data + {weights_written // 4}, sizeof({str(cl.dtype)[7:]}) * {len(cl._buf)});"
                )
                weights_written += fw.write(bytes(memoryview(cl._buf)))
        cprog += ["}"]
        f.write("\n".join(cprog) + "\n")

        # write embedding weights
        cprog = [
            f"float *emb[{len(model.emb.lazydata.realized._buf)}];",
            "void init_emb(float *emb_data) {",
        ]
        with open("out.emb.bin", "wb") as fe:
            cprog.append(
                f"memcpy(emb, emb_data, sizeof(float) * {len(model.emb.lazydata.realized._buf)});"
            )
            fe.write(bytes(memoryview(model.emb.lazydata.realized._buf)))
        cprog += ["}"]
        f.write("\n".join(cprog) + "\n")

        cprog = list(functions.values())
        f.write("\n".join(cprog) + "\n")

        cprog = (
            [
                "float *infer(float *input) {",
            ]
            + statements
            + ["return output;", "}"]
        )
        f.write("\n".join(cprog) + "\n")

        layers = 24
        dim = 1024
        cprog = [
            f"""
int main(int argc, char *argv[]) {{
  // load init state
  FILE *fs = fopen("out.state.bin", "rb");
  float *state_data = malloc({layers} * 5 * {dim} * sizeof(float));
  fread(state_data, 1, {len(state.lazydata.realized._buf)}, fs);
  fclose(fs);
  init_state(state_data);

  // load weights
  FILE *fw = fopen("out.weights.bin", "rb");
  float *weight_data = malloc({weights_written});
  fread(weight_data, 1, {weights_written}, fw);
  fclose(fw);
  init_weights(weight_data);

  // load embedding weights
  FILE *fe = fopen("out.emb.bin", "rb");
  float *emb_data = malloc({len(model.emb.lazydata.realized._buf)});
  fread(emb_data, 1, {len(model.emb.lazydata.realized._buf)}, fe);
  fclose(fe);
  init_emb(emb_data);

  float input[{dim} + {layers} * 5 * {dim}];

  // load input
  memcpy(input, emb + 510, sizeof(float) * {dim});
  memcpy(input + {dim}, state, sizeof(float) * {layers} * 5 * {dim});
  float *output = infer(input);
  memcpy(state, output + 50277, sizeof(float) * {layers} * 5 * {dim});
  memcpy(input, emb + 3158, sizeof(float) * {dim});
  memcpy(input + {dim}, state, sizeof(float) * {layers} * 5 * {dim});
  output = infer(input);
  memcpy(state, output + 50277, sizeof(float) * {layers} * 5 * {dim});

  int idx = 8516;
  for (int i = 0; i < 100; i++) {{
    memcpy(input, emb + idx, sizeof(float) * {dim});
    memcpy(input + {dim}, state, sizeof(float) * {layers} * 5 * {dim});

    output = infer(input);

    memcpy(state, output + 50277, sizeof(float) * {layers} * 5 * {dim});

    float best = -INFINITY;
    int best_idx = -1;
    for (int j = 0; j < 50277; j++) {{
      if (output[j] > best) {{
        best = output[j];
        best_idx = j;
      }}
    }}
    printf("%d, ", best_idx);
    fflush(stdout);

    idx = best_idx;
  }}
}}
"""
        ]
        f.write("\n".join(cprog) + "\n")

elif sys.argv[1] == "gpt":
    np.random.seed(42)

    # load tokenizer
    tokenizer = Tokenizer.from_file("tokenizer.json")

    # load model
    model = RWKV_GPT(1024, 50277, 768, 12)
    print(f"model has ~{count_parameters(model) / 1000 / 1000}M parameters")

    # load weights
    import torch

    weights = torch.load("./RWKV-4-Pile-169M-20220807-8023.pth", map_location="cpu")
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
    model.forward(Tensor(np.array([[187, 510]], dtype=np.float32)))

    # encode initial context
    ctx_str = "The quick brown"
    ctx = tokenizer.encode(ctx_str).ids

    # run model
    print(ctx_str, end="", flush=True)
    alpha_counter = np.zeros(50277)
    for i in range(100):
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
elif sys.argv[1] == "ptr":
    if len(sys.argv) < 3:
        print("usage: python3 run.py ptr <.pth> <out.pkl>")
        sys.exit(1)

    # load weights
    import torch

    weights = torch.load(sys.argv[2], map_location="cpu")
    # convert to tinygrad
    tg_weights = {}
    for k, v in tqdm(weights.items()):
        tg_weights[k] = v.float().numpy()
    del weights

    # write weights
    with open(sys.argv[3], "wb") as f:
        pickle.dump(tg_weights, f)
elif sys.argv[1] == "tra":
    if len(sys.argv) < 13:
        print(
            "usage: python3 run.py tra <start_lr> <end_lr> <b1> <b2> <wd> <start_epoch> <epochs> <steps> <batch_size> <ctx_size> <ckpt_name>"
        )
        sys.exit(1)

    start_lr = float(sys.argv[2])
    end_lr = float(sys.argv[3])
    b1 = float(sys.argv[4])
    b2 = float(sys.argv[5])
    wd = float(sys.argv[6])
    start_epoch = int(sys.argv[7])
    epochs = int(sys.argv[8])
    steps = int(sys.argv[9])
    batch_size = int(sys.argv[10])
    ctx_size = int(sys.argv[11])
    ckpt_name = sys.argv[12]

    # load tokenizer
    tokenizer = Tokenizer.from_file("tokenizer.json")

    # load model
    model = RWKV_GPT(ctx_size, 50277, 768, 12)
    print(f"model has ~{count_parameters(model) / 1000 / 1000}M parameters")

    # load weights
    with open(f"tra_ckpts/{ckpt_name}.weights.pkl", "rb") as f:
        weights = pickle.load(f)

    loaded = 0
    skipped = 0
    for k, v in tqdm(weights.items()):
        try:
            w = get_child(model, k)
            loaded += 1
        except:
            w = None
            skipped += 1
        if w is not None:
            assert w.shape == v.shape
            w.assign(v)

    print(f"loaded {loaded} weights, skipped {skipped} weights")
    del weights
    gc.collect()

    from tinygrad.nn.optim import AdamW
    from tinygrad.extra.training import sparse_categorical_crossentropy

    print("starting optimizer...")
    params = get_parameters(model)
    optimizer = AdamW(params, lr=start_lr, b1=b1, b2=b2, wd=wd)

    # load optimizer state if it exists
    if os.path.exists(f"tra_ckpts/{ckpt_name}.optimizer.pkl"):
        with open(f"tra_ckpts/{ckpt_name}.optimizer.pkl", "rb") as f:
            optimizer_state = pickle.load(f)

        optimizer.t.assign(optimizer_state["t"])
        for i in range(len(optimizer.m)):
            optimizer.m[i].assign(optimizer_state["m"][i])
        for i in range(len(optimizer.v)):
            optimizer.v[i].assign(optimizer_state["v"][i])

    # scale learning rate according to start epoch
    lr_decay = (end_lr / start_lr) ** (1 / epochs)
    for i in range(start_epoch):
        optimizer.lr *= lr_decay

    print("done starting optimizer")
    gc.collect()

    print("loading training data...")
    train_data = np.load("train.npy").astype(int)
    print("done loading training data")
    gc.collect()

    Tensor.training = True
    optimizer.zero_grad()
    for epoch in range(start_epoch, epochs):
        for step in (t := trange(steps)):
            sample = np.random.randint(
                0, len(train_data) - (model.ctx_size + 1), size=batch_size
            )
            sampled = [
                train_data[samp : samp + (model.ctx_size + 1)] for samp in sample
            ]

            x = Tensor([samp[:-1] for samp in sampled], requires_grad=False)
            y = np.array([samp[1:] for samp in sampled])

            out = model.forward(x)

            loss = sparse_categorical_crossentropy(out.log_softmax(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = (np.argmax(out.numpy(), axis=-1) == y).mean()
            loss = loss.numpy()
            t.set_description("loss %.4f acc %.4f" % (loss, accuracy))

        # save model
        print("saving model...")
        with open(f"epoch_{epoch + 1}.weights.pkl", "wb") as f:
            weights = {}
            for key, param in get_state_dict(model).items():
                weights[key] = param.numpy()
            pickle.dump(weights, f)

        # save optimizer
        print("saving optimizer...")
        with open(f"epoch_{epoch + 1}.optimizer.pkl", "wb") as f:
            t = optimizer.t.numpy()
            m = []
            for tensor in optimizer.m:
                m.append(tensor.numpy())
            v = []
            for tensor in optimizer.v:
                v.append(tensor.numpy())

            pickle.dump(
                {
                    "t": t,
                    "m": m,
                    "v": v,
                },
                f,
            )

        # decay learning rate
        lr_decay = (end_lr / start_lr) ** (1 / epochs)
        optimizer.lr *= lr_decay
