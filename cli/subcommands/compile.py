from tinygrad.tensor import Tensor
from tqdm import tqdm

from argparse import Namespace, _SubParsersAction, ArgumentParser
import os
import shutil
import subprocess

from tinyrwkv import RWKV_RNN
from tinyrwkv.utils.model import compile_net


def generate_parser(subparsers: "_SubParsersAction[ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "cmp",
        help="compile a RNN model into c source code and a compiled executable (need to run with CLANG=1)",
    )
    parser.add_argument(
        "--tokenizer_path",
        help="path to the tokenizer file",
        type=str,
        default="tokenizer.json",
    )
    parser.add_argument(
        "--model_path", help="path to the weights file", type=str, required=True
    )
    parser.add_argument(
        "--output_path",
        help="directory to output the compiled model",
        type=str,
        required=True,
    )
    parser.set_defaults(func=compile)


def compile(args: Namespace) -> None:
    Tensor.no_grad = True
    # ensure that we are running with CLANG=1
    assert "CLANG" in os.environ, "need to run with environment variable CLANG=1"

    # load model
    model = RWKV_RNN(args.model_path)
    assert model.dtype == "float", "only float supported for now"

    # run model twice to initialize the jit
    embed = Tensor(model.index_embed(0).numpy())
    state = model.init_state()
    the_input = model.build_input(embed, state)

    the_output = model.forward(the_input)
    the_output = model.forward(the_input)

    # fix some stuff
    jitted = model.forward.func.__self__
    for (j, i), _ in jitted.input_replace.items():
        jitted.jit_cache[j][1][i] = the_input.lazydata.realized

    # make the output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # copy the tokenizer
    shutil.copy(args.tokenizer_path, args.output_path)

    # generate the c code
    with open(os.path.join(args.output_path, "main.c"), "w") as f:
        functions, statements, bufs, bufs_to_save = compile_net(
            jitted,
            {
                id(the_input.lazydata.realized): "input",
                id(the_output.lazydata.realized): "output",
            },
        )

        cprog = [
            "#include <assert.h>",
            "#include <math.h>",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            "#include <time.h>",
            "#include <sys/mman.h>",
            "#include <sys/stat.h>",
            "#include <sys/types.h>",
            "#include <fcntl.h>",
            '#include "tokenizers.h"',
            "#define max(x,y) ((x>y)?x:y)",
            "#define half __fp16",
        ]
        f.write("\n".join(cprog) + "\n")

        # write init state
        print("Writing state...")
        with open(os.path.join(args.output_path, "state.bin"), "wb") as fe:
            fe.write(state.lazydata.realized._buffer())

        # buffers
        cprog = [
            f"{str(dtype)[7:]} {name}[{len}];"
            if name not in bufs_to_save
            else f"{str(dtype)[7:]} *{name};"
            for name, len, dtype in bufs.values()
            if name != "input"
        ]
        f.write("\n".join(cprog) + "\n")

        # write weights
        print("Writing weights...")
        cprog = [f"void init_weights({model.dtype} *weight_data) {{"]
        weights_written = 0
        with open(os.path.join(args.output_path, "weights.bin"), "wb") as fw:
            for name, cl in tqdm(bufs_to_save.items()):
                cprog.append(f"{name} = weight_data + {weights_written // 4};")
                weights_written += fw.write(cl._buffer())
        cprog += ["}"]
        f.write("\n".join(cprog) + "\n")

        # write embedding weights
        print("Writing embedding weights...")
        with open(os.path.join(args.output_path, "emb.bin"), "wb") as fe:
            fe.write(model.emb.lazydata.realized._buffer())

        cprog = list(functions.values())
        f.write("\n".join(cprog) + "\n")

        cprog = (
            [
                "void infer(float *input) {",
            ]
            + statements
            + ["}"]
        )
        f.write("\n".join(cprog) + "\n")

        layers = model.layers
        dim = model.embed_size
        cprog = [
            """
typedef struct {
  int index;
  float value;
} index_value;

int compare_index_value(const void *a, const void *b) {
  index_value *ia = (index_value *)a;
  index_value *ib = (index_value *)b;
  return (int)(100.f * (ib->value - ia->value));
}

int sample(float *logits, float temperature, float tau) {
  // try to bypass nan
  for (unsigned int i = 0; i < 50277; i++) {
    if (logits[i] != logits[i]) {
      logits[i] = 0;
    }
  }

  if (temperature == 0) {
    // greedy sampling
    float max = -INFINITY;
    int max_index = 0;
    for (unsigned int i = 0; i < 50277; i++) {
      if (logits[i] > max) {
        max = logits[i];
        max_index = i;
      }
    }
    return max_index;
  }

  // -- typical sampling --
  // softmax
  float exp_sum = 0;
  for (unsigned int i = 0; i < 50277; i++) {
    logits[i] = expf(logits[i]);
    exp_sum += logits[i];
  }

  float probs[50277];
  for (unsigned int i = 0; i < 50277; i++) {
    probs[i] = logits[i] / exp_sum;
  }

  // entropy
  float entropy = 0;
  for (unsigned int i = 0; i < 50277; i++) {
    logits[i] = -logf(probs[i]);
    if (logits[i] == logits[i]) {
      entropy += probs[i] * logits[i];
    }
  }
  for (unsigned int i = 0; i < 50277; i++) {
    logits[i] = fabsf(logits[i] - entropy);
  }

  // sort keeping track of indices
  index_value iv[50277];
  for (unsigned int i = 0; i < 50277; i++) {
    iv[i].index = i;
    iv[i].value = logits[i];
  }
  qsort(iv, 50277, sizeof(index_value), compare_index_value);

  // sort probs using indices
  float sorted_probs[50277];
  for (unsigned int i = 0; i < 50277; i++) {
    sorted_probs[i] = probs[iv[i].index];
  }

  // cumulative sum
  float cumsum[50277];
  cumsum[0] = sorted_probs[0];
  for (unsigned int i = 1; i < 50277; i++) {
    cumsum[i] = cumsum[i - 1] + sorted_probs[i];
  }

  // calculate cutoff
  int cutoff = 0;
  for (unsigned int i = 0; i < 50277; i++) {
    if (cumsum[i] < tau) {
      cutoff += 1;
    } else
      break;
  }

  // set probs to 0 if logits greater than cutoff
  for (unsigned int i = 0; i < 50277; i++) {
    if (logits[i] > iv[cutoff].value) {
      probs[i] = 0;
    }
  }

  // temperature
  for (unsigned int i = 0; i < 50277; i++) {
    probs[i] = powf(probs[i], 1.0 / temperature);
  }

  // normalize
  float sum = 0;
  for (unsigned int i = 0; i < 50277; i++) {
    sum += probs[i];
  }
  for (unsigned int i = 0; i < 50277; i++) {
    probs[i] = probs[i] / sum;
  }

  // sample
  float r = (float)rand() / (float)RAND_MAX;
  float cumsum2 = 0;
  for (unsigned int i = 0; i < 50277; i++) {
    cumsum2 += probs[i];
    if (r < cumsum2) {
      return i;
    }
  }
  return 0;
}
            """,
            f"""
int main(int argc, char *argv[]) {{
  // init random
  srand(time(NULL));

  fprintf(stderr, "Loading embedding weights...\\n");
  // load embedding weights using mmap
  int fe = open("emb.bin", O_RDONLY);
  struct stat fesb;
  fstat(fe, &fesb);
  float *emb = mmap(NULL, fesb.st_size, PROT_READ, MAP_SHARED, fe, 0);
  assert(emb != MAP_FAILED);

  fprintf(stderr, "Loading weights...\\n");
  // load weights using mmap
  int fw = open("weights.bin", O_RDONLY);
  struct stat fwsb;
  fstat(fw, &fwsb);
  {model.dtype} *weight_data = mmap(NULL, fwsb.st_size, PROT_READ, MAP_SHARED, fw, 0);
  assert(weight_data != MAP_FAILED);
  init_weights(weight_data);

  fprintf(stderr, "Loading initial state...\\n");
  // load init state using mmap
  int fs = open("state.bin", O_RDONLY);
  struct stat fssb;
  fstat(fs, &fssb);
  float *state_data = mmap(NULL, fssb.st_size, PROT_READ, MAP_SHARED, fs, 0);
  assert(state_data != MAP_FAILED);

  fprintf(stderr, "Loading tokenizer...\\n");
  // setup tokenizer
  void *tokenizer = tk_from_file("tokenizer.json");
  if (tokenizer == NULL) {{
    fprintf(stderr, "Failed to load tokenizer\\n");
    return 1;
  }}

  // read temperature and tau from args
  float temperature = 0.85;
  float tau = 0.95;
  if (argc > 2) {{
    temperature = atof(argv[1]);
    tau = atof(argv[2]);
  }}
  fprintf(stderr, "Using temperature: %f, tau: %f\\n", temperature, tau);

  // setup input
  float *input = malloc(sizeof(float) * ({dim} + {layers} * 5 * {dim}));
  memcpy(input + {dim}, state_data, sizeof(float) * {layers} * 5 * {dim});

  // input string from stdin
  char input_str[4096];
  int read = fread(&input_str, sizeof(input_str), 1, stdin);

  printf("%s", input_str);
  fflush(stdout);

  // tokenize input
  unsigned int tokenized_len;
  unsigned int *input_tokens = tk_encode(tokenizer, input_str, &tokenized_len);

  // preprocess input by running it through the model
  for (int i = 0; i < tokenized_len - 1; i++) {{
    memcpy(input, emb + (input_tokens[i] * {dim}), sizeof(float) * {dim});
    infer(input);
    memcpy(input + {dim}, output + 50277, sizeof(float) * {layers} * 5 * {dim});
  }}
  free(input_tokens);

  unsigned int last_token = input_tokens[tokenized_len - 1];

  // run model
  while (1) {{
    memcpy(input, emb + (last_token * {dim}), sizeof(float) * {dim});
    infer(input);
    memcpy(input + {dim}, output + 50277, sizeof(float) * {layers} * 5 * {dim});

    // -- sampling --
    last_token = sample(output, 0.85, 0.95);
    if (last_token == 0 && argc < 4) break;

    char *decoded = tk_decode(tokenizer, &last_token, 1);
    printf("%s", decoded);
    fflush(stdout);
    free(decoded);
  }}

  // cleanup
  free(input);
  tk_free(tokenizer);
}}
""",
        ]
        f.write("\n".join(cprog) + "\n")

    # compile
    print("Compiling...")
    try:
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
            ],
            cwd=os.path.join(os.path.dirname(__file__), "../../deps/tokenizers2c/"),
            check=True,
        )
    except FileNotFoundError:
        print("cargo not found, skipping compilation")
        return
    except subprocess.CalledProcessError:
        print("cargo build failed, skipping compilation")
        return

    try:
        subprocess.run(
            [
                "clang",
                os.path.join(args.output_path, "main.c"),
                f'-I{os.path.join(os.path.dirname(__file__), "../../deps/tokenizers2c/")}',
                "-o",
                os.path.join(args.output_path, "tinyrwkv"),
                "-Ofast",
                "-ffast-math",
                "-march=native",
                "-flto",
                "-fPIC",
                "-lm",
                "-ltokenizers2c",
                "-lunwind",
                "--rtlib=compiler-rt",
                f'-L{os.path.join(os.path.dirname(__file__), "../../deps/tokenizers2c/target/release/")}',
                "-s",
            ],
            check=True,
        )
    except FileNotFoundError:
        print("clang not found, skipping compilation")
        return
    except subprocess.CalledProcessError:
        print("clang build failed, skipping compilation")
        return
