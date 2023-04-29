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

    # copy the main.c file
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "compile/main.c"), args.output_path
    )

    # generate the c code
    # start with the static header
    with open(os.path.join(os.path.dirname(__file__), "compile/tinyrwkv.h"), "r") as f:
        header = f.read()

    with open(os.path.join(args.output_path, "tinyrwkv.h"), "w") as f:
        functions, statements, bufs, bufs_to_save = compile_net(
            jitted,
            {
                id(the_input.lazydata.realized): "input",
                id(the_output.lazydata.realized): "output",
            },
        )

        f.write(
            header.replace(
                "#define TINYRWKV_DIM 0", f"#define TINYRWKV_DIM {model.embed_size}"
            )
            .replace(
                "#define TINYRWKV_LAYERS 0", f"#define TINYRWKV_LAYERS {model.layers}"
            )
            .replace(
                "#define TINYRWKV_DTYPE float", f"#define TINYRWKV_DTYPE {model.dtype}"
            )
            + "\n"
        )

        cprog = [
            "#include <math.h>",
            "#define max(x,y) ((x>y)?x:y)",
        ]
        f.write("\n".join(cprog) + "\n")

        # write init state
        print("Writing state...")
        with open(os.path.join(args.output_path, "state.bin"), "wb") as fe:
            fe.write(state.lazydata.realized._buffer())

        # scratch buffers
        scratch_buffers = [
            f"{str(dtype)[7:]} {name}[{len}];"
            for name, len, dtype in bufs.values()
            if name not in ("input", "output") and name not in bufs_to_save
        ]

        # write weights
        print("Writing weights...")
        with open(os.path.join(args.output_path, "weights.bin"), "wb") as fw:
            for _, cl in tqdm(bufs_to_save.items()):
                fw.write(cl._buffer())

        # write embedding weights
        print("Writing embedding weights...")
        with open(os.path.join(args.output_path, "emb.bin"), "wb") as fe:
            fe.write(model.emb.lazydata.realized._buffer())

        cprog = list(functions.values())
        f.write("\n".join(cprog) + "\n")

        cprog = (
            [
                "void tinyrwkv_infer(float *input, float *output, TINYRWKV_DTYPE *weights) {",
            ]
            + scratch_buffers
            + statements
            + ["}"]
        )
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
