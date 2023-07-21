from tinygrad.tensor import Tensor

from argparse import Namespace, _SubParsersAction, ArgumentParser
import os
import shutil

from tinyrwkv import RWKV_RNN
from tinyrwkv.utils.model import compile_net_js


def generate_parser(subparsers: "_SubParsersAction[ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "cjs",
        help="compile a RNN model into js source code that runs with webgpu (need to run with WEBGPU=1)",
    )
    parser.add_argument(
        "--tokenizer_path",
        help="path to the tokenizer file",
        type=str,
        default="tinyrwkv/vocab/tokenizer.json",
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
    # ensure that we are running with WEBGPU=1
    assert "WEBGPU" in os.environ, "need to run with environment variable WEBGPU=1"

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

    # copy the tokenizer vocab
    shutil.copy(args.tokenizer_path, args.output_path)

    # copy the index.html file
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "compile_webgpu/index.html"),
        args.output_path,
    )

    # generate the js code
    functions, statements, bufs, bufs_to_save = compile_net_js(
        jitted,
        {
            id(the_input.lazydata.realized): "input",
            id(the_output.lazydata.realized): "output",
        },
    )
