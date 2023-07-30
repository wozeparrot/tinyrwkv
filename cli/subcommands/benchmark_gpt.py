from tinygrad.tensor import Tensor
from tokenizers import Tokenizer
from tqdm import tqdm, trange

from argparse import Namespace, _SubParsersAction, ArgumentParser
import time

from tinyrwkv.v4 import RWKV_GPT


def generate_parser(subparsers: "_SubParsersAction[ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "bpt",
        help="benchmark the gpt mode",
    )
    parser.add_argument(
        "--model_path", help="path to the weights file", type=str, required=True
    )
    parser.set_defaults(func=benchmark)


def benchmark(args: Namespace) -> None:
    Tensor.no_grad = True

    # load model
    model = RWKV_GPT(args.model_path)

    the_input = Tensor([[187] * 512]).realize()

    # warmup 4 runs
    for _ in trange(4):
        model.forward(the_input)

    # benchmark 20 runs
    now = time.time()
    for _ in trange(20):
        model.forward(the_input)
    end = time.time()
    diff = end - now
    print(f"20 runs in {diff:.2f}s")
    print(f"runs per second: {20 / diff:.2f}")
