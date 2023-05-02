from tinygrad.tensor import Tensor
from tokenizers import Tokenizer
from tqdm import tqdm, trange

from argparse import Namespace, _SubParsersAction, ArgumentParser
import time

from tinyrwkv import RWKV_RNN


def generate_parser(subparsers: "_SubParsersAction[ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "bch",
        help="benchmark the rnn mode",
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
    parser.set_defaults(func=benchmark)


def benchmark(args: Namespace) -> None:
    Tensor.no_grad = True

    # load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # load model
    model = RWKV_RNN(args.model_path)
    assert (
        model.vocab_size == tokenizer.get_vocab_size()
    ), "vocab size mismatch (are you using the correct tokenizer?)"

    last_token = 510
    state = model.init_state()

    # warmup 10 tokens
    for i in trange(10):
        embed = model.index_embed(int(last_token))
        the_input = model.build_input(embed, state)
        the_output = model.forward(the_input)
        state = the_output[50277:]

    last_token = 510
    state = model.init_state()

    # benchmark 1000 tokens
    now = time.time()
    for i in trange(1000):
        embed = model.index_embed(int(last_token))
        the_input = model.build_input(embed, state)
        the_output = model.forward(the_input)
        state = the_output[50277:]
    end = time.time()
    diff = end - now
    print(f"1000 tokens in {diff:.2f}s")
    print(f"tokens per second: {1000 / diff:.2f}")