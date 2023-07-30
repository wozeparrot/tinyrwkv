from tinygrad.tensor import Tensor
from tokenizers import Tokenizer
from tqdm import tqdm
import numpy as np

from argparse import Namespace, _SubParsersAction, ArgumentParser
import gc
import pickle

from tinyrwkv.v4 import RWKV_GPT
from tinyrwkv.utils.misc import get_child
from tinyrwkv.utils.model import count_parameters
from tinyrwkv.utils.sampling import sample_logits


def generate_parser(subparsers: "_SubParsersAction[ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "gpt",
        help="freeform generation using the GPT mode (requires a preprocessed model using `ptr`)",
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
        "--prompt",
        help='initial prompt (default: "The quick brown")',
        type=str,
        default="The quick brown",
    )
    parser.add_argument(
        "--temperature", help="temperature (default: 0.85)", type=float, default=0.85
    )
    parser.add_argument("--top_k", help="topk (default: 0)", type=int, default=0)
    parser.add_argument(
        "--typical_tau", help="typical tau (default: 0.95)", type=float, default=0.95
    )
    parser.add_argument(
        "--alpha_presence",
        help="alpha presence (default: 0.1)",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--alpha_frequency",
        help="alpha frequency (default: 0.1)",
        type=float,
        default=0.1,
    )
    parser.set_defaults(func=generate_gpt)


def generate_gpt(args: Namespace) -> None:
    Tensor.no_grad = True

    # load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # load model
    model = RWKV_GPT(args.model_path)
    print(f"model has ~{count_parameters(model) / 1000 / 1000}M parameters")
    assert (
        model.vocab_size == tokenizer.get_vocab_size()
    ), "vocab size mismatch (are you using the correct tokenizer?)"

    # encode initial context
    initial_context = (
        args.prompt.replace("\\n", "\n").replace("\\r", "\r").replace("\r\n", "\n")
    )
    ctx = tokenizer.encode(initial_context).ids
    print(f"\n{initial_context}", end="", flush=True)

    gc.collect()

    alpha_counter = np.zeros(50277)
    while True:
        the_input = Tensor([ctx])
        out = model.forward(the_input)
        logits = out.cpu().numpy()[-1][-1]

        # sample
        sampled = sample_logits(
            logits,
            alpha_counter=alpha_counter,
            alpha_presence=args.alpha_presence,
            alpha_frequency=args.alpha_frequency,
            temperature=args.temperature,
            typical_tau=args.typical_tau,
            top_k=args.top_k,
        )

        alpha_counter[sampled] += 1

        if sampled == 0:
            break

        decoded = tokenizer.decode([sampled])
        print(decoded, end="", flush=True)
        ctx = np.concatenate((ctx, [sampled]), axis=0)
