from tinygrad.tensor import Tensor
from tokenizers import Tokenizer
from tqdm import tqdm
import numpy as np

from argparse import Namespace, _SubParsersAction, ArgumentParser
import gc

from tinyrwkv.v4 import RWKV_RNN
from tinyrwkv.utils.sampling import sample_logits
from tinyrwkv.tokenizer import Tokenizer as WorldTokenizer


def generate_parser(subparsers: "_SubParsersAction[ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "gen",
        help="freeform generation using the RNN mode (requires a preprocessed model using `pre`)",
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
    parser.add_argument(
        "--ignore_eot",
        help="ignore end of text token (default: False)",
        default=False,
        action="store_true",
    )
    parser.set_defaults(func=generate)


def generate(args: Namespace) -> None:
    Tensor.no_grad = True

    # load model
    model = RWKV_RNN(args.model_path)

    # load tokenizer
    tokenizer = (
        Tokenizer.from_file(args.tokenizer_path)
        if model.model_type != "world"
        else WorldTokenizer()
    )

    # check vocab size
    assert (
        model.vocab_size == tokenizer.get_vocab_size()
        or model.model_type == "world"  # world tokenizer models are padded
    ), f"vocab size mismatch (are you using the correct tokenizer?), model: {model.vocab_size}, tokenizer: {tokenizer.get_vocab_size()}"

    # encode initial context
    initial_context = (
        args.prompt.replace("\\n", "\n").replace("\\r", "\r").replace("\r\n", "\n")
    )
    encoded_inital_context = tokenizer.encode(initial_context).ids

    print("Preprocessing...")
    state = model.init_state()
    for i in tqdm(range(len(encoded_inital_context) - 1)):
        embed = model.index_embed(encoded_inital_context[i])
        the_input = model.build_input(embed, state)
        out = model.forward(the_input)
        state = out[model.vocab_size :]
    last_token = encoded_inital_context[-1]

    print(f"\n{initial_context}", end="", flush=True)

    gc.collect()

    alpha_counter = np.zeros(model.vocab_size, dtype=np.float32)
    while True:
        embed = model.index_embed(int(last_token))
        the_input = model.build_input(embed, state)
        the_output = model.forward(the_input)
        logits = the_output[: model.vocab_size]
        state = the_output[model.vocab_size :]
        # logits to cpu
        logits = logits.cpu().numpy()

        # ignore end of text token if specified
        if args.ignore_eot:
            logits[0] = -1e9

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

        # update alpha counter
        alpha_counter[sampled] += 1

        # break if end of text token
        if sampled == 0:
            break

        # update last token
        last_token = sampled
        decoded = tokenizer.decode([sampled])
        print(decoded, end="", flush=True)
