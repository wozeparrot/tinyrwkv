from tinygrad.tensor import Tensor
from tokenizers import Tokenizer
from tqdm import tqdm
import numpy as np

from argparse import Namespace, _SubParsersAction, ArgumentParser
import gc
import json

from tinyrwkv.utils.sampling import sample_logits
from tinyrwkv.tokenizer import Tokenizer as WorldTokenizer


def generate_parser(subparsers: "_SubParsersAction[ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "cht",
        help="chat with a model in RNN mode (requires a preprocessed model using `pre`)",
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
        "--temperature", help="temperature (default: 0.8)", type=float, default=0.8
    )
    parser.add_argument("--top_k", help="topk (default: 35)", type=int, default=35)
    parser.add_argument(
        "--typical_tau", help="typical tau (default: 0.9)", type=float, default=0.9
    )
    parser.add_argument(
        "--alpha_presence",
        help="alpha presence (default: 0.2)",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--alpha_frequency",
        help="alpha frequency (default: 0.2)",
        type=float,
        default=0.2,
    )
    parser.set_defaults(func=chat)


def chat(args: Namespace) -> None:
    Tensor.no_grad = True

    # get model version
    with open(args.model_path + ".json", "r") as f:
        version = json.load(f)["version"]
    if version == "v4":
        from tinyrwkv.v4 import RWKV_RNN
    elif version == "v5":
        from tinyrwkv.v5 import RWKV_RNN
    else:
        raise RuntimeError("Unknown model version")

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
    initial_context = """
The following is a conversation between a user and Pippa Pipkin, a rabbit virtual youtuber from Phase Connect. Pippa is quirky, super energetic, paranoid. She loves to mess with people and tell puns and stupid jokes.

User: hello, who are you?

Pippa: whats up! i'm pippa, a rabbit vtuber from phase connect.

User: that's pretty cool, wanna hear a joke?

Pippa: no, how about you shut the fuck up?

User: whoa, how about we just restart this conversation.

Pippa: sure, i'm pippa, what do you want?"""
    encoded_inital_context = tokenizer.encode(initial_context).ids

    state = model.init_state()
    for i in tqdm(range(len(encoded_inital_context))):
        embed = model.index_embed(encoded_inital_context[i])
        the_input = model.build_input(embed, state)
        out = model.forward(the_input)
        state = out[model.vocab_size :]

    gc.collect()

    alpha_counter = np.zeros(model.vocab_size)
    while True:
        print("\n\n", end="", flush=True)
        user_input = input("> ")
        if user_input == "":
            break
        user_input = f"\n\nUser: {user_input.strip()}\n\nPippa:"
        encoded_user_input = tokenizer.encode(user_input).ids
        for i in range(len(encoded_user_input) - 1):
            embed = model.index_embed(encoded_user_input[i])
            the_input = model.build_input(embed, state)
            out = model.forward(the_input)
            state = out[model.vocab_size :]
        last_token = encoded_user_input[-1]

        print("$", end="", flush=True)
        output = ""
        while True:
            embed = model.index_embed(int(last_token))
            the_input = model.build_input(embed, state)
            the_output = model.forward(the_input)
            logits = the_output[: model.vocab_size]
            # logits to cpu
            logits = logits.cpu().numpy()

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
            output += decoded

            # break on double newline
            if "\n" in output:
                break
            elif "User" in output:
                break
            else:
                # update state
                state = the_output[model.vocab_size :]
                print(decoded, end="", flush=True)
