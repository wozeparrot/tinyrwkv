from tinygrad.tensor import Tensor
from tokenizers import Tokenizer
from tqdm import tqdm, trange
import numpy as np

import cli.subcommands.music.midi_llm_tokenizer.midi_util as midi_util

from argparse import Namespace, _SubParsersAction, ArgumentParser
import gc
import pathlib

from tinyrwkv import RWKV_RNN
from tinyrwkv.utils.sampling import sample_logits


def generate_parser(subparsers: "_SubParsersAction[ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "mus",
        help="music generation using the RNN mode (requires a preprocessed model using `pre`)",
    )
    parser.add_argument(
        "--tokenizer_path",
        help="path to the tokenizer file",
        type=str,
        default="tinyrwkv/vocab/tokenizer-midi.json",
    )
    parser.add_argument(
        "--model_path", help="path to the weights file", type=str, required=True
    )
    parser.add_argument(
        "--temperature", help="temperature (default: 1.1)", type=float, default=1.1
    )
    parser.add_argument("--top_k", help="topk (default: 12)", type=int, default=12)
    parser.add_argument(
        "--typical_tau", help="typical tau (default: 0.8)", type=float, default=0.8
    )
    parser.add_argument(
        "--alpha_presence",
        help="alpha presence (default: 0.0)",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--alpha_frequency",
        help="alpha frequency (default: 0.5)",
        type=float,
        default=0.5,
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

    # load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # load model
    model = RWKV_RNN(args.model_path)
    assert (
        model.vocab_size == tokenizer.get_vocab_size()
    ), f"vocab size mismatch (are you using the correct tokenizer?), model: {model.vocab_size}, tokenizer: {tokenizer.get_vocab_size()}"

    # encode initial context
    initial_context = "<pad> pi:4a:7"
    encoded_inital_context = tokenizer.encode(initial_context).ids

    print("Preprocessing...")
    state = model.init_state()
    for i in tqdm(range(len(encoded_inital_context) - 1)):
        embed = model.index_embed(encoded_inital_context[i])
        the_input = model.build_input(embed, state)
        out = model.forward(the_input)
        state = out[model.vocab_size:]
    last_token = encoded_inital_context[-1]

    gc.collect()

    generated = "<start> pi:4a:7"

    alpha_counter = np.zeros(model.vocab_size, dtype=np.float32)
    for i in trange(4096):
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
        else:
            logits[0] += (i - 2000) / 500

        # ignore t125
        logits[127] -= 1

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
        alpha_counter *= 0.997
        alpha_counter[sampled] += 1 if sampled >= 127 else 0.3

        # break if end of text token
        if sampled == 0:
            break

        # update last token
        last_token = sampled
        decoded = tokenizer.decode([sampled])
        generated += " " + decoded

    generated += " <end>"

    # convert text to midi
    cfg = midi_util.VocabConfig.from_json(
        pathlib.Path(__file__).parent / "midi_llm_tokenizer" / "vocab_config.json"
    )
    midi = midi_util.convert_str_to_midi(cfg, generated)
    midi.save("generated.mid")
