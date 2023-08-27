from tinygrad.nn.state import safe_save
from tqdm import tqdm
import torch

import json
from argparse import Namespace, _SubParsersAction, ArgumentParser


def generate_parser(subparsers: "_SubParsersAction[ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "ptr",
        help="preprocess pytorch weights weights into GPT form for training or inference",
    )
    parser.add_argument(
        "--input_path", help="path to the weights file", type=str, required=True
    )
    parser.add_argument(
        "--output_path", help="path to the output file", type=str, required=True
    )
    parser.set_defaults(func=preprocess)


def preprocess(args: Namespace) -> None:
    weights = torch.load(args.input_path, map_location="cpu")

    # convert to tinygrad
    for k, v in tqdm(weights.items()):
        weights[k] = v.float().numpy()

    print("Writing weights...")
    safe_save(weights, args.output_path)

    print("Writing info...")
    info = {
        "vocab_size": weights["emb.weight"].shape[0],
        "embed_size": weights["emb.weight"].shape[1],
        "layers": sum("ln1.weight" in k for k in weights.keys()),
        "dtype": "float",
        "model_type": "20b",
        "version": "v4",
    }
    with open(args.output_path + ".json", "w") as f:
        json.dump(info, f)
