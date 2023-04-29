from tinygrad.tensor import Tensor
from tqdm import tqdm
import numpy as np

import json
import gc
import pickle
from argparse import Namespace, _SubParsersAction, ArgumentParser


def generate_parser(subparsers: "_SubParsersAction[ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "pre",
        help="preprocess either tinyrwkv trained weights or pytorch trained weights into RNN form",
    )
    parser.add_argument(
        "--input_path", help="path to the weights file", type=str, required=True
    )
    parser.add_argument(
        "--output_path", help="path to the output file", type=str, required=True
    )
    parser.add_argument(
        "--dtype",
        choices=["float", "half"],
        default="float",
        help="data type to use (default: float)",
    )
    parser.set_defaults(func=preprocess)


def preprocess(args: Namespace) -> None:
    if args.input_path.endswith(".pth"):
        import torch

        weights = torch.load(args.input_path, map_location="cpu")
    elif args.input_path.endswith(".pkl"):
        with open(args.input_path, "rb") as f:
            weights = pickle.load(f)
    else:
        raise Exception("Unknown file type")

    # refine weights
    for k, v in tqdm(weights.items()):
        if args.dtype == "half" and "emb" not in k and "blocks.0.ln0" not in k:
            if isinstance(v, np.ndarray):
                v = v.astype(np.float16)
            else:
                v = v.half().numpy()
        else:
            if isinstance(v, np.ndarray):
                v = v.astype(np.float32)
            else:
                v = v.float().numpy()

        if ".time_" in k:
            v = v.squeeze()
        if ".time_decay" in k:
            v = -np.exp(v)

        weights[k] = v

    # precompute ln0 with emb.weight
    print("Precomputing emb.weight with ln0...")
    weights["emb.weight"] = (
        Tensor(weights["emb.weight"])
        .layernorm()
        .linear(
            Tensor(weights["blocks.0.ln0.weight"]), Tensor(weights["blocks.0.ln0.bias"])
        )
        .numpy()
    )

    print("Writing weights...")
    with open(args.output_path, "wb") as f:
        pickle.dump(weights, f)

    print("Writing info...")
    info = {
        "vocab_size": weights["emb.weight"].shape[0],
        "embed_size": weights["emb.weight"].shape[1],
        "layers": sum("ln1.weight" in k for k in weights.keys()),
        "dtype": args.dtype,
    }
    with open(args.output_path + ".json", "w") as f:
        json.dump(info, f)
