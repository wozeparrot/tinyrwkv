from tinygrad.lazy import Device
from tinygrad.nn.state import safe_save, safe_load
from tinygrad.tensor import Tensor
from tqdm import tqdm

from argparse import Namespace, _SubParsersAction, ArgumentParser
import json


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
    parser.add_argument(
        "--model_type",
        help="the type of model to use (default: world)",
        choices=["20b", "world", "midi"],
        default="world",
    )
    parser.add_argument(
        "--version",
        help="the version of the model to use (default: v4)",
        choices=["v4", "v5"],
        default="v4",
    )
    parser.set_defaults(func=preprocess)


def preprocess(args: Namespace) -> None:
    if args.input_path.endswith(".pth"):
        import torch

        weights_t = torch.load(args.input_path, map_location="cpu")
    elif args.input_path.endswith(".safetensors"):
        weights_t = safe_load(args.input_path)
    else:
        raise Exception("Unknown file type")

    # convert all weights to tinygrad
    weights = {}
    print("Converting weights to numpy...")
    for k, v in tqdm(weights_t.items()):
        if isinstance(v, Tensor):
            weights[k] = v.to(Device.DEFAULT).float().realize()
        else:
            weights[k] = Tensor(v.float().numpy())

    # precompute ln0 with emb.weight
    print("Precomputing emb.weight with ln0...")
    weights["emb.weight"] = (
        weights["emb.weight"]
        .layernorm()
        .linear(weights["blocks.0.ln0.weight"], weights["blocks.0.ln0.bias"])
    )
    del weights["blocks.0.ln0.weight"]
    del weights["blocks.0.ln0.bias"]

    # refine weights
    print("Refining weights...")
    for k, v in tqdm(weights.items()):
        if ".time_" in k:
            v = v.squeeze().realize()
        if ".time_decay" in k:
            if args.version == "v4":
                v = -v.exp().realize()
            elif args.version == "v5":
                v = (-v.exp()).exp().unsqueeze(-1).unsqueeze(-1).realize()
        if ".time_first" in k:
            if args.version == "v5":
                v = v.exp().unsqueeze(-1).unsqueeze(-1).realize()

        # convert to correct dtype
        if (
            args.dtype == "half"
            and ".time_decay" not in k
            and ".time_first" not in k
            and "emb.weight" not in k
        ):
            v = v.half()
        else:
            v = v.float()

        weights[k] = v.realize()

    print("Writing weights...")
    safe_save(weights, args.output_path)

    print("Writing info...")
    info = {
        "vocab_size": weights["emb.weight"].shape[0],
        "embed_size": weights["emb.weight"].shape[1],
        "layers": sum("ln1.weight" in k for k in weights.keys()),
        "dtype": args.dtype,
        "model_type": args.model_type,
        "version": args.version,
    }
    if args.version == "v5":
        info["n_heads"] = weights["blocks.0.att.time_decay"].shape[0]
        info["head_dim"] = weights["blocks.0.ln1.weight"].shape[0] // info["n_heads"]
    with open(args.output_path + ".json", "w") as f:
        json.dump(info, f)
