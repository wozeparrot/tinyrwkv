"""Main CLI entry point"""

from tinygrad.tensor import Tensor
import numpy as np

import argparse

from .subcommands import (
    preprocess,
    generate,
    compile,
    preprocess_gpt,
    generate_gpt,
    train,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tinyrwkv-cli", description="CLI for tinyrwkv"
    )
    parser.add_argument("--seed", help="seed for random", type=int, default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)

    # preprocess subcommand
    preprocess.generate_parser(subparsers)

    # generate subcommand
    generate.generate_parser(subparsers)

    # compile subcommand
    compile.generate_parser(subparsers)

    # preprocess gpt subcommand
    preprocess_gpt.generate_parser(subparsers)

    # generate gpt subcommand
    generate_gpt.generate_parser(subparsers)

    # train subcommand
    train.generate_parser(subparsers)

    args = parser.parse_args()
    np.random.seed(args.seed)
    Tensor.manual_seed(args.seed)

    args.func(args)


if __name__ == "__main__":
    main()
