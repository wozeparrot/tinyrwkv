from tinygrad.tensor import Tensor
from tqdm import tqdm, trange

from argparse import Namespace, _SubParsersAction, ArgumentParser
import time

from tinyrwkv.wkv import OpenCLWKV, StdWKV, ConvWKV


def generate_parser(subparsers: "_SubParsersAction[ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "wkv",
        help="benchmark/test each wkv module",
    )
    parser.set_defaults(func=test_wkv)


def get_error(a: Tensor, b: Tensor) -> float:
    err = (a - b).flatten().square().mean().sqrt().numpy().item()
    base = a.flatten().square().mean().sqrt().numpy().item()
    return err / base


def test_wkv(args: Namespace) -> None:
    # initial testing runs are done with batch size of 1 because ConvWKV only supports a batch size of 1
    B = 1
    T = 16
    C = 768

    u0 = Tensor.uniform(C, requires_grad=False)
    w0 = Tensor.uniform(C, requires_grad=False)
    k0 = Tensor.uniform(B, T, C, requires_grad=False)
    v0 = Tensor.uniform(B, T, C, requires_grad=False)

    # first make sure that the output for each is the same or close enough
    # StdWKV is the reference implementation
    u1 = Tensor.zeros(C, requires_grad=True) + u0
    w1 = Tensor.zeros(C, requires_grad=True) + w0
    k1 = Tensor.zeros(B, T, C, requires_grad=True) + k0
    v1 = Tensor.zeros(B, T, C, requires_grad=True) + v0

    f1 = StdWKV()(B, T, C, u1, w1, k1, v1)

    # l1 = ((f1 * f1) - f1.tanh()).sum()
    # l1.backward()

    u2 = Tensor.zeros(C, requires_grad=True) + u0
    w2 = Tensor.zeros(C, requires_grad=True) + w0
    k2 = Tensor.zeros(B, T, C, requires_grad=True) + k0
    v2 = Tensor.zeros(B, T, C, requires_grad=True) + v0

    f2 = ConvWKV()(B, T, C, u2, w2, k2, v2)
    print(f"ConvWKV error: {get_error(f1, f2)}")

    l2 = ((f2 * f2) - f2.tanh()).sum()
    l2.backward()

    u3 = Tensor.zeros(C, requires_grad=True) + u0
    w3 = Tensor.zeros(C, requires_grad=True) + w0
    k3 = Tensor.zeros(B, T, C, requires_grad=True) + k0
    v3 = Tensor.zeros(B, T, C, requires_grad=True) + v0

    f3 = OpenCLWKV()(B, T, C, u3, w3, k3, v3)
    print(f"OpenCLWKV error: {get_error(f1, f3)}")

    l3 = ((f3 * f3) - f3.tanh()).sum()
    l3.backward()

    print(f"gu error: {get_error(u2.grad, u3.grad)}")
    print(f"gw error: {get_error(w2.grad, w3.grad)}")
    print(f"gk error: {get_error(k2.grad, k3.grad)}")
    print(f"gv error: {get_error(v2.grad, v3.grad)}")

    # now = time.time()
    # for _ in trange(20):
    #     pass
    # end = time.time()
    # diff = end - now
    # print(f"20 runs in {diff:.2f}s")
    # print(f"runs per second: {20 / diff:.2f}")
