from tinygrad.tensor import Tensor


def matvec(mat: Tensor, vec: Tensor) -> Tensor:
    return vec @ mat.T


def elemmax(x: Tensor, y: Tensor) -> Tensor:
    return (x > y) * (x - y) + y
