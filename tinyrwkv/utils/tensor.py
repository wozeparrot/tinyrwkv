from tinygrad.tensor import Tensor


def matvec(mat: Tensor, vec: Tensor) -> Tensor:
    return vec @ mat.T


def elemmax(x: Tensor, y: Tensor) -> Tensor:
    xgty = x.sub(y).relu() - (x.sub(y) - 1).relu()
    ygtx = y.sub(x).relu() - (y.sub(x) - 1).relu()
    return xgty * x + ygtx * y
