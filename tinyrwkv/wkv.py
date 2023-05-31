from tinygrad.helpers import dtypes, prod
from tinygrad.lazy import LazyBuffer, create_lazybuffer, Device
from tinygrad.ops import ASTRunner, LazyOp, LoadOps, BinaryOps, UnaryOps, MovementOps
from tinygrad.tensor import Tensor, Function

import abc
import os
from typing import Optional, Tuple


# abstract base class for WKV kernels
class WKV(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        B: int,
        T: int,
        C: int,
        time_first: Tensor,
        time_decay: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        raise NotImplementedError


# standard WKV kernel, same as upstream cuda kernel
class StdWKV(WKV):
    def __call__(
        self,
        B: int,
        T: int,
        C: int,
        time_first: Tensor,
        time_decay: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        time_w = time_decay.exp()

        # state
        aa = Tensor.zeros(B, 1, C, requires_grad=False)
        bb = Tensor.zeros(B, 1, C, requires_grad=False)
        pp = Tensor.ones(B, 1, C, requires_grad=False) * -1e38

        wkv: list[Tensor] = []
        # calculate for all tokens up to the last one
        for i in range(T - 1):
            kk = key[:, i, :]
            vv = value[:, i, :]

            ww = kk + time_first
            p = pp.maximum(ww)
            e1 = (pp - p).exp()
            e2 = (ww - p).exp()
            a = (e1 * aa) + (e2 * vv)
            b = (e1 * bb) + e2
            wkv.append(a / b)

            ww = pp - time_w
            p = ww.maximum(kk)
            e1 = (ww - p).exp()
            e2 = (kk - p).exp()
            aa = ((e1 * aa) + (e2 * vv)).realize()
            bb = ((e1 * bb) + e2).realize()
            pp = p.realize()

        # calculate the last token outside to avoid having to compute the next state
        kk = key[:, -1, :]
        vv = value[:, -1, :]
        ww = kk + time_first
        p = pp.maximum(ww)
        e1 = (pp - p).exp()
        e2 = (ww - p).exp()
        a = (e1 * aa) + (e2 * vv)
        b = (e1 * bb) + e2
        wkv.append(a / b)

        return Tensor.cat(*wkv, dim=1).realize()


# convolutional WKV kernel, almost the same as rwkv < 4 wkv
# faster in tinygrad than the standard WKV kernel
# only works for B = 1
class ConvWKV(WKV):
    def __call__(
        self,
        _: int,
        T: int,
        C: int,
        time_first: Tensor,
        time_decay: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        ek = key.clip(-60, 60).transpose(1, 2).exp()
        ekv = ek * value.transpose(1, 2)

        time_curve = Tensor(
            [-(T - 2 - i) for i in range(T - 1)], requires_grad=False
        ).unsqueeze(0)

        time_w = (time_decay.exp().unsqueeze(1) * time_curve).cat(
            time_first.unsqueeze(1), dim=-1
        )
        w = time_w.exp().unsqueeze(1)
        w = w.reshape(w.shape[0], w.shape[1], w.shape[2], 1)

        ekv = (
            ekv.reshape(1, *ekv.shape)
            .pad2d((T - 1, 0, 0, 0))
            .reshape(ekv.shape[0], ekv.shape[1], ekv.shape[2] + T - 1, 1)
        )
        wkv = ekv.conv2d(w, groups=C).reshape(
            ekv.shape[0], ekv.shape[1], ekv.shape[2] - T + 1
        )
        ek = (
            ek.reshape(1, *ek.shape)
            .pad2d((T - 1, 0, 0, 0))
            .reshape(ek.shape[0], ek.shape[1], ek.shape[2] + T - 1, 1)
        )
        wk = (
            ek.conv2d(w, groups=C).reshape(
                ek.shape[0], ek.shape[1], ek.shape[2] - T + 1
            )
            + 1e-8
        )

        wkv = (wkv / wk).transpose(1, 2)

        return wkv


# load kernels once for opencl wkv
with open(os.path.join(os.path.dirname(__file__), "kernels/wkv_forward.cl"), "r") as f:
    WKV_FORWARD_PRG = f.read()
with open(os.path.join(os.path.dirname(__file__), "kernels/wkv_backward.cl"), "r") as f:
    WKV_BACKWARD_PRG = f.read()


# opencl WKV forward runner
def opencl_wkv_forward(
    ret: LazyBuffer,
    shape: LazyBuffer,
    time_first: LazyBuffer,
    time_decay: LazyBuffer,
    key: LazyBuffer,
    value: LazyBuffer,
):
    ret.realized = Device[ret.device].buffer(prod(ret.shape), ret.dtype)

    ASTRunner(
        "wkv_forward",
        WKV_FORWARD_PRG,
        global_size=[ret.shape[0] * ret.shape[2]],
        local_size=[min(ret.shape[2], 32)],
    ).build(Device[ret.device].runtime).exec(
        [ret, shape, time_first, time_decay, key, value]
    )
    return ret.realized


# opencl WKV backward runner
def opencl_wkv_backward(
    ret: LazyBuffer,
    shape: LazyBuffer,
    time_first: LazyBuffer,
    time_decay: LazyBuffer,
    key: LazyBuffer,
    value: LazyBuffer,
    wkv: LazyBuffer,
    grad: LazyBuffer,
):
    ret.realized = Device[ret.device].buffer(prod(ret.shape), ret.dtype)

    ASTRunner(
        "wkv_backward",
        WKV_BACKWARD_PRG,
        global_size=[ret.shape[0] * ret.shape[2]],
        local_size=[min(ret.shape[2], 32)],
    ).build(Device[ret.device].runtime).exec(
        [ret, shape, time_first, time_decay, key, value, wkv, grad]
    )
    return ret.realized


# opencl WKV function
class OpenCLWKVFunction(Function):
    def forward(
        self,
        shape: LazyBuffer,
        time_first: LazyBuffer,
        time_decay: LazyBuffer,
        key: LazyBuffer,
        value: LazyBuffer,
        _shape: Tuple[int, int, int],
    ) -> LazyBuffer:
        # save for backward
        self.shape = shape
        self.time_first = time_first
        self.time_decay = time_decay
        self.key = key
        self.value = value
        self._shape = _shape

        ast = LazyOp(
            LoadOps.CUSTOM,
            (
                shape.contiguous(),
                time_first.contiguous(),
                time_decay.contiguous(),
                key.contiguous(),
                value.contiguous(),
            ),
            opencl_wkv_forward,
        )
        self.wkv = create_lazybuffer(
            time_first.device, _shape, LoadOps, ast, time_first.dtype
        )
        return self.wkv

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[
        None,
        Optional[LazyBuffer],
        Optional[LazyBuffer],
        Optional[LazyBuffer],
        Optional[LazyBuffer],
    ]:
        ast = LazyOp(
            LoadOps.CUSTOM,
            (
                self.shape.contiguous(),
                self.time_first.contiguous(),
                self.time_decay.contiguous(),
                self.key.contiguous(),
                self.value.contiguous(),
                self.wkv.contiguous(),
                grad_output.contiguous(),
            ),
            opencl_wkv_backward,
        )
        ret = create_lazybuffer(
            grad_output.device,
            (self._shape[0] * 2 + 2, self._shape[1], self._shape[2]),
            LoadOps,
            ast,
            grad_output.dtype,
        )
        g_time_first = ret.movement_op(
            MovementOps.SHRINK,
            (
                (self._shape[0] * 0, self._shape[0] * 1),
                (0, self._shape[1]),
                (0, self._shape[2]),
            ),
        )
        g_time_decay = ret.movement_op(
            MovementOps.SHRINK,
            (
                (self._shape[0] * 1, self._shape[0] * 2),
                (0, self._shape[1]),
                (0, self._shape[2]),
            ),
        )
        g_key = ret.movement_op(
            MovementOps.SHRINK,
            (
                (self._shape[0] * 2, self._shape[0] * 2 + 1),
                (0, 1),
                (0, self._shape[2]),
            ),
        ).movement_op(MovementOps.RESHAPE, (self._shape[2],))
        g_value = ret.movement_op(
            MovementOps.SHRINK,
            (
                (self._shape[0] * 2 + 1, self._shape[0] * 2 + 2),
                (0, 1),
                (0, self._shape[2]),
            ),
        ).movement_op(MovementOps.RESHAPE, (self._shape[2],))
        print(g_time_first.shape)
        print(g_time_decay.shape)
        print(g_key.shape)
        print(g_value.shape)
        return None, g_value, g_key, g_time_decay, g_time_first


# opencl WKV kernel
class OpenCLWKV(WKV):
    def __call__(
        self,
        B: int,
        T: int,
        C: int,
        time_first: Tensor,
        time_decay: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        return OpenCLWKVFunction.apply(
            Tensor([B, T, C], requires_grad=False, dtype=dtypes.int32),
            time_first,
            -(time_decay.exp()),
            key,
            value,
            _shape=(B, T, C),
        )
