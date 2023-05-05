from tinygrad.helpers import dtypes, prod
from tinygrad.lazy import LazyBuffer, create_lazybuffer, Device
from tinygrad.ops import ASTRunner, LazyOp, LoadOps
from tinygrad.tensor import Tensor, Function

import abc
import os
from typing import Optional, Tuple

from utils.tensor import elemmax


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
            p = elemmax(pp, ww)
            e1 = (pp - p).exp()
            e2 = (ww - p).exp()
            a = (e1 * aa) + (e2 * vv)
            b = (e1 * bb) + e2
            wkv.append(a / b)

            ww = pp - time_w
            p = elemmax(ww, kk)
            e1 = (ww - p).exp()
            e2 = (kk - p).exp()
            aa = ((e1 * aa) + (e2 * vv)).realize()
            bb = ((e1 * bb) + e2).realize()
            pp = p.realize()

        # calculate the last token outside to avoid having to compute the next state
        kk = key[:, -1, :]
        vv = value[:, -1, :]
        ww = kk + time_first
        p = elemmax(pp, ww)
        e1 = (pp - p).exp()
        e2 = (ww - p).exp()
        a = (e1 * aa) + (e2 * vv)
        b = (e1 * bb) + e2
        wkv.append(a / b)

        return Tensor.cat(*wkv, dim=1).realize()


# convolutional WKV kernel, same as rwkv < 4 wkv
# faster in tinygrad than the standard WKV kernel
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

        time_curve = Tensor([-(T - 2 - i) for i in range(T - 1)], requires_grad=False)

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


# standard WKV kernel but trying to parallelize more
# turns out to be slower than the standard one
class SplitWKV(WKV):
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

        # calculate a tensor thats the decayed pp at each token
        pp_decayed = [Tensor.ones(B, 1, C, requires_grad=False) * -1e38]
        for i in range(T):
            pp_decayed.append((pp_decayed[i] - time_w - key[:, i, :]).relu())
        pp_decayed_tensor = Tensor.cat(*pp_decayed, dim=1)

        k_time_first = key + time_first
        pp_decayed_max_ktf = elemmax(pp_decayed_tensor[:, :-1, :], k_time_first)
        e_ppd_pdmk = (pp_decayed_tensor[:, :-1, :] - pp_decayed_max_ktf).exp()
        e_ktf_pdmk = (k_time_first - pp_decayed_max_ktf).exp()

        pp_decayed_decayed = pp_decayed_tensor[:, :-1, :] - time_w
        e_ppdd_ppd = (pp_decayed_decayed - pp_decayed_tensor[:, 1:, :]).exp()
        e_k_ppd = (key - pp_decayed_tensor[:, 1:, :]).exp()

        aa = [e_k_ppd[:, 0:1, :] * value[:, 0, :]]
        bb = [e_k_ppd[:, 0:1, :]]

        for i in range(1, T):
            a = (e_ppdd_ppd[:, i, :] * aa[i - 1]) + (e_k_ppd[:, i, :] * value[:, i, :])
            b = (e_ppdd_ppd[:, i, :] * bb[i - 1]) + e_k_ppd[:, i, :]
            aa.append(a)
            bb.append(b)

        wkv = [
            (e_ppd_pdmk[:, 0, :] * aa[0] + e_ktf_pdmk[:, 0, :] * value[:, 0, :])
            / (e_ppd_pdmk[:, 0, :] * bb[0] + e_ktf_pdmk[:, 0, :])
        ]
        for i in range(1, T):
            e1 = e_ppd_pdmk[:, i, :]
            e2 = e_ktf_pdmk[:, i, :]
            a = (e1 * aa[i - 1]) + (e2 * value[:, i, :])
            b = (e1 * bb[i - 1]) + e2
            wkv.append(a / b)

        return Tensor.cat(*wkv, dim=1)


# load kernels once for opencl wkv
with open(os.path.join(os.path.dirname(__file__), "kernels/wkv_forward.cl"), "r") as f:
    WKV_FORWARD_PRG = f.read()


# opencl WKV runner
def opencl_wkv(
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
        ast = LazyOp(
            LoadOps.CUSTOM,
            (
                shape.contiguous(),
                time_first.contiguous(),
                time_decay.contiguous(),
                key.contiguous(),
                value.contiguous(),
            ),
            opencl_wkv,
        )
        return create_lazybuffer(
            time_first.device, _shape, LoadOps, ast, time_first.dtype
        )

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[
        Optional[LazyBuffer],
        Optional[LazyBuffer],
        Optional[LazyBuffer],
        Optional[LazyBuffer],
    ]:
        ...


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
        time_decay = -(time_decay.exp())

        return OpenCLWKVFunction.apply(
            Tensor([B, T, C], requires_grad=False, dtype=dtypes.int32),
            time_first,
            time_decay,
            key,
            value,
            _shape=(B, T, C),
        )
