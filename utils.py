from tinygrad.nn.optim import get_parameters
from tinygrad.tensor import Tensor
import numpy as np


def sample_logits(
    logits: np.ndarray,
    *,
    alpha_counter: np.ndarray | None = None,
    alpha_presence: float = 0.0,
    alpha_frequency: float = 0.0,
    temperature: float = 0.8,
    top_p: float = 0.9,
    typical_p: float = 0.0,
    top_k: int = 50,
) -> int:
    logits = np.nan_to_num(logits)

    if temperature == 0.0:
        return int(np.argmax(logits))

    # alpha sampling
    if alpha_counter is not None and alpha_presence > 0.0 and alpha_frequency > 0.0:
        for i in range(logits.shape[0]):
            logits[i] -= (alpha_counter[i] * alpha_frequency) + (
                float(alpha_counter[i] > 0) * alpha_presence
            )

    # top-k sampling
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = float("-Inf")

    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

    # top-p sampling
    if top_p > 0.0:
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0
        if temperature != 1.0:
            probs = pow(probs, 1.0 / temperature)
        probs = probs / np.sum(probs, axis=0)
        out = np.random.choice(a=len(probs), p=probs)
        return out

    # typical sampling
    if typical_p > 0.0:
        logits = -np.log(probs)
        entropy = np.nansum(logits * probs, axis=-1, keepdims=True)
        logits = np.abs(logits - entropy)
        sorted_idxs = np.argsort(logits)
        sorted_logits = logits[sorted_idxs]
        sorted_probs = probs[sorted_idxs]
        cumulative_probs = np.cumsum(sorted_probs, axis=-1)
        cutoff = np.sum(cumulative_probs < typical_p)
        probs[logits > sorted_logits[cutoff]] = 0
        if temperature != 1.0:
            probs = pow(probs, 1.0 / temperature)
        probs = probs / np.sum(probs, axis=0)
        out = np.random.choice(a=len(probs), p=probs)
        return out

    # default sampling
    if temperature != 1.0:
        probs = pow(probs, 1.0 / temperature)
    probs = probs / np.sum(probs, axis=0)
    out = np.random.choice(a=len(probs), p=probs)
    return out


def matvec(mat: Tensor, vec: Tensor) -> Tensor:
    return vec @ mat.T


def elemmax(x: Tensor, y: Tensor) -> Tensor:
    xgty = x.sub(y).relu() - (x.sub(y) - 1).relu()
    ygtx = y.sub(x).relu() - (y.sub(x) - 1).relu()
    return xgty * x + ygtx * y


def compile_net(run, special_names):
    # functions that run the net
    functions = {}
    bufs = {}
    bufnum = 0
    statements = []
    bufs_to_save = {}
    for fxn, args in run.jit_cache:
        functions[
            fxn.name
        ] = fxn.prg  # NOTE: this assumes all with the same name are the same
        cargs = []
        for i, arg in enumerate(args):
            key = id(arg)
            if key not in bufs:
                if key in special_names:
                    bufs[key] = (special_names[key], len(arg._buf), arg.dtype)
                else:
                    bufs[key] = (f"buf_{bufnum}", len(arg._buf), arg.dtype)
                    bufnum += 1
                    if i > 0:
                        bufs_to_save[
                            bufs[key][0]
                        ] = arg  # if first usage of a buffer is not an output, and it's not a special name
            cargs.append(bufs[key][0])
        statements.append(f"{fxn.name}({', '.join(cargs)});")

    return functions, statements, bufs, bufs_to_save


def get_child(parent, key):
    obj = parent
    for k in key.split("."):
        if k.isnumeric():
            obj = obj[int(k)]
        elif isinstance(obj, dict):
            obj = obj[k]
        else:
            obj = getattr(obj, k)
    return obj


def count_parameters(model):
    params = get_parameters(model)
    count = 0
    for p in params:
        param_count = 1
        for s in p.shape:
            param_count *= s
        count += param_count
    return count
