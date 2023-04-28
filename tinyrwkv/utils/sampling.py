from typing import Union

import numpy as np


def sample_logits(
    logits: np.ndarray,
    *,
    alpha_counter: Union[np.ndarray, None] = None,
    alpha_presence: float = 0.0,
    alpha_frequency: float = 0.0,
    temperature: float = 0.8,
    typical_tau: float = 0.0,
    top_k: int = 50,
) -> int:
    # try to prevent NaNs for at least a little bit
    logits = np.nan_to_num(logits)

    # greedy sampling
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

    # softmax
    logits_exp = np.exp(logits)
    probs = logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)

    # typical sampling
    if typical_tau > 0.0:
        logits = -np.log(probs)
        entropy = np.nansum(logits * probs, axis=-1, keepdims=True)
        logits = np.abs(logits - entropy)
        sorted_idxs = np.argsort(logits)
        sorted_logits = logits[sorted_idxs]
        sorted_probs = probs[sorted_idxs]
        cumulative_probs = np.cumsum(sorted_probs, axis=-1)
        cutoff = np.sum(cumulative_probs < typical_tau)
        probs[logits > sorted_logits[cutoff]] = 0

    # default sampling
    if temperature != 1.0:
        probs = pow(probs, 1.0 / temperature)
    probs = probs / np.sum(probs, axis=0)
    out = np.random.choice(a=len(probs), p=probs)
    return out
