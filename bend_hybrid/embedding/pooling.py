"""
Pooling functions and mode enum for embeddings.
"""

import enum

import numpy as np


class PoolingMode(enum.Enum):
    """Enumeration of supported pooling output modes.

    Values are used as directory names and identifiers downstream.
    """

    DEFAULT = "default"
    MEAN = "mean"
    MEAN_NO_UPSAMPLE = "mean_no_upsample"
    MAX = "max"
    CLS = "cls"
    EOS = "eos"


def pool_embeddings(embeddings: np.ndarray, mode: PoolingMode) -> np.ndarray:
    """
    Pool embeddings according to the specified mode.
    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings to pool.
    mode : PoolingMode
        Pooling mode to use.
    Returns
    -------
    np.ndarray
        Pooled embeddings.
    """

    match mode:
        case PoolingMode.DEFAULT:
            emb = embeddings
        case PoolingMode.CLS:
            emb = embeddings[:, 0:1, :]
        case PoolingMode.EOS:
            emb = embeddings[:, -1:, :]
        case PoolingMode.MEAN | PoolingMode.MEAN_NO_UPSAMPLE:
            emb = np.mean(embeddings, axis=1, keepdims=True)
        case PoolingMode.MAX:
            emb = np.max(embeddings, axis=1, keepdims=True)
        case _:
            raise ValueError(f"Unknown pooling mode: {mode}")

    return emb, mode.value
