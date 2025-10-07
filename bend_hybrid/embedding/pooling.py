"""
Pooling functions for embeddings.
"""

import enum

import numpy as np


class PoolingMode(enum.Enum):
    """
    Enum for pooling modes.
    """

    NONE = "none"
    MEAN = "mean"
    MEAN_NO_UPSAMPLE = "mean_no_upsample"
    MAX = "max"
    MIN_MAX = "min-max"


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
        case PoolingMode.NONE:
            return embeddings
        case PoolingMode.MEAN:
            return np.mean(embeddings, axis=0, keepdims=True)
        case PoolingMode.MEAN_NO_UPSAMPLE:
            unique_emb = np.unique(embeddings, axis=0)
            return np.mean(unique_emb, axis=0, keepdims=True)
        case PoolingMode.MAX:
            return np.max(embeddings, axis=0, keepdims=True)
        case PoolingMode.MIN_MAX:
            idx_max_abs = np.argmax(np.abs(embeddings), axis=0, keepdims=True)
            return embeddings[idx_max_abs, np.arange(embeddings.shape[1])]
        case _:
            raise ValueError(f"Unknown pooling mode: {mode}")
