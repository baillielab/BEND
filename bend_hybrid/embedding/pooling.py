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
    MEAN_UPSAMPLE = "mean_upsample"
    MAX = "max"
    CLS = "cls"
    EOS = "eos"


def pool_mean(embeddings: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """
    Pool embeddings by taking the mean across the sequence length dimension.
    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings to pool.
    Returns
    -------
    np.ndarray
        Pooled embeddings.
    """
    if isinstance(embeddings, list):
        pooled = []
        for emb in embeddings:
            pooled.append(np.nanmean(emb, axis=0, keepdims=True))
        return np.stack(pooled)

    return np.nanmean(embeddings, axis=1, keepdims=True)


def pool_max(embeddings: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """
    Pool embeddings by taking the max across the sequence length dimension.
    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings to pool.
    Returns
    -------
    np.ndarray
        Pooled embeddings.
    """
    if isinstance(embeddings, list):
        pooled = []
        for emb in embeddings:
            pooled.append(np.nanmax(emb, axis=0, keepdims=True))
        return np.stack(pooled)

    return np.nanmax(embeddings, axis=1, keepdims=True)


def pool_cls(embeddings: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """
    Pool embeddings by taking the CLS token embedding.
    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings to pool.
    Returns
    -------
    np.ndarray
        Pooled embeddings.
    """

    if isinstance(embeddings, list):
        cls = []
        for emb in embeddings:
            cls.append(emb[0:1, :])
        return np.stack(cls)

    return embeddings[:, 0:1, :]


def pool_eos(embeddings: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """
    Pool embeddings by taking the EOS token embedding.
    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings to pool.
    Returns
    -------
    np.ndarray
        Pooled embeddings.
    """

    if isinstance(embeddings, list):
        eos = []
        for emb in embeddings:
            eos.append(emb[-1:, :])
        return np.stack(eos)

    return embeddings[:, -1:, :]


pool_name_to_function = {
    PoolingMode.DEFAULT: lambda x: x,
    PoolingMode.MEAN: pool_mean,
    PoolingMode.MEAN_UPSAMPLE: pool_mean,
    PoolingMode.MAX: pool_max,
    PoolingMode.CLS: pool_cls,
    PoolingMode.EOS: pool_eos,
}
