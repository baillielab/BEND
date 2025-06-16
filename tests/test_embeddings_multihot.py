from tqdm.auto import tqdm
import pysam
import pandas as pd
import numpy as np
import pytest
from bend.utils.datasets import DatasetMultiHot
import os
from bend.utils.embedders import HyenaDNAEmbedder, ConvNetEmbedder, AWDLSTMEmbedder
from hydra import compose, initialize
from omegaconf import OmegaConf
import hydra
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

EMBEDDERS = ["hyenadna-tiny-1k"]

N_EMBEDDINGS = 1000  # Number of embeddings to retrieve for testing
MIN_CORR = 1 - 1e-5  # Minimum Pearson correlation between embeddings
ABS_TOL = 1e-4  # Maximum allowed difference between any two embedding values -> Results are batch dependent! (at least for HyenaDNA, due to normalisation based on batch)

with initialize(version_base=None, config_path="../conf/embedding/"):
    CFG = compose(config_name="embed")


def get_gt_embeddings(gt_sequences, embedder):
    embedder = hydra.utils.instantiate(CFG[embedder], mode="sequential")
    sequences_subset = gt_sequences[:N_EMBEDDINGS]

    gt_embeddings = []
    for idx_sample, seq in tqdm(
        enumerate(sequences_subset), desc="Embedding GT sequences"
    ):
        seq_embed = embedder(seq)
        gt_embeddings.extend(seq_embed)

    return gt_embeddings


def get_batch_embeddings(dataset, embedder, task):

    embedder = hydra.utils.instantiate(CFG[embedder], mode="batch")

    with initialize(version_base=None, config_path="../conf/supervised_tasks/"):
        cfg_task = compose(config_name=task)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg_task["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg_task["data"]["num_workers"],
    )

    embeddings = []
    for idx_batch, (seq, _) in tqdm(enumerate(dataloader), desc="Embedding batches"):
        batch_embedded = embedder.embed(seq)

        embeddings.extend(batch_embedded)  # list of seq_len x embed_dim numpy arrays

        idx_sample = (idx_batch + 1) * len(batch_embedded)
        if idx_sample >= N_EMBEDDINGS:
            break

    embeddings = embeddings[:N_EMBEDDINGS]

    return embeddings


@pytest.mark.parametrize(
    "embedder",
    EMBEDDERS,
)
def test_embeddings(data, embedder):

    task, split, gt_data, dataset = data

    print(
        f"\nTesting embeddings for task: {task}, split: {split}, embedder: {embedder}\n"
    )

    gt_sequences, _ = gt_data

    batch_embeddings = get_batch_embeddings(dataset, embedder, task)
    batch_embeddings = np.array(batch_embeddings).astype(np.float64)
    print(f"Batch Embeddings shape: {batch_embeddings.shape}")

    gt_embeddings = get_gt_embeddings(gt_sequences, embedder)
    gt_embeddings = np.array(gt_embeddings).astype(np.float64)
    print(f"GT Embeddings shape: {gt_embeddings.shape}")

    assert (
        gt_embeddings.shape == batch_embeddings.shape
    ), f"Shape mismatch: {gt_embeddings.shape} != {batch_embeddings.shape}"

    batch_embeddings = batch_embeddings.flatten()
    gt_embeddings = gt_embeddings.flatten()

    pearson_corr = pearsonr(batch_embeddings, gt_embeddings)[0]

    print(f"Pearson correlation: {pearson_corr}")

    assert pearson_corr > MIN_CORR, f"Pearson correlation too low: {pearson_corr}"
    assert np.allclose(
        gt_embeddings, batch_embeddings, atol=ABS_TOL
    ), f"Max difference too high: {np.max(np.abs(gt_embeddings - batch_embeddings))}"
