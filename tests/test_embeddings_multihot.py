from tqdm.auto import tqdm
import pysam
import pandas as pd
import numpy as np
from bend.io.sequtils import get_embeddings_from_bed
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
ABS_TOL = 1e-4  # Maximum allowed difference between any two embedding values

with initialize(version_base=None, config_path="../conf/embedding/"):
    CFG = compose(config_name="embed")


def get_gt_embeddings(gt_sequences, embedder):
    embedder = hydra.utils.instantiate(CFG[embedder], mode="sequential")

    gt_embeddings = []
    for idx_sample, seq in tqdm(enumerate(gt_sequences), desc="Embedding GT sequences"):
        seq_embed = embedder(seq)
        gt_embeddings.extend(seq_embed)

        if (idx_sample + 1) == N_EMBEDDINGS:
            break

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


def get_embedding_metrics(batch_embedding, seq_embedding):
    batch_emb = batch_embedding.flatten()
    seq_emb = seq_embedding.flatten()

    return pearsonr(batch_emb, seq_emb)[0], np.max(np.abs(batch_emb - seq_emb))


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

    gt_embeddings = get_gt_embeddings(gt_sequences, embedder)
    batch_embeddings = get_batch_embeddings(dataset, embedder, task)

    gt_emb = np.array(gt_embeddings)
    batch_emb = np.array(batch_embeddings)

    print(f"GT Embeddings shape: {gt_emb.shape}")
    print(f"Batch Embeddings shape: {batch_emb.shape}")

    assert (
        gt_emb.shape == batch_emb.shape
    ), f"Shape mismatch: {gt_emb.shape} != {batch_emb.shape}"

    pearson_corr, max_diff = get_embedding_metrics(batch_emb, gt_emb)
    assert pearson_corr > MIN_CORR, f"Pearson correlation too low: {pearson_corr}"
    assert np.allclose(
        gt_emb, batch_emb, atol=ABS_TOL
    ), f"Max difference too high: {max_diff}"
