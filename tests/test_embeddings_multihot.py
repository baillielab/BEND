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


N_EMBEDDINGS = 1000  # Number of embeddings to retrieve for testing
MIN_CORR = 1 - 1e-5  # Minimum Pearson correlation between embeddings
MAX_DIFF = 1e-4  # Maximum allowed difference between any two embedding values

with initialize(version_base=None, config_path="../conf/embedding/"):
    cfg = compose(config_name="embed")


class DatasetMultiHotTest(DatasetMultiHot):

    def _get_data(
        self,
        annotations,
        genome,
        label_column_idx,
        strand_column_idx,
        flank,
        label_depth,
    ):
        annotations = annotations.head(N_EMBEDDINGS)
        return super()._get_data(
            annotations, genome, label_column_idx, strand_column_idx, flank, label_depth
        )


def get_gt_embeddings(task, embedder, split, n_embeddings):

    gt_embeddings = get_embeddings_from_bed(
        cfg[task]["bed"],
        cfg[task]["reference_fasta"],
        embedder=hydra.utils.instantiate(cfg[embedder], mode="sequential"),
        label_depth=cfg[task]["label_depth"],
        read_strand=cfg[task]["read_strand"],
        split=split,
        chunk_size=n_embeddings,
        chunk=0,
    )

    return gt_embeddings


def get_batch_embeddings(task, embedder, split, n_embeddings):

    embedder = hydra.utils.instantiate(cfg[embedder], mode="batch")
    dataset = DatasetMultiHotTest(
        cfg[task]["bed"],
        cfg[task]["reference_fasta"],
        cfg[task]["label_depth"],
        split=split,
    )

    with initialize(version_base=None, config_path="../conf/supervised_tasks/"):
        cfg_task = compose(config_name=task)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg_task["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg_task["data"]["num_workers"],
    )

    embeddings = []
    for idx_batch, (seq, label) in tqdm(enumerate(dataloader)):
        batch_embedded = embedder.embed(seq)

        embeddings.extend(batch_embedded)  # list of seq_len x embed_dim numpy arrays

        idx_sample = (idx_batch + 1) * len(batch_embedded)
        if idx_sample >= n_embeddings:
            break

    embeddings = embeddings[:n_embeddings]

    return embeddings


@pytest.fixture
def embeddings(request):
    task, embedder, split, n_embeddings = request.param

    gt_embeddings = get_gt_embeddings(task, embedder, split, n_embeddings)
    batch_embeddings = get_batch_embeddings(task, embedder, split, n_embeddings)

    return gt_embeddings, batch_embeddings


def get_embedding_metrics(batch_embedding, seq_embedding):
    batch_emb = batch_embedding.flatten()
    seq_emb = seq_embedding.flatten()

    return pearsonr(batch_emb, seq_emb)[0], np.max(np.abs(batch_emb - seq_emb))


@pytest.mark.parametrize(
    "embeddings",
    [
        ("cpg_methylation", "hyenadna-tiny-1k", "train", N_EMBEDDINGS),
        ("cpg_methylation", "hyenadna-tiny-1k", "valid", N_EMBEDDINGS),
        ("cpg_methylation", "hyenadna-tiny-1k", "test", N_EMBEDDINGS),
    ],
    indirect=True,
)
def test_sequence_labels(embeddings):
    gt_embeddings, batch_embeddings = embeddings

    gt_emb = np.array(gt_embeddings)
    batch_emb = np.array(batch_embeddings)

    print(f"GT Embeddings shape: {gt_emb.shape}")
    print(f"Batch Embeddings shape: {batch_emb.shape}")

    assert (
        gt_emb.shape == batch_emb.shape
    ), f"Shape mismatch: {gt_emb.shape} != {batch_emb.shape}"

    pearson_corr, max_diff = get_embedding_metrics(batch_emb, gt_emb)
    assert pearson_corr > MIN_CORR, f"Pearson correlation too low: {pearson_corr}"
    assert max_diff < MAX_DIFF, f"Max difference too high: {max_diff}"
