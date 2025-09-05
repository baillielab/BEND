"""
Test generated embeddings for BEND tasks using different embedder models.
"""

import hydra
import numpy as np
import pytest
from conftest import BatchData, DefaultData
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

EMBEDDERS = [
    "hyenadna-tiny-1k",
    "hyenadna-large-1m",
    "nt_transformer_ms",
    "nt_transformer_1000g",
    "nt_transformer_human_ref",
    "nt_transformer_v2_500m",
    "dnabert2",
    "awdlstm",
    "resnetlm",
]

# Number of embeddings to retrieve for testing
N_EMBEDDINGS = 1
# Minimum Pearson correlation between embeddings
MIN_CORR = 1 - 1e-5
# Maximum allowed difference between any two embedding values
# Results can be batch dependent!
# (ie for HyenaDNA, due to normalisation based on batch)
ABS_TOL = 0
PADDING_VALUE = -100


def get_gt_embeddings(gt_data: DefaultData, split: str, embedder: str):
    """
    Generate embeddings for the given sequences using BEND method and the specified embedder.
    """

    cfg = gt_data.cfg
    gt_split = gt_data.get_split_samples(split)
    sequences_subset = [seq for seq, _ in gt_split[:N_EMBEDDINGS]]

    embedder = hydra.utils.instantiate(cfg[embedder])

    gt_embeddings = []
    sequences = []

    for _, seq in tqdm(enumerate(sequences_subset), desc="Embedding GT sequences"):
        sequences.append(seq)
        seq_embed = embedder(seq, upsample_embeddings=True)
        gt_embeddings.extend(seq_embed)

    return gt_embeddings, sequences


def get_batch_embeddings(batch_data: BatchData, split: str, embedder: str):
    """
    Generate embeddings for a batch of sequences using our approach and the specified embedder.
    """

    cfg = batch_data.cfg
    dataset = batch_data.get_split_dataset(split)

    embedder = hydra.utils.instantiate(cfg.embedding[embedder])

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    is_data_uneven = True if cfg.task.dataset.sequence_length is None else False

    embeddings = []
    sequences = []

    for _, (seq, _) in tqdm(enumerate(dataloader), desc="Embedding batches"):

        seq_embed = embedder(seq, is_data_uneven)

        embeddings.extend(seq_embed)  # list of one sequence as batch is of size 1
        sequences.extend(seq)  # list of one sequence as batch is of size 1

        if len(embeddings) >= N_EMBEDDINGS:
            break

    return embeddings, sequences


def assert_sequences(gt_sequences, batch_sequences):
    """Asserts that ground truth sequences and batch sequences are equal."""

    assert len(gt_sequences) == len(batch_sequences), (
        f"GT sequences and batch sequences length mismatch: "
        f"{len(batch_sequences)} != {len(gt_sequences)}"
    )
    for b_seq, gt_seq in zip(batch_sequences, gt_sequences):
        assert b_seq == gt_seq, "Sequence mismatch!"


def assert_embeddings(gt_embeddings, batch_embeddings):
    """Asserts that ground truth embeddings and batch embeddings are similar
    using Pearson correlation and absolute tolerance.
    """

    assert len(gt_embeddings) == len(batch_embeddings) and len(gt_embeddings) > 0, (
        f"GT embeddings and batch embeddings length mismatch: "
        f"{len(batch_embeddings)} != {len(gt_embeddings)}"
    )

    pearson_corr = []
    max_diff = 0

    for gt_emb, batch_emb in zip(gt_embeddings, batch_embeddings):

        assert gt_emb.shape == batch_emb.shape, (
            f"GT embeddings and batch embeddings shape mismatch: "
            f"{gt_emb.shape} != {batch_emb.shape}"
        )

        batch_emb = batch_emb.flatten()
        gt_emb = gt_emb.flatten()

        pearson_corr.append(pearsonr(gt_emb, batch_emb)[0])

        max_diff = max(max_diff, np.max(np.abs(gt_emb - batch_emb)))
        assert np.allclose(
            gt_emb, batch_emb, atol=ABS_TOL
        ), f"Max difference too high: {max_diff}"

    pearson_corr = np.mean(np.array(pearson_corr))
    assert pearson_corr > MIN_CORR, f"Pearson correlation too low: {pearson_corr}"


@pytest.mark.parametrize(
    "embedder",
    EMBEDDERS,
)
def test_supervised_embeddings(supervised_data, embedder):
    """
    Test that the embeddings generated using our approach match BEND's approach
    for the specified embedder.
    """

    task, gt_data, batch_data = supervised_data

    print(f"\nTesting embeddings for task: {task}, embedder: {embedder}\n")

    for split in gt_data.get_split_names():
        print(f"Testing split: {split}")

        batch_embeddings, batch_sequences = get_batch_embeddings(
            batch_data, split, embedder
        )
        gt_embeddings, gt_sequences = get_gt_embeddings(gt_data, split, embedder)

    assert_sequences(gt_sequences, batch_sequences)
    assert_embeddings(gt_embeddings, batch_embeddings)
