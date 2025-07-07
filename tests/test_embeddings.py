from tqdm.auto import tqdm
import numpy as np
import pytest
from hydra import compose, initialize
import hydra
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from bend.utils.set_seed import set_seed
import torch

set_seed()

EMBEDDERS = [
    "dnabert2",
    "hyenadna-tiny-1k",
    "resnetlm",
]


N_EMBEDDINGS = 100  # Number of embeddings to retrieve for testing
MIN_CORR = 1 - 1e-5  # Minimum Pearson correlation between embeddings
ABS_TOL = 1e-4  # Maximum allowed difference between any two embedding values -> Results are batch dependent! (at least for HyenaDNA, due to normalisation based on batch)

PADDING_VALUE = -100

with initialize(version_base=None, config_path="../conf/embedding/"):
    CFG_SEQ = compose(config_name="embed")
with initialize(version_base=None, config_path="../config_memoryless/embedders_batch/"):
    CFG_BATCH = compose(config_name="embedders")
    CFG_BATCH["embedders_dir"] = CFG_SEQ["embedders_dir"]


def get_gt_embeddings(gt_sequences, embedder):
    embedder = hydra.utils.instantiate(CFG_SEQ[embedder])
    sequences_subset = gt_sequences[:N_EMBEDDINGS]

    gt_embeddings = []
    sequences = []

    for idx_sample, seq in tqdm(
        enumerate(sequences_subset), desc="Embedding GT sequences"
    ):
        sequences.append(seq)
        seq_embed = embedder(seq, upsample_embeddings=True)
        gt_embeddings.extend(seq_embed)

    gt_embeddings = np.array(gt_embeddings).astype(np.float64)

    return gt_embeddings, sequences


def get_batch_embeddings(dataset, embedder):

    embedder = hydra.utils.instantiate(CFG_BATCH[embedder])

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    embeddings = []
    sequences = []

    for idx_batch, (seq, _) in tqdm(enumerate(dataloader), desc="Embedding batches"):
        sequences.extend(seq)
        batch_embedded = embedder(seq)

        embeddings.extend(batch_embedded)  # list of seq_len x embed_dim numpy arrays

        idx_sample = (idx_batch + 1) * len(batch_embedded)
        if idx_sample >= N_EMBEDDINGS:
            break

    embeddings = embeddings[:N_EMBEDDINGS]
    embeddings = np.array(embeddings).astype(np.float64)

    sequences = sequences[:N_EMBEDDINGS]

    return embeddings, sequences


def assert_sequences(gt_sequences, batch_sequences):
    """Asserts that ground truth sequences and batch sequences are equal."""
    assert len(batch_sequences) == len(gt_sequences), (
        f"Batch sequences and GT sequences length mismatch: "
        f"{len(batch_sequences)} != {len(gt_sequences)}"
    )
    for b_seq, gt_seq in zip(batch_sequences, gt_sequences):
        assert b_seq == gt_seq, f"Sequence mismatch: {b_seq} != {gt_seq}"


def assert_embeddings(gt_embeddings, batch_embeddings):
    """
    Asserts that ground truth embeddings and batch embeddings are similar
    using Pearson correlation and absolute tolerance.
    """

    assert batch_embeddings.shape == gt_embeddings.shape, (
        f"Batch embeddings and GT embeddings shape mismatch: "
        f"{batch_embeddings.shape} != {gt_embeddings.shape}"
    )

    batch_embeddings = batch_embeddings.flatten()
    gt_embeddings = gt_embeddings.flatten()

    pearson_corr = pearsonr(batch_embeddings, gt_embeddings)[0]

    print(f"Pearson correlation: {pearson_corr}")

    assert pearson_corr > MIN_CORR, f"Pearson correlation too low: {pearson_corr}"

    max_diff = np.max(np.abs(gt_embeddings - batch_embeddings))
    print(f"Max difference: {max_diff}")
    assert np.allclose(
        gt_embeddings, batch_embeddings, atol=ABS_TOL
    ), f"Max difference too high: {max_diff}"


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

    batch_embeddings, batch_sequences = get_batch_embeddings(dataset, embedder)
    print(f"Batch Embeddings shape: {batch_embeddings.shape}")

    gt_embeddings, gt_sequences = get_gt_embeddings(gt_sequences, embedder)
    print(f"GT Embeddings shape: {gt_embeddings.shape}")

    assert_sequences(gt_sequences, batch_sequences)

    assert_embeddings(gt_embeddings, batch_embeddings)
