"""
Test generated embeddings for BEND tasks using different embedder models.
"""

import hydra
import numpy as np
import pytest
from conftest import (
    BATCH_CFG,
    DEFAULT_CFG,
    SUPERVISED_DATASETS,
    VARIANT_EFFECTS_DATASETS,
)
from scipy.stats import pearsonr
from test_datasets import assert_splits_match


EMBEDDERS = (
    "default_embedder, batch_embedder",
    [
        pytest.param(
            embedder_name,
            embedder_name,
            id=f"{embedder_name}",
        )
        for embedder_name in [
            "hyenadna-tiny-1k",
            "hyenadna-large-1m",
            "nt_transformer_ms",
            "nt_transformer_1000g",
            "nt_transformer_human_ref",
            "nt_transformer_v2_500m",
            "dnabert2",
            "resnetlm",
        ]
    ],
)

# Number of embeddings to retrieve for testing
N_EMBEDDINGS = 1
# Minimum Pearson correlation between embeddings
MIN_CORR = 1 - 1e-5
# Maximum allowed difference between any two embedding values
# Results can be batch dependent!
# (ie for HyenaDNA, due to normalisation based on batch)
ABS_TOL = 0


@pytest.fixture(scope="module")
def default_embedder(request):
    """
    Fixture to provide the embedder model and autoregressive flag.
    """
    embedder = request.param

    autoregressive = False
    if "awdlstm" in embedder or "hyenadna" in embedder:
        autoregressive = True

    return hydra.utils.instantiate(DEFAULT_CFG[embedder]), autoregressive


@pytest.fixture(scope="module")
def batch_embedder(request):
    """
    Fixture to provide the embedder model and autoregressive flag.
    """
    embedder = request.param

    task_cfg = BATCH_CFG[
        list(BATCH_CFG.keys())[0]
    ]  # Any task will do, they all have the same embedding config

    embedder_model = hydra.utils.instantiate(task_cfg["embedding"][embedder])

    return embedder_model, embedder_model.autoregressive


def assert_embedding(gt_emb, batch_emb):
    """Asserts that ground truth embeddings and batch embeddings are similar"""

    assert gt_emb.shape == batch_emb.shape, (
        f"GT embeddings and batch embeddings shape mismatch: "
        f"{gt_emb.shape} != {batch_emb.shape}"
    )

    batch_emb = batch_emb.flatten()
    gt_emb = gt_emb.flatten()

    max_diff = np.max(np.abs(gt_emb - batch_emb))
    assert np.allclose(
        gt_emb, batch_emb, atol=ABS_TOL
    ), f"Max difference too high: {max_diff}"

    pearson_corr = pearsonr(gt_emb, batch_emb)[0]
    assert pearson_corr > MIN_CORR, f"Pearson correlation too low: {pearson_corr}"


@pytest.mark.parametrize(
    *SUPERVISED_DATASETS,
    indirect=True,
)
@pytest.mark.parametrize(
    *EMBEDDERS,
    indirect=True,
)
def test_supervised_embeddings(
    supervised_dataset, default_embedder, batch_embedder
):  # pylint: disable=redefined-outer-name
    """
    Test that the embeddings generated using our approach match BEND's approach
    for the specified embedder.
    """

    task, gt_data, batch_data = supervised_dataset
    gt_embedder, _ = default_embedder
    batch_embedder, _ = batch_embedder

    splits = assert_splits_match(gt_data, batch_data)

    for split in splits:
        for idx, (gt_sample, bat_sample) in enumerate(
            zip(gt_data.next_sample(split=split), batch_data.next_sample(split=split))
        ):
            gt_seq, _ = gt_sample
            bat_seq, _ = bat_sample

            bat_emb = batch_embedder(
                [bat_seq],  # add batch dimension
                BATCH_CFG[task].task.dataset.sequence_length,
            )

            gt_emb = gt_embedder(
                gt_seq, upsample_embeddings=True
            )  # batch dimension added in __call___

            if isinstance(bat_emb, list):
                bat_emb = bat_emb[0]  # only one sequence in batch
                gt_emb = gt_emb[0]  # remove batch dimension

            assert_embedding(gt_emb, bat_emb)

            if idx + 1 >= N_EMBEDDINGS:
                gt_data.next_sample().close()
                batch_data.next_sample().close()
                break


@pytest.mark.parametrize(
    *VARIANT_EFFECTS_DATASETS,
    indirect=True,
)
@pytest.mark.parametrize(
    *EMBEDDERS,
    indirect=True,
)
def test_unsupervised_embeddings(
    var_eff_dataset, default_embedder, batch_embedder
):  # pylint: disable=redefined-outer-name
    """
    Test that the embeddings generated using our approach match BEND's approach
    for the specified embedder.
    """

    task, is_data_autoregressive, def_data, batch_data = var_eff_dataset
    def_embedder, is_def_emb_autoregressive = default_embedder
    batch_embedder, is_batch_emb_autoregressive = batch_embedder

    assert is_def_emb_autoregressive == is_batch_emb_autoregressive, (
        f"Embedder autoregressive setting mismatch: "
        f"{is_def_emb_autoregressive} != {is_batch_emb_autoregressive}"
    )

    if is_data_autoregressive != is_def_emb_autoregressive:
        pytest.skip(
            f"Autoregressive mismatch! Data {is_data_autoregressive} - Embedder {is_def_emb_autoregressive}"
        )

    for idx, (def_sample, bat_sample) in enumerate(
        zip(def_data.next_sample(), batch_data.next_sample())
    ):
        for def_seq, bat_seq in zip(
            def_sample[:2], bat_sample[:2]
        ):  # only sequences, ignore labels

            def_emb = def_embedder(def_seq, upsample_embeddings=True)
            bat_emb = batch_embedder(
                [bat_seq],  # add batch dimension
                BATCH_CFG[task].task.dataset.sequence_length,
            )

            assert_embedding(def_emb, bat_emb)

        if idx + 1 >= N_EMBEDDINGS:
            def_data.next_sample().close()
            batch_data.next_sample().close()
            break
