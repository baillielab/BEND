"""
Test that the following datasets produce the same sequences and labels as the ground truth:
- DataSupervised
"""

from itertools import zip_longest

import numpy as np
import pytest
from conftest import SUPERVISED_DATASETS, VARIANT_EFFECTS_DATASETS


def assert_splits_match(gt_data, batch_data) -> tuple[list[str], list[str]]:
    """
    Assert that the splits in the ground truth data and batch data match.
    Returns the splits for further processing.
    """
    gt_splits = gt_data.get_split_names()
    batch_splits = batch_data.get_split_names()

    assert set(batch_splits) == set(
        gt_splits
    ), f"Splits do not match: {sorted(batch_splits)} vs {sorted(gt_splits)}"

    return gt_splits


@pytest.mark.parametrize(
    *SUPERVISED_DATASETS,
    indirect=True,
)
def test_supervised_samples(supervised_dataset):
    """
    Test that the sequences and labels from the DataSupervised dataset match the ground truth data.
    """

    _, gt_data, batch_data = supervised_dataset

    splits = assert_splits_match(gt_data, batch_data)

    for split in splits:

        for idx, (gt_sample, batch_sample) in enumerate(
            zip_longest(
                gt_data.next_sample(split=split),
                batch_data.next_sample(split=split),
                fillvalue=None,
            )
        ):
            assert (
                gt_sample is not None
            ), f"Ground truth data has fewer samples in split {split} at index {idx}"
            assert (
                batch_sample is not None
            ), f"Batch data has fewer samples in split {split} at index {idx}"

            gt_seq, gt_lbl = gt_sample
            batch_seq, batch_lbl = batch_sample

            assert (
                gt_seq == batch_seq
            ), f"Sequences do not match for split {split} at index {idx}"
            assert np.array_equal(
                gt_lbl, batch_lbl
            ), f"Labels do not match for split {split} at index {idx}"


@pytest.mark.parametrize(
    *VARIANT_EFFECTS_DATASETS,
    indirect=True,
)
def test_unsupervised_samples(var_eff_dataset):
    """
    Test that the sequences and labels from the DataVariantEffects dataset match the ground truth data.
    """

    _, _, gt_data, batch_data = var_eff_dataset

    for idx, (gt_sample, bat_sample) in enumerate(
        zip_longest(gt_data.next_sample(), batch_data.next_sample())
    ):
        gt_ref_seq, gt_snp_seq, gt_lbl = gt_sample
        bat_ref_seq, bat_snp_seq, bat_lbl = bat_sample

        assert (
            gt_ref_seq == bat_ref_seq
        ), f"Reference sequences do not match at index {idx}"
        assert gt_snp_seq == bat_snp_seq, f"SNP sequences do not match at index {idx}"
        assert np.array_equal(gt_lbl, bat_lbl), f"Labels do not match at index {idx}"
