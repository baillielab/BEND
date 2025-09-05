"""
Test that the following datasets produce the same sequences and labels as the ground truth:
- DataSupervised
"""

import numpy as np


def test_supervised_sequences_and_labels(supervised_data):
    """
    Test that the sequences and labels from the DataSupervised dataset match the ground truth data.
    """

    task, gt_data, batch_data = supervised_data

    print(f"Testing sequences and labels for task: {task}")

    batch_splits = batch_data.get_split_names()
    gt_splits = gt_data.get_split_names()

    assert set(batch_splits) == set(
        gt_splits
    ), f"Splits do not match: {sorted(batch_splits)} vs {sorted(gt_splits)}"

    for split in gt_splits:
        print(f"Testing split: {split}")

        gt_samples = gt_data.get_samples(split)
        batch_samples = batch_data.get_samples(split)

        assert len(gt_samples) == len(
            batch_samples
        ), f"Number of samples do not match for split {split}: {len(gt_samples)} vs {len(batch_samples)}"

        for idx, (gt_sample, batch_sample) in enumerate(zip(gt_samples, batch_samples)):

            gt_seq, gt_lbl = gt_sample
            batch_seq, batch_lbl = batch_sample

            assert (
                gt_seq == batch_seq
            ), f"Sequences do not match for split {split} at index {idx}"
            assert np.array_equal(
                gt_lbl, batch_lbl
            ), f"Labels do not match for split {split} at index {idx}"
