"""
Test that the following datasets produce the same sequences and labels as the ground truth:
- DataSupervised
- DataVariantEffects
"""

import numpy as np


def test_supervised_sequences_and_labels(supervised_data):
    """
    Test that the sequences and labels from the DataSupervised dataset match the ground truth data.
    """

    task, split, gt_data, dataset = supervised_data

    print(f"Testing sequences and labels for task: {task}, split: {split}")

    gt_sequences, gt_labels = gt_data
    dataset_sequences, dataset_labels = dataset.sequences, dataset.labels

    assert len(gt_sequences) == len(dataset_sequences)
    assert len(gt_labels) == len(dataset_labels)

    for gt_seq, ds_seq in zip(gt_sequences, dataset_sequences):
        assert gt_seq == ds_seq

    for gt_lbl, ds_lbl in zip(gt_labels, dataset_labels):
        assert np.array_equal(gt_lbl, ds_lbl)
