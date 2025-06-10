import numpy as np
import pytest


def test_sequences_and_labels(data):

    gt_data, dataset = data
    gt_sequences, gt_labels = gt_data
    dataset_sequences, dataset_labels = dataset.sequences, dataset.labels

    assert len(gt_sequences) == len(dataset_sequences)
    assert len(gt_labels) == len(dataset_labels)

    for gt_seq, ds_seq in zip(gt_sequences, dataset_sequences):
        assert gt_seq == ds_seq

    for gt_lbl, ds_lbl in zip(gt_labels, dataset_labels):
        assert np.array_equal(gt_lbl, ds_lbl)
