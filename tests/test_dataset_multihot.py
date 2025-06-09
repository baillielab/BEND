from tqdm.auto import tqdm
import pysam
import pandas as pd
import numpy as np
from bend.io.sequtils import data_from_bed
import pytest
from bend.utils.datasets import DatasetMultiHot
import os
from hydra import compose, initialize
from omegaconf import OmegaConf


with initialize(version_base=None, config_path="../conf/embedding/"):
    cfg = compose(config_name="embed")
    # cfg = OmegaConf.to_yaml(cfg)
    # print(cfg)


def get_annotation_path(task):
    return os.path.join("data", task, f"{task}.bed")


def get_gt_data(task, split):
    # task, split = request.param

    chunk_size = cfg["chunk_size"]
    df = pd.read_csv(cfg[task]["bed"], sep="\t", low_memory=False)
    df = df[df.iloc[:, -1] == split] if split is not None else df
    chunks = list(range(int(len(df) / chunk_size) + 1))
    print(f"Splitting {len(df)} rows into {len(chunks)} chunks of size {chunk_size}.")

    data = {}

    for n, chunk in enumerate(chunks):
        sequences, labels = data_from_bed(
            cfg[task]["bed"],
            cfg[task]["reference_fasta"],
            label_depth=cfg[task]["label_depth"],
            read_strand=cfg[task]["read_strand"],
            split=split,
            chunk_size=chunk_size,
            chunk=chunk,
        )

        data[f"chunk_{chunk}"] = (sequences, labels)

    sequences = []
    labels = []

    for chunk, (seqs, lbls) in tqdm(data.items(), desc="Merging chunks"):
        sequences.extend(seqs)
        labels.extend(lbls)

    return sequences, labels


def get_dataset_data(task, split):
    dataset = DatasetMultiHot(
        cfg[task]["bed"],
        cfg[task]["reference_fasta"],
        label_depth=cfg[task]["label_depth"],
        split=split,
    )

    return dataset.sequences, dataset.labels


@pytest.fixture
def data(request):
    task, split = request.param

    gt_seq, gt_labels = get_gt_data(task, split)
    dataset_seq, dataset_labels = get_dataset_data(task, split)

    return gt_seq, gt_labels, dataset_seq, dataset_labels


@pytest.mark.parametrize(
    "data",
    [
        ("cpg_methylation", "train"),
        ("cpg_methylation", "valid"),
        ("cpg_methylation", "test"),
        ("histone_modification", "train"),
        ("histone_modification", "valid"),
        ("histone_modification", "test"),
        ("chromatin_accessibility", "train"),
        ("chromatin_accessibility", "valid"),
        ("chromatin_accessibility", "test"),
    ],
    indirect=True,
)
def test_sequence_labels(data):
    gt_sequences, gt_labels, dataset_sequences, dataset_labels = data

    assert len(gt_sequences) == len(dataset_sequences)
    assert len(gt_labels) == len(dataset_labels)

    for gt_seq, ds_seq in zip(gt_sequences, dataset_sequences):
        assert gt_seq == ds_seq

    for gt_lbl, ds_lbl in zip(gt_labels, dataset_labels):
        assert np.array_equal(gt_lbl, ds_lbl)
