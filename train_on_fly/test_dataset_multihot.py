from tqdm.auto import tqdm
import pysam
import pandas as pd
import numpy as np
from bend.io.sequtils import data_from_bed
import pytest
from datasets import DatasetMultiHot
import os

GENOME_PATH = "data/genomes/GRCh38.primary_assembly.genome.fa"
READ_STRAND = True


cfg_cpg = {
    "annotation_path": "data/cpg_methylation/cpg_methylation.bed",
    "genome_path": GENOME_PATH,
    "label_depth": 7,
}

cfg_histone = {
    "annotation_path": "data/histone_modification/histone_modification.bed",
    "genome_path": GENOME_PATH,
    "label_depth": 18,
}


def get_annotation_path(task):
    return os.path.join("data", task, f"{task}.bed")


@pytest.fixture
def gt_data(request):
    cfg, split = request.param
    annotation_path = cfg["annotation_path"]
    genome_path = cfg["genome_path"]
    label_depth = cfg["label_depth"]

    chunk_size = 50000
    df = pd.read_csv(annotation_path, sep="\t", low_memory=False)
    df = df[df.iloc[:, -1] == split] if split is not None else df
    chunks = list(range(int(len(df) / chunk_size) + 1))
    print(f"Splitting {len(df)} rows into {len(chunks)} chunks of size {chunk_size}.")

    data = {}

    for n, chunk in enumerate(chunks):
        sequences, labels = data_from_bed(
            annotation_path,
            genome_path,
            label_depth=label_depth,
            read_strand=READ_STRAND,
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


@pytest.fixture
def dataset(request):
    cfg, split = request.param
    annotation_path = cfg["annotation_path"]
    genome_path = cfg["genome_path"]
    label_depth = cfg["label_depth"]

    dataset = DatasetMultiHot(annotation_path, genome_path, label_depth, split=split)

    return dataset.sequences, dataset.labels


@pytest.mark.parametrize(
    "gt_data, dataset",
    [
        ((cfg_cpg, "valid"), (cfg_cpg, "valid")),
        ((cfg_histone, "valid"), (cfg_histone, "valid")),
        ((cfg_cpg, "train"), (cfg_cpg, "train")),
        ((cfg_histone, "train"), (cfg_histone, "train")),
        ((cfg_cpg, "test"), (cfg_cpg, "test")),
        ((cfg_histone, "test"), (cfg_histone, "test")),
    ],
    indirect=True,
)
def test_sequence_labels(gt_data, dataset):
    gt_sequences, gt_labels = gt_data
    dataset_sequences, dataset_labels = dataset

    assert len(gt_sequences) == len(dataset_sequences)
    assert len(gt_labels) == len(dataset_labels)

    for gt_seq, ds_seq in zip(gt_sequences, dataset_sequences):
        assert gt_seq == ds_seq

    for gt_lbl, ds_lbl in zip(gt_labels, dataset_labels):
        assert np.array_equal(gt_lbl, ds_lbl)
