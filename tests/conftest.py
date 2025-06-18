from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from bend.io.sequtils import data_from_bed
import pytest
from bend.utils.datasets import DatasetAnnotations
import os
from hydra import compose, initialize


TASKS = ["gene_finding", "cpg_methylation"]
SPLITS = ["train", "valid", "test"]

with initialize(version_base=None, config_path="../conf/embedding/"):
    CFG = compose(config_name="embed")


def get_gt_data(task, split):
    chunk_size = CFG["chunk_size"]
    df = pd.read_csv(CFG[task]["bed"], sep="\t", low_memory=False)
    df = df[df.iloc[:, -1] == split] if split is not None else df
    chunks = list(range(int(len(df) / chunk_size) + 1))
    print(f"Splitting {len(df)} rows into {len(chunks)} chunks of size {chunk_size}.")

    data = {}

    for n, chunk in enumerate(chunks):
        sequences, labels = data_from_bed(
            CFG[task]["bed"],
            CFG[task]["reference_fasta"],
            hdf5_file=CFG[task].get("hdf5_file", None),
            label_depth=CFG[task].get("label_depth", None),
            read_strand=CFG[task]["read_strand"],
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

    return (sequences, labels)


def get_dataset(task, split):
    dataset = DatasetAnnotations(
        CFG[task]["bed"],
        CFG[task]["reference_fasta"],
        hdf5_path=CFG[task].get("hdf5_file", None),
        label_depth=CFG[task].get("label_depth", None),
        split=split,
    )

    return dataset


@pytest.fixture(
    params=[(task, split) for task in TASKS for split in SPLITS], scope="session"
)
def data(request):
    task, split = request.param

    return task, split, get_gt_data(task, split), get_dataset(task, split)
