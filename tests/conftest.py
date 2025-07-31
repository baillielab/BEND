"""
Initialization file for pytest configuration.
This file sets up fixtures and configurations for testing datasets in the BEND project.
"""

from tqdm.auto import tqdm
import pandas as pd
import pytest
from hydra import compose, initialize
import h5py
from bend.io.sequtils import multi_hot, Fasta
from bend_batch.datasets import DataSupervised


with initialize(version_base=None, config_path="../conf/embedding/"):
    CFG_DEFAULT = compose(config_name="embed")

SUPERVISED_TASKS = {
    "gene_finding": ["train", "valid", "test"],
    "cpg_methylation": ["train", "valid", "test"],
    "chromatin_accessibility": ["train", "valid", "test"],
    "histone_modification": ["train", "valid", "test"],
    "enhancer_annotation": [f"part{i}" for i in range(1, 10)],
}


def supervised_data_from_bed(
    bed,
    reference_fasta,
    hdf5_labels=None,
    chunk_size=None,
    chunk: int = None,
    read_strand=False,
    label_column_idx=6,
    label_depth=None,
    split=None,
    flank=0,
):
    """
    Load sequences and labels from a BED file, optionally using an HDF5 file for labels.
    """
    fasta = Fasta(reference_fasta)
    f = pd.read_csv(bed, header="infer", sep="\t", engine="python")
    # open hdf5 file
    hdf5_labels = h5py.File(hdf5_labels, mode="r")["labels"] if hdf5_labels else None
    if split:
        mask = f.iloc[:, -1] == split
        f = f[mask]
        if hdf5_labels is not None:
            hdf5_labels = hdf5_labels[mask.to_numpy()]  # mask the labels

    label_column_idx = (
        f.columns.get_loc("label") if "label" in f.columns else label_column_idx
    )
    strand_column_idx = f.columns.get_loc("strand") if "strand" in f.columns else 3

    if chunk is not None:
        # check if chunk is valid
        if chunk * chunk_size > len(f):
            raise ValueError(
                f"Requested chunk {chunk}, but chunk ids range from 0-{int(len(f) / chunk_size)}"
            )
        f = f[chunk * chunk_size : (chunk + 1) * chunk_size].reset_index(drop=True)

    start_offset = chunk * chunk_size

    sequences = []
    targets = []

    for n, line in tqdm(f.iterrows(), total=len(f), desc="Loading sequences"):
        # get bed row
        if read_strand:
            chrom, start, end, strand = (
                line.iloc[0],
                int(line.iloc[1]),
                int(line.iloc[2]),
                line.iloc[strand_column_idx],
            )
        else:
            chrom, start, end, strand = (
                line.iloc[0],
                int(line.iloc[1]),
                int(line.iloc[2]),
                "+",
            )
        if hdf5_labels is not None:
            labels = hdf5_labels[n + start_offset]
        else:
            labels = line.iloc[label_column_idx]
            labels = (
                list(map(int, labels.split(","))) if isinstance(labels, str) else []
            )  # if no label for sample
            labels = multi_hot(labels, label_depth)
        # get sequence
        sequence = fasta.fetch(
            chrom, start, end, strand=strand, flank=flank
        )  # categorical labels

        expected_seq_length = end - start + 2 * flank
        if len(sequence) != expected_seq_length:
            print(
                f"Embedding length does not match sequence length ({len(sequence)} != {expected_seq_length} : {n} {chrom}:{start}-{end}{strand})"
            )
            print(n, chrom, start, end, strand)
            continue
        sequences.append(sequence)
        targets.append(labels)

    return sequences, targets


def get_supervised_gt_data(task, split):
    """
    Load ground truth data for supervised learning.
    """

    chunk_size = CFG_DEFAULT["chunk_size"]
    df = pd.read_csv(CFG_DEFAULT[task]["bed"], sep="\t", low_memory=False)
    df = df[df.iloc[:, -1] == split] if split is not None else df
    chunks = list(range(int(len(df) / chunk_size) + 1))

    data = {}

    for n, chunk in enumerate(chunks):
        sequences, labels = supervised_data_from_bed(
            CFG_DEFAULT[task]["bed"],
            CFG_DEFAULT[task]["reference_fasta"],
            hdf5_labels=CFG_DEFAULT[task].get("hdf5_file", None),
            label_depth=CFG_DEFAULT[task].get("label_depth", None),
            read_strand=CFG_DEFAULT[task]["read_strand"],
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


def get_supervised_dataset(task, split):
    """
    Load a supervised dataset for a given task and split.
    """

    with initialize(version_base=None, config_path="../config/"):
        cfg_batch = compose(config_name="config", overrides=[f"tasks@task={task}"])

    dataset = DataSupervised(
        cfg_batch["task"]["dataset"]["annotations_path"],
        cfg_batch["task"]["dataset"]["genome_path"],
        hdf5_path=cfg_batch["task"]["dataset"].get("hdf5_path", None),
        label_depth=cfg_batch["task"]["dataset"].get("label_depth", None),
        sequence_length=cfg_batch["task"]["dataset"].get("sequence_length", None),
        split=split,
    )

    return dataset


@pytest.fixture(
    params=[
        (task, split) for task, splits in SUPERVISED_TASKS.items() for split in splits
    ],
    scope="session",
)
def supervised_data(request):
    """
    Fixture to provide task and split data of supervised datasets.
    """

    task, split = request.param

    return (
        task,
        split,
        get_supervised_gt_data(task, split),
        get_supervised_dataset(task, split),
    )
