"""
Initialization file for pytest configuration.
This file sets up fixtures and configurations for testing datasets in the BEND project.
"""

import abc

import h5py
import pandas as pd
import pytest
from bend.io import sequtils
from bend.io.sequtils import Fasta, multi_hot
from hydra import compose, initialize
from torch.utils.data import Subset
from tqdm.auto import tqdm

from bend_hybrid.embedding import datasets
from bend_hybrid.embedding.datasets import DataSupervised

SUPERVISED_TASKS = [
    "gene_finding",
    "histone_modification",
    "cpg_methylation",
    "chromatin_accessibility",
    "enhancer_annotation",
]


class SupervisedData:
    """
    Simple class to hold sequences and label of a specific task.
    """

    def __init__(self, task):
        self.task = task
        self.cfg = None

    def _set_cfg(self, config_path: str, config_name: str, override_task: bool = False):
        """
        Load configuration using Hydra.
        """
        with initialize(version_base=None, config_path=config_path):
            self.cfg = compose(
                config_name=config_name,
                overrides=[f"task={self.task}"] if override_task else None,
            )

    @abc.abstractmethod
    def get_split_names(self) -> list[str]:
        """
        Return the data split names or an empty list if no splits are defined.
        """
        return []

    @abc.abstractmethod
    def get_samples(self, split: str = None) -> list[tuple[str, int]]:
        """
        Return all samples or return the samples for a given split.
        Each sample is a tuple of (sequence, label).
        """
        return [(None, None)]

    def get_cfg(self):
        """Return the Hydra configuration dictionary."""
        return self.cfg


class DefaultSupervisedData(SupervisedData):
    """
    Class to load sequences and labels using default BEND approach.
    """

    def __init__(self, task, config_path, config_name):
        super().__init__(task)

        self._set_cfg(config_path, config_name)

        self.splits = sequtils.get_splits(self.cfg[self.task]["bed"])

        self.data = {split: self._get_sequences_labels(split) for split in self.splits}

    def _get_data_from_bed(
        self,
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
        hdf5_labels = (
            h5py.File(hdf5_labels, mode="r")["labels"] if hdf5_labels else None
        )
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

        samples = []

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
            samples.append((sequence, labels))

        return samples

    def _get_sequences_labels(self, split: str):
        """
        Convert annotations to sequences and labels.
        """

        chunk_size = self.cfg["chunk_size"]
        df = pd.read_csv(self.cfg[self.task]["bed"], sep="\t", low_memory=False)
        df = df[df.iloc[:, -1] == split] if split is not None else df
        chunks = list(range(int(len(df) / chunk_size) + 1))

        data = {}

        for _, chunk in enumerate(chunks):
            chunk_samples = self._get_data_from_bed(
                self.cfg[self.task]["bed"],
                self.cfg[self.task]["reference_fasta"],
                hdf5_labels=self.cfg[self.task].get("hdf5_file", None),
                label_depth=self.cfg[self.task].get("label_depth", None),
                read_strand=self.cfg[self.task]["read_strand"],
                split=split,
                chunk_size=chunk_size,
                chunk=chunk,
            )

            data[f"chunk_{chunk}"] = chunk_samples

        samples = []

        for chunk, sample in tqdm(data.items(), desc="Merging chunks"):
            samples.extend(sample)

        return samples

    def get_split_names(self) -> list[str]:
        return self.splits

    def get_samples(self, split: str = None) -> list[tuple[str, int]]:
        if split is None:
            samples = []
            for split_samples in self.data.values():
                samples.extend(split_samples)
            return samples

        return self.data[split]


class BatchSupervisedData(SupervisedData):
    """
    Class to load data using the DataSupervised dataset.
    """

    def __init__(self, task, config_path, config_name):
        super().__init__(task)

        self._set_cfg(config_path, config_name, override_task=True)

        self.dataset = DataSupervised(
            self.cfg.task.dataset.annotations_path,
            self.cfg.task.dataset.genome_path,
            hdf5_path=self.cfg.task.dataset.get("hdf5_path", None),
            label_depth=self.cfg.task.dataset.get("label_depth", None),
            sequence_length=self.cfg.task.dataset.get("sequence_length", None),
        )

        self.annotations_splits = datasets.get_splits(
            self.cfg.task.dataset.annotations_path
        )

        self.samples = {}  # split -> list of (sequence, label)
        self.subsets = {}  # split -> Subset(dataset, split) object

        for split, annotations in self.annotations_splits.items():
            indices = annotations.index.tolist()
            self.samples[split] = [
                (self.dataset.sequences[idx], self.dataset.labels[idx])
                for idx in indices
            ]
            self.subsets[split] = Subset(self.dataset, indices)

    def get_split_names(self):
        return list(self.annotations_splits.keys())

    def get_samples(self, split: str = None):
        if split is None:
            samples = []
            for split_samples in self.samples.values():
                samples.extend(split_samples)
            return samples

        return self.samples[split]

    def get_dataset(self, split: str = None):
        """
        Return the whole dataset or a subset for a given split.
        """
        if split is None:
            return self.dataset
        return self.subsets[split]


@pytest.fixture(
    params=[task for task in SUPERVISED_TASKS],
    scope="session",
)
def supervised_data(request) -> tuple[str, DefaultSupervisedData, BatchSupervisedData]:
    """
    Fixture to provide task and split data of supervised datasets.
    """

    task = request.param

    return (
        task,
        DefaultSupervisedData(task, "./conf/embedding/", "embed"),
        BatchSupervisedData(task, "../config/", "config"),
    )
