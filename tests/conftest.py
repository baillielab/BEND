"""
Initialization file for pytest configuration.
This file sets up fixtures and configurations for testing datasets in the BEND project.
"""

import abc

from typing import Generator

import h5py
import hydra
import pandas as pd
import pytest
from bend.io import sequtils
from bend.io.sequtils import Fasta, multi_hot
from bend.utils import Annotation
from hydra import compose, initialize
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from bend_hybrid.embedding.datasets import (
    DataVariantEffects,
    collate_fn,
)

SUPERVISED_TASKS = [
    "gene_finding",
    "histone_modification",
    "cpg_methylation",
    "chromatin_accessibility",
    "enhancer_annotation",
]
UNSUPERVISED_TASKS = ["var_effects_expression", "var_effects_disease"]

EXTRA_CONTEXT = {True: (512, 0), False: (256, 256)}  # autoregressive: (left, right)


def compose_cfg(config_path: str, config_name: str, task: str = None):
    """
    Load configuration using Hydra.
    """
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(
            config_name=config_name,
            overrides=[f"task={task}"] if task is not None else None,
        )
    return cfg


DEFAULT_CFG = compose_cfg("./conf/embedding/", "embed")
BATCH_CFG = {
    task: compose_cfg("../config/", "config", task)
    for task in SUPERVISED_TASKS + UNSUPERVISED_TASKS
}

SUPERVISED_DATASETS = (
    "supervised_dataset",
    [
        pytest.param(
            task,
            id=f"{task}",
        )
        for task in SUPERVISED_TASKS
    ],
)

VARIANT_EFFECTS_DATASETS = (
    "var_eff_dataset",
    [
        pytest.param(
            (task, autoregressive),
            id=f"{task}-{EXTRA_CONTEXT[autoregressive]}",
        )
        for task in UNSUPERVISED_TASKS
        for autoregressive in [True, False]
    ],
)


class Data:
    """
    Simple class to hold sequences and label of a specific task.
    """

    def __init__(self, task):
        self.task = task

    @abc.abstractmethod
    def get_split_names(self) -> list[str]:
        """
        Return the data split names or an empty list if no splits are defined.
        """
        return []

    @abc.abstractmethod
    def next_sample(self, **kwargs):
        """
        Generator to yield samples one by one.
        """
        yield


class DefaultSupervisedData(Data):
    """
    Class to load sequences and labels using default BEND approach.
    """

    def __init__(self, task):
        super().__init__(task)

        self.splits = sequtils.get_splits(DEFAULT_CFG[self.task]["bed"])

        self.samples = {
            split: self._fetch_sequences_labels(split) for split in self.splits
        }

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

    def _fetch_sequences_labels(self, split: str):
        """
        Convert annotations to sequences and labels.
        """

        chunk_size = DEFAULT_CFG["chunk_size"]
        df = pd.read_csv(DEFAULT_CFG[self.task]["bed"], sep="\t", low_memory=False)
        df = df[df.iloc[:, -1] == split] if split is not None else df
        chunks = list(range(int(len(df) / chunk_size) + 1))

        samples = []

        for chunk_idx in chunks:
            chunk_samples = self._get_data_from_bed(
                DEFAULT_CFG[self.task]["bed"],
                DEFAULT_CFG[self.task]["reference_fasta"],
                hdf5_labels=DEFAULT_CFG[self.task].get("hdf5_file", None),
                label_depth=DEFAULT_CFG[self.task].get("label_depth", None),
                read_strand=DEFAULT_CFG[self.task]["read_strand"],
                split=split,
                chunk_size=chunk_size,
                chunk=chunk_idx,
            )

            samples.extend(chunk_samples)

        return samples

    def get_split_names(self) -> list[str]:
        return self.splits

    def next_sample(self, **kwargs):
        """
        Generator to yield samples of a given split one by one.
        Each sample is a tuple of (sequence, label).
        """
        split = kwargs.get("split", None)
        if split is None:
            raise ValueError("Missing required argument: 'split'")

        for dna, label in self.samples[split]:
            yield (dna, label)


class BatchSupervisedData(Data):
    """
    Class to load data using the DataSupervised dataset.
    """

    def __init__(self, task):
        super().__init__(task)

        cfg = BATCH_CFG[task]

        self.dataset = hydra.utils.instantiate(cfg.task.dataset)

        samples_idx_by_split = self.dataset.get_samples_idx_by_split()
        self.subsets = {}  # split -> Subset(dataset, split) object

        for split, indices in samples_idx_by_split.items():
            self.subsets[split] = Subset(self.dataset, indices)

    def get_split_names(self):
        return list(self.subsets.keys())

    def next_sample(self, **kwargs):
        """
        Generator to yield samples of a given split one by one.
        Each sample is a tuple of (sequence, label).
        """
        split = kwargs.get("split", None)
        if split is None:
            raise ValueError("Missing required argument: 'split'")

        dataloader = DataLoader(
            self.subsets[split],
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn if self.dataset.is_uneven() else None,
        )

        for dna, label in dataloader:
            yield dna[0], label[0]  # remove batch dimension


@pytest.fixture(
    scope="session",
)
def supervised_dataset(
    request,
) -> tuple[str, DefaultSupervisedData, BatchSupervisedData]:
    """
    Fixture to provide task and split data of supervised datasets.
    """

    task = request.param

    return (
        task,
        DefaultSupervisedData(task),
        BatchSupervisedData(task),
    )


class DefaultVariantEffectsData(Data):
    """
    Class to load sequences of variant effects datasets using default BEND approach.
    """

    def __init__(
        self,
        task: str,
        annotation_path: str,
        reference_path: str,
        extra_context_left: int = 0,
        extra_context_right: int = 0,
    ):
        super().__init__(task)

        self.samples = self._fetch_sequences_labels(
            annotation_path,
            reference_path,
            extra_context_left,
            extra_context_right,
        )

    def _fetch_sequences_labels(
        self,
        annotation_path: str,
        reference_path: str,
        extra_context_left: int = 0,
        extra_context_right: int = 0,
    ) -> list[tuple[str, str, int]]:
        """
        Convert annotations to reference and SNP sequences, and labels.
        """

        genome_annotation = Annotation(annotation_path, reference_genome=reference_path)
        if extra_context_left > 0 or extra_context_right > 0:
            genome_annotation.extend_segments(
                extra_context_left=extra_context_left,
                extra_context_right=extra_context_right,
            )

        samples = []

        # iterate over the genome annotation
        for index, row in tqdm(genome_annotation.annotation.iterrows()):

            # get the reference and alternate dna sequences
            dna = genome_annotation.get_dna_segment(index=index)
            dna_alt = [x for x in dna]
            if extra_context_left == extra_context_right:
                dna_alt[len(dna_alt) // 2] = row["alt"]
            elif extra_context_right == 0:
                dna_alt[-1] = row["alt"]
            elif extra_context_left == 0:
                dna_alt[0] = row["alt"]
            else:
                raise ValueError("Not implemented")
            dna_alt = "".join(dna_alt)

            samples.append((dna, dna_alt, row["label"]))

        return samples

    def get_split_names(self) -> list[str]:
        return []

    def next_sample(self, **kwargs) -> Generator[str, str, int]:
        """
        Generator to yield samples one by one.
        Each sample is a tuple of (reference sequence, SNP sequence, label).
        """
        for dna, dna_alt, label in self.samples:
            yield dna, dna_alt, label


class BatchVariantEffectsData(Data):
    """
    Class to load sequences of variant effects datasets using DataVariantEffects dataset.
    """

    def __init__(
        self,
        task: str,
        extra_context_left: int = 0,
        extra_context_right: int = 0,
    ):
        super().__init__(task)

        cfg = BATCH_CFG[task]

        self.dataset = DataVariantEffects(
            annotation_path=cfg.task.dataset.annotations_path,
            genome_path=cfg.task.dataset.genome_path,
            extra_context_left=extra_context_left,
            extra_context_right=extra_context_right,
        )

    def get_split_names(self) -> list[str]:
        return []

    def next_sample(self, **kwargs) -> Generator[str, str, int]:
        """
        Generator to yield samples one by one.
        Each sample is a tuple of (reference sequence, SNP sequence, label).
        """
        dataloader = DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=1
        )

        for ref_dna, snp_dna, label in dataloader:
            yield ref_dna[0], snp_dna[0], label.item()  # remove batch dimension


@pytest.fixture(scope="session")
def var_eff_dataset(
    request,
) -> tuple[str, bool, DefaultVariantEffectsData, BatchVariantEffectsData]:
    """
    Fixture to provide DefaultVariantEffectsData and BatchVariantEffectsData instances.
    """

    task, autoregressive = request.param
    extra_context_left, extra_context_right = EXTRA_CONTEXT[autoregressive]

    return (
        task,
        autoregressive,
        DefaultVariantEffectsData(
            task,
            annotation_path=f"./data/variant_effects/{task.replace('var', 'variant')}.bed",
            reference_path="./data/genomes/GRCh38.primary_assembly.genome.fa",
            extra_context_left=extra_context_left,
            extra_context_right=extra_context_right,
        ),
        BatchVariantEffectsData(
            task,
            extra_context_left=extra_context_left,
            extra_context_right=extra_context_right,
        ),
    )
