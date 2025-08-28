import math
from typing import Union

import h5py
import numpy as np
import pandas as pd
import pysam
from Bio import SeqIO
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from bend_hybrid.utils import SEED

DEFAULT_FLANK = 0  # Default flank size for sequence fetching
DEFAULT_LABEL_COLUMN_IDX = 6  # Default index for label column in BED file
DEFAULT_STRAND_COLUMN_IDX = 3  # Default index for strand column in BED file
DEFAULT_SPLIT_COLUMN_IDX = -1  # Default index for split column in BED file


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in a batch.
    Pads sequences and labels to the maximum length in the batch.
    """
    sequences, labels = zip(*batch)

    return sequences, labels


def get_splits(
    annotations_path,
    split_column_idx=DEFAULT_SPLIT_COLUMN_IDX,
):
    annotations = pd.read_csv(annotations_path, sep="\t", low_memory=False)

    splits = annotations.iloc[:, split_column_idx].unique().tolist()

    annotations_splits = {
        split: annotations[annotations.iloc[:, split_column_idx] == split]
        for split in splits
    }

    return annotations_splits


def undersample(annotations_splits, n_samples: Union[int, None] = None):

    if n_samples is not None:
        if not "train" in annotations_splits.keys():
            print("Warning: To use n_samples, the dataset must have a 'train' split.")
            return annotations_splits
        if n_samples <= 0:
            print("Warning: n_samples must be a positive integer.")
            return annotations_splits

        undersampling_ratio = n_samples / len(annotations_splits["train"])

        if undersampling_ratio >= 1.0:
            print(
                f"Warning: n_samples ({n_samples}) is greater than the total number of training annotations. Using all training annotations."
            )
            return annotations_splits

        print(f"Undersampling ratio: {undersampling_ratio:.2f}")

        for split in annotations_splits.keys():
            if split == "test":
                continue

            n_samples_split = max(
                1,
                math.ceil(undersampling_ratio * len(annotations_splits[split])),
            )

            print(f"Undersampling {split} to {n_samples_split} samples.")

            annotations_splits[split] = annotations_splits[split].sample(
                n=n_samples_split,
                random_state=SEED,
            )

    return annotations_splits


class Fasta(pysam.FastaFile):
    """Class for fetching sequences from a reference genome fasta file."""

    def fetch(
        self, chrom: str, start: int, end: int, strand: str = "+", flank: int = 0
    ) -> str:
        """
        Fetch a sequence from the reference genome fasta file.

        Parameters
        ----------
        chrom : str
            Chromosome name.
        start : int
            Start coordinate.
        end : int
            End coordinate.
        strand : str, optional
            Strand. The default is '+'.
            If strand is '-', the sequence will be reverse-complemented before returning.
        flank : int, optional
            Number of bases to add to the start and end coordinates. The default is 0.
        Returns
        -------
        str
            Sequence from the reference genome fasta file.
        """
        sequence = super().fetch(str(chrom), start - flank, end + flank).upper()

        if strand == "+":
            pass
        elif strand == "-":
            sequence = "".join(self._reverse_complement(sequence))
        else:
            raise ValueError(f"Unknown strand: {strand}")

        return sequence

    def _reverse_complement(self, dna_string: str):
        # """Returns the reverse-complement for a DNA string."""
        """
        Returns the reverse-complement for a DNA string.

        Parameters
        ----------
        dna_string : str
            DNA string to reverse-complement.

        Returns
        -------
        str
            Reverse-complement of the input DNA string.
        """

        baseComplement = {"A": "T", "C": "G", "G": "C", "T": "A"}

        complement = [baseComplement.get(base, "N") for base in dna_string]
        reversed_complement = reversed(complement)
        return "".join(list(reversed_complement))


class DataSupervised(Dataset):
    def __init__(
        self,
        annotations_path: str,
        genome_path: str,
        label_depth: int = None,
        hdf5_path: str = None,
        sequence_length: int = None,
        default_label_column_idx: int = DEFAULT_LABEL_COLUMN_IDX,
        default_strand_column_idx: int = DEFAULT_STRAND_COLUMN_IDX,
        flank: int = DEFAULT_FLANK,
    ):

        if hdf5_path is None and label_depth is None:
            raise ValueError(
                "Either hdf5_path or label_depth must be provided to initialize DatasetAnnotations."
            )

        annotations = pd.read_csv(annotations_path, sep="\t", low_memory=False)

        genome = Fasta(genome_path)

        self.sequence_length = sequence_length

        if hdf5_path:
            self.sequences, self.labels = self._get_data_hdf5(
                annotations,
                genome,
                hdf5_path,
                flank,
            )
        else:
            self.sequences, self.labels = self._get_data_multi_hot(
                annotations,
                genome,
                label_depth,
                default_label_column_idx,
                default_strand_column_idx,
                flank,
            )

    def is_uneven(self):
        return True if self.sequence_length is None else False

    def _get_data_hdf5(self, annotations, genome, hdf5_path, flank):
        with h5py.File(hdf5_path, mode="r") as h5f:
            labels = h5f["labels"][()]

        sequences = []
        for idx, item in tqdm(annotations.iterrows(), total=len(annotations)):

            # fetch sequence from genome
            chrom, start, end, strand = (
                item.iloc[0],
                int(item.iloc[1]),
                int(item.iloc[2]),
                "+",
            )

            sequence = genome.fetch(chrom, start, end, strand=strand, flank=flank)
            sequences.append(sequence)

        return sequences, labels

    def _get_data_multi_hot(
        self,
        annotations,
        genome,
        label_depth,
        default_label_column_idx,
        default_strand_column_idx,
        flank,
    ):

        label_column_idx = (
            annotations.columns.get_loc("label")
            if "label" in annotations.columns
            else default_label_column_idx
        )

        strand_column_idx = (
            annotations.columns.get_loc("strand")
            if "strand" in annotations.columns
            else default_strand_column_idx
        )

        sequences = []
        labels = []
        for idx, item in tqdm(annotations.iterrows(), total=len(annotations)):

            # fetch sequence from genome
            chrom, start, end, strand = (
                item.iloc[0],
                int(item.iloc[1]),
                int(item.iloc[2]),
                item.iloc[strand_column_idx],
            )

            sequence = genome.fetch(chrom, start, end, strand=strand, flank=flank)
            if self.sequence_length and len(sequence) != self.sequence_length:
                continue
            sequences.append(sequence)

            # compute labels
            label = item.iloc[label_column_idx]
            label = (
                list(map(int, label.split(","))) if isinstance(label, str) else []
            )  # if no label for sample
            label = self._multi_hot(label, label_depth)

            labels.append(label)

        return sequences, labels

    def _multi_hot(self, labels, num_labels):
        """
        Convert a numpy array to a one-hot encoded numpy array.

        Parameters
        ----------
        labels : list
            The labels that are true
        num_labels : int
            The number of potential labels.

        Returns
        -------
        numpy.ndarray
            A multi-hot encoded numpy array.
        """
        encoded = np.eye(num_labels, dtype=np.int64)[labels].sum(axis=0)
        return encoded

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        sequence = self.sequences[idx]

        labels = self.labels[idx]

        return (sequence, labels)


class DataVariantEffects(Dataset):

    def __init__(
        self,
        annotation_path: str,
        genome_path: str,
        extra_context_left: int = 0,
        extra_context_right: int = 0,
    ):
        super().__init__()

        if (
            extra_context_left != extra_context_right
            and extra_context_left != 0
            and extra_context_right != 0
        ):
            raise ValueError(
                "Left and right context must be equal or one of them must be 0"
            )

        self.annotation = pd.read_csv(annotation_path, sep="\t")
        self.annotation["distance"] = 0.0

        if not {"chromosome", "start", "end", "alt"}.issubset(self.annotation.columns):
            raise ValueError(
                "Annotation dataframe must contain columns: chromosome, start, end, alt"
            )

        self.genome_dict = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))

        # SNP annotation has start position equal to end position -> need to include context
        if extra_context_left == extra_context_right and extra_context_left == 0:
            # avoid having empty sequence in case of no extra context
            extra_context_right += 1

        self.annotation.loc[:, "start"] = (
            self.annotation.loc[:, "start"] - extra_context_left
        )
        self.annotation.loc[:, "end"] = (
            self.annotation.loc[:, "end"] + extra_context_right
        )

        self.idx_alt = extra_context_left
        if extra_context_right == 0:
            # if no right context, alt nucleotide is at the end of the sequence
            self.idx_alt = extra_context_left - 1

    def __len__(self):
        return self.annotation.shape[0]

    def __getitem__(self, idx):
        # return the data and label for the given index

        item = self.annotation.iloc[idx]
        dna_seq = str(
            self.genome_dict[item["chromosome"]].seq[item["start"] : item["end"]]
        )

        alt_dna_seq = [n for n in dna_seq]
        alt_dna_seq[self.idx_alt] = item["alt"]
        alt_dna_seq = "".join(alt_dna_seq)

        return (dna_seq, alt_dna_seq)
