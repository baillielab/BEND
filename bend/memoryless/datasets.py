from tqdm.auto import tqdm
import pysam
import pandas as pd
import numpy as np
import webdataset as wds
import h5py
import pickle
from torch.utils.data import Dataset

DEFAULT_FLANK = 0  # Default flank size for sequence fetching
DEFAULT_LABEL_COLUMN_IDX = 6  # Default index for label column in BED file
DEFAULT_STRAND_COLUMN_IDX = 3  # Default index for strand column in BED file
DEFAULT_SPLIT_COLUMN_IDX = -1  # Default index for split column in BED file

baseComplement = {"A": "T", "C": "G", "G": "C", "T": "A"}


def reverse_complement(dna_string: str):
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

    complement = [baseComplement.get(base, "N") for base in dna_string]
    reversed_complement = reversed(complement)
    return "".join(list(reversed_complement))


# %%
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
            sequence = "".join(reverse_complement(sequence))
        else:
            raise ValueError(f"Unknown strand: {strand}")

        return sequence


class DatasetAnnotations(Dataset):
    def __init__(
        self,
        annotations_path: str,
        genome_path: str,
        label_depth: int = None,
        hdf5_path: str = None,
        default_label_column_idx: int = DEFAULT_LABEL_COLUMN_IDX,
        default_strand_column_idx: int = DEFAULT_STRAND_COLUMN_IDX,
        split: str = None,
        split_column_idx: int = DEFAULT_SPLIT_COLUMN_IDX,
        flank: int = DEFAULT_FLANK,
    ):

        if hdf5_path is None and label_depth is None:
            raise ValueError(
                "Either hdf5_path or label_depth must be provided to initialize DatasetAnnotations."
            )

        if hdf5_path and label_depth:
            raise ValueError(
                "Only one of hdf5_path or label_depth should be provided to initialize DatasetAnnotations."
            )

        annotations = pd.read_csv(annotations_path, sep="\t", low_memory=False)
        genome = Fasta(genome_path)

        mask = None
        if split:
            annotations, mask = self._filter_annotations(
                annotations, split, split_column_idx
            )

        if hdf5_path:
            self.sequences, self.labels = self._get_data_hdf5(
                annotations,
                genome,
                hdf5_path,
                flank,
                mask=mask,
            )

        if label_depth:
            self.sequences, self.labels = self._get_data_multi_hot(
                annotations,
                genome,
                label_depth,
                default_label_column_idx,
                default_strand_column_idx,
                flank,
            )

    def _filter_annotations(self, annotations, split, split_column_idx):
        print(f"Filtering annotations {len(annotations)} for split: {split}")
        # Get only data belonging to specific split
        mask = annotations.iloc[:, split_column_idx] == split
        annotations = annotations[mask]
        annotations = annotations.reset_index(drop=True)
        print(f"Filtered annotations to {len(annotations)} for split: {split}")

        return annotations, mask

    def _get_data_hdf5(self, annotations, genome, hdf5_path, flank, mask=None):
        labels = h5py.File(hdf5_path, mode="r")["labels"]
        if mask is not None:
            labels = labels[mask.to_numpy()]

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
