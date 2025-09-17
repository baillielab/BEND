"""
Dataset classes and utility functions for loading sequences and labels of supervised and unsupervised tasks.

- Fasta: Class for fetching sequences from a reference genome fasta file.
- DatasetSupervised: Class for loading sequences and labels for supervised tasks.
- DatasetVariantEffect: Class for loading sequences and labels for variant effect prediction tasks.
"""

import os
import h5py
import numpy as np
import pandas as pd
import pysam
from Bio import SeqIO
from torch.utils.data import Dataset
from bend_hybrid.utils import SEED
from tqdm.auto import tqdm


DEFAULT_FLANK = 0  # Default flank size for sequence fetching
DEFAULT_LABEL_COLUMN_IDX = 6  # Default index for label column in BED file
DEFAULT_STRAND_COLUMN_IDX = 3  # Default index for strand column in BED file
DEFAULT_SPLIT_COLUMN_IDX = -1  # Default index for split column in BED file


def collate_fn(batch) -> tuple:
    """
    Custom collate function to allow loading variable-length sequences in a batch.
    """
    sequences, labels = zip(*batch)

    return sequences, labels


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

        base_complement = {"A": "T", "C": "G", "G": "C", "T": "A"}

        complement = [base_complement.get(base, "N") for base in dna_string]
        reversed_complement = reversed(complement)
        return "".join(list(reversed_complement))


class DataSupervised(Dataset):
    """
    A dataset for loading sequences and labels for supervised learning tasks.

    Methods
    -------
    __init__
        Initialize the dataset with the given parameters.
    is_uneven
        Check if the dataset has uneven sequence lengths.
    _get_data_hdf5
        Convert annotations to sequences and loads labels from an HDF5 file.
        Used by gene finding and enhancer annotation tasks.
    _get_data_multi_hot
        Convert annotations to sequences and generates labels as multi-hot encodings.
        Used by cpg methylation, histone modification, and chromatin accessibility tasks.
    _multi_hot
        Convert a list to a one-hot encoded numpy array.
    """

    def __init__(
        self,
        annotations_path: str,
        genome_path: str,
        label_depth: int = None,
        hdf5_path: str = None,
        sequence_length: int = None,
        samples_to_exclude: list[int] = None,
        n_samples: int = None,
        label_column_idx: int = DEFAULT_LABEL_COLUMN_IDX,
        strand_column_idx: int = DEFAULT_STRAND_COLUMN_IDX,
        split_column_idx: int = DEFAULT_SPLIT_COLUMN_IDX,
        flank: int = DEFAULT_FLANK,
    ):
        """
        Parameters
        ----------
        annotations_path : str
            Path to the annotations file.
        genome_path : str
            Path to the reference genome fasta file.
        label_depth : int, optional
            Depth of the labels. If None, the labels are inferred from the data.
        hdf5_path : str, optional
            Path to the HDF5 file containing precomputed sequences and labels.
        sequence_length : int, optional
            Length of the input sequences. If None, sequences can be of variable length.
        samples_to_exclude : list[int], optional
            List of sample indices to exclude from the dataset.
        n_samples : int, optional
            Number of samples to keep in the dataset. If None, all samples are kept.
        label_column_idx : int, optional
            Index of the label column in the annotations file.
        strand_column_idx : int, optional
            Index of the strand column in the annotations file.
        flank : int, optional
            Number of flanking bases to include in the sequences.
        """

        if hdf5_path is None and label_depth is None:
            raise ValueError(
                "Either hdf5_path or label_depth must be provided to initialize DatasetAnnotations."
            )

        if not os.path.exists(annotations_path):
            raise SystemExit(
                f"The annotations file {annotations_path} does not exist\nExiting script"
            )
        annotations = pd.read_csv(annotations_path, sep="\t", low_memory=False)

        if samples_to_exclude is not None:
            annotations.drop(index=samples_to_exclude, inplace=True)

        undersampled_indices = None
        if n_samples is not None:
            annotations, undersampled_indices = self._undersample(
                annotations, split_column_idx, n_samples=n_samples
            )

        self.samples_idx_by_split = {
            split: annotations[
                annotations.iloc[:, split_column_idx] == split
            ].index.tolist()
            for split in annotations.iloc[:, split_column_idx].unique()
        }

        genome = Fasta(genome_path)
        self.sequence_length = sequence_length

        if hdf5_path:
            # if HDF5 path is provided, load labels from HDF5
            self.samples = self._get_data_hdf5(
                annotations, genome, hdf5_path, flank, undersampled_indices
            )
        else:
            # generate labels as multi-hot encodings
            self.samples = self._get_data_multi_hot(
                annotations,
                genome,
                label_depth,
                label_column_idx,
                strand_column_idx,
                flank,
            )

    def _undersample(self, annotations, split_column_idx, n_samples) -> pd.DataFrame:
        """
        Undersample train and validation splits to have at most n_samples in each split.

        Parameters
        ----------
        annotations : pd.DataFrame
            The annotations dataframe.
        split_column_idx : int
            The index of the split column.
        n_samples : int
            The number of samples to keep in each split.

        Returns
        -------
        pd.DataFrame
            The undersampled annotations dataframe.
        """

        undersampled_indices = None

        if n_samples is not None and n_samples > 1:
            splits = annotations.iloc[:, split_column_idx].unique()

            if "train" in splits:

                train_samples = annotations[
                    annotations.iloc[:, split_column_idx] == "train"
                ]

                undersample_ratio = n_samples / len(train_samples)
                if undersample_ratio < 1.0:
                    undersampled_df_splits = []
                    for split in splits:
                        df_split = annotations[
                            annotations.iloc[:, split_column_idx] == split
                        ]

                        if split == "test":
                            undersampled_df_splits.append(df_split)
                            continue

                        undersampled_df_splits.append(
                            df_split.sample(
                                frac=undersample_ratio, random_state=SEED, replace=False
                            )
                        )
                        print(
                            f"Undersampled {split} split from {len(df_split)} to {len(undersampled_df_splits[-1])} samples"
                        )

                    annotations = pd.concat(undersampled_df_splits)
                    undersampled_indices = annotations.index.to_numpy()
                    annotations = annotations.reset_index(drop=True)
                else:
                    print(
                        f"Warning: Cannot undersample as the number of training samples is less than n_samples ({len(train_samples)} < {n_samples})."
                    )
            else:
                print(
                    'Warning: Cannot undersample as "train" split is not present in the annotations.'
                )

        return annotations, undersampled_indices

    def _get_data_hdf5(
        self, annotations, genome, hdf5_path, flank, undersampled_indices=None
    ) -> tuple[list[str], np.ndarray]:
        """
        Convert annotations to sequences and loads labels from an HDF5 file.
        Creates a dictionary mapping split names to lists of sample indices.
        If self.sequence_length is not None, sequences that do not match the length are filtered out.

        Parameters
        ----------
        annotations : pd.DataFrame
            The annotations dataframe.
        genome : Fasta
            The genome fasta object.
        hdf5_path : str
            The path to the HDF5 file.
        flank : int
            The number of flanking bases to include in the sequences.

        Returns
        -------
        tuple[list[str], np.ndarray]
            A tuple containing the list of sequences and the array of labels.
        dict[str, list[int]]
            A dictionary mapping split names to lists of sample indices.
        """

        with h5py.File(hdf5_path, mode="r") as h5f:
            labels = h5f["labels"][()]

        if undersampled_indices is not None:
            labels = labels[undersampled_indices]

        sequences = []
        for _, item in tqdm(annotations.iterrows(), total=len(annotations)):

            # fetch sequence from genome
            chrom, start, end, strand = (
                item.iloc[0],
                int(item.iloc[1]),
                int(item.iloc[2]),
                "+",
            )

            sequence = genome.fetch(chrom, start, end, strand=strand, flank=flank)

            sequences.append(sequence)

        samples = list(zip(sequences, labels))

        return samples

    def _get_data_multi_hot(
        self,
        annotations,
        genome,
        label_depth,
        default_label_column_idx,
        default_strand_column_idx,
        flank,
    ) -> tuple[tuple[list[str], np.ndarray], dict[str, list[int]]]:
        """
        Converts annotations to sequences and generates labels as multi-hot encodings.
        Creates a dictionary mapping split names to lists of sample indices.
        If self.sequence_length is not None, sequences that do not match the length are filtered out.

        Parameters
        ----------
        annotations : pd.DataFrame
            The annotations dataframe.
        genome : Fasta
            The genome fasta object.
        label_depth : int
            The depth of the labels.
        default_label_column_idx : int
            The default index of the label column in the annotations file.
        default_strand_column_idx : int
            The default index of the strand column in the annotations file.
        flank : int
            The number of flanking bases to include in the sequences.

        Returns
        -------
        tuple[list[str], np.ndarray]
            A tuple containing the list of sequences and the array of labels.
        dict[str, list[int]]
            A dictionary mapping split names to lists of sample indices.
        """

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

        samples = []

        for _, item in tqdm(annotations.iterrows(), total=len(annotations)):

            # fetch sequence from genome
            chrom, start, end, strand = (
                item.iloc[0],
                int(item.iloc[1]),
                int(item.iloc[2]),
                item.iloc[strand_column_idx],
            )
            sequence = genome.fetch(chrom, start, end, strand=strand, flank=flank)

            # compute labels
            label = item.iloc[label_column_idx]
            label = (
                list(map(int, label.split(","))) if isinstance(label, str) else []
            )  # if no label for sample
            label = self._multi_hot(label, label_depth)

            samples.append((sequence, label))

        return samples

    def _multi_hot(self, labels: list, num_labels: int) -> np.ndarray:
        """
        Convert a list to a one-hot encoded numpy array.

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

    def is_uneven(self) -> bool:
        """
        Check if the dataset has uneven sequence lengths.
        """
        return True if self.sequence_length is None else False

    def get_samples_idx_by_split(self) -> dict[str, list[int]]:
        """
        Get the sample indices by split.

        Returns
        -------
        dict[str, list[int]]
            A dictionary mapping split names to lists of sample indices.
        """
        return self.samples_idx_by_split

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[str, np.ndarray]:
        """
        Get the sequence and labels for a given index.
        """
        return self.samples[idx]


class DataVariantEffects(Dataset):
    """
    Dataset for variant effects tasks.

    Methods
    -------

    """

    def __init__(
        self,
        annotation_path: str,
        genome_path: str,
        extra_context_left: int = 0,
        extra_context_right: int = 0,
        n_samples: int = None,
    ):
        """
        Parameters
        ----------
        annotation_path : str
            The path to the annotation file.
        genome_path : str
            The path to the genome fasta file.
        extra_context_left : int
            The number of extra context bases to include on the left.
        extra_context_right : int
            The number of extra context bases to include on the right.
        n_samples : int, optional
            The number of samples to keep in the dataset. If None, all samples are kept.
        """

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
        if not {"chromosome", "start", "end", "alt"}.issubset(self.annotation.columns):
            raise ValueError(
                "Annotation dataframe must contain columns: chromosome, start, end, alt"
            )
        if n_samples is not None:
            self.annotation = self._undersample(n_samples)

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

    def _undersample(self, n_samples: int) -> pd.DataFrame:
        """
        Undersample the annotations to have at most n_samples.

        Parameters
        ----------
        n_samples : int
            The number of samples to keep in the dataset.

        Returns
        -------
        pd.DataFrame
            The undersampled dataset.
        """

        if n_samples is not None and n_samples > 1 and len(self.annotation) > n_samples:
            print(f"Undersampling from {len(self.annotation)} to {n_samples} samples")
            self.annotation = self.annotation.sample(
                n=n_samples, random_state=SEED, replace=False
            ).reset_index(drop=True)

        return self.annotation

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        """
        return self.annotation.shape[0]

    def __getitem__(self, idx) -> tuple[str, str, int]:
        """
        Get the reference and variant sequences for a given index.
        """

        item = self.annotation.iloc[idx]
        dna_seq = str(
            self.genome_dict[item["chromosome"]].seq[item["start"] : item["end"]]
        )

        alt_dna_seq = [n for n in dna_seq]
        alt_dna_seq[self.idx_alt] = item["alt"]
        alt_dna_seq = "".join(alt_dna_seq)

        return (dna_seq, alt_dna_seq, item["label"])
