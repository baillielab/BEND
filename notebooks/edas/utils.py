import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_DIR = "../../data/"


def load_annotations(task):
    """
    Load the annotations for a given task from a .bed file.
    Parameters
    ----------
    task : str
        The name of the task for which to load annotations.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the annotations.
    """

    bed_file = os.path.join(DATA_DIR, task, f"{task}.bed")
    annotations = pd.read_csv(bed_file, sep="\t", low_memory=False)

    annotations["chromosome"] = annotations["chromosome"].str.replace("chr", "")

    chromosomes = [f"{idx}" for idx in range(1, 23)] + ["X", "Y"]
    annotations["chromosome"] = pd.Categorical(
        annotations["chromosome"], categories=chromosomes, ordered=True
    )

    return annotations


def plot_chr_in_splits(annotations):

    sns.set_theme(style="white")

    sns.histplot(annotations, x="chromosome", hue="split", multiple="stack")
    ticks = plt.xticks(rotation=45)


def plot_seq_overlap(annotations):
    """
    Plot the sequence overlap in the annotations DataFrame.
    Parameters
    ----------
    annotations : pd.DataFrame
        The DataFrame containing the annotations with 'chromosome', 'start', and 'end' columns.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the overlaps with 'chromosome', 'length', and 'pct_overlap' columns.
    """

    if not all(col in annotations.columns for col in ["chromosome", "start", "end"]):
        raise ValueError(
            "Annotations DataFrame must contain 'chromosome', 'start', and 'end' columns."
        )

    annotations = annotations.sort_values(by=["chromosome", "start"])
    annotations["length"] = annotations["end"] - annotations["start"]

    annotations["overlap"] = (
        annotations["start"]
        .groupby(annotations["chromosome"], observed=True)
        .shift(-1, fill_value=pd.NA)
        - annotations["end"]
    )

    overlap = annotations[["chromosome", "length", "overlap"]].dropna()
    overlap = overlap[overlap["overlap"] < 0]
    overlap["overlap"] = overlap["overlap"].abs()

    overlap["pct_overlap"] = overlap["overlap"] / overlap["length"] * 100

    n_overlaps = overlap["pct_overlap"].describe()["count"]
    print(f"Number of annotations with overlaps: {n_overlaps}")
    n_samples = len(annotations)
    print(f"Total number of samples: {n_samples}")
    pct_overlaps = n_overlaps / n_samples * 100
    print(f"Percentage of annotations with overlaps: {pct_overlaps:.2f}%")

    sns.boxplot(x=overlap["chromosome"], y=overlap["pct_overlap"])

    return overlap


def multi_hot(labels, num_labels):
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


def get_labels(annotations, label_depth):
    """Get the multi-hot encoded labels from the annotations DataFrame.
    Parameters
    ----------
    annotations : pd.DataFrame
        The DataFrame containing the annotations with labels.

    Returns
    -------
    pd.DataFrame, pd.Series
        A DataFrame with multi-hot encoded labels and a Series with the total counts of each label.
    """

    label_column_idx = (
        annotations.columns.get_loc("label") if "label" in annotations.columns else None
    )
    labels = []

    for n, line in annotations.iterrows():
        l = line.iloc[label_column_idx]
        l = (
            list(map(int, l.split(","))) if isinstance(l, str) else []
        )  # if no label for sample
        labels.append(multi_hot(l, label_depth))

    labels_df = pd.DataFrame(labels, columns=[f"label_{i}" for i in range(label_depth)])
    total_labels = labels_df.sum(axis=0)

    return labels_df, total_labels
