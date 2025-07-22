import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_DIR = "../../data/"


def plot_chr_in_splits(annotations, chromosomes):
    sns.set_theme(style="white")

    annotations["chromosome"] = pd.Categorical(
        annotations["chromosome"], categories=chromosomes, ordered=True
    )

    sns.histplot(annotations, x="chromosome", hue="split", multiple="stack")
    ticks = plt.xticks(rotation=45)


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
