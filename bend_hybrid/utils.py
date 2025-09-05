"""
Utility functions for the BEND project.
"""

import os
import random

import numpy as np
import pandas as pd
import torch

SEED = 42


def set_seed(seed: int = SEED):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}.")


def seed_worker(worker_id: int):
    """
    Set the random seed for each worker in a DataLoader.
    As found in: https://docs.pytorch.org/docs/stable/notes/randomness.html#reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    """
    Get the device to use for training.
    Returns:
        torch.device: The device to use (CPU, CUDA, or MPS).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def record_embedding_time(
    task: str,
    model: str,
    running_time: float,
    n_samples: int,
    tar_path: str,
    output_dir: str,
) -> None:
    """
    Record the time taken for embedding in a CSV file.

    Parameters
    ----------
    task : str
        The name of the task.
    model : str
        The name of the model used for embedding.
    running_time : float
        The time taken to run the embedding.
    n_samples : int
        The number of samples embedded.
    tar_path : str
        The path where the tar files are stored.
    output_dir : str
        The directory where the output CSV file will be saved.
    """

    print(f"Embedding completed in {running_time:.2f} seconds")

    tar_size = sum(
        os.path.getsize(os.path.join(tar_path, f))
        for f in os.listdir(tar_path)
        if f.endswith(".tar.gz")
    )

    file_path = os.path.join(output_dir, "embeddings_stats.csv")

    new_df = pd.DataFrame.from_dict(
        {
            "task": [task],
            "model": [model],
            "time": [running_time],
            "n_samples": [n_samples],
            "size (bytes)": [tar_size],
        }
    )

    if not os.path.exists(file_path):
        os.makedirs(output_dir, exist_ok=True)
        new_df.to_csv(file_path, index=False)

    old_df = pd.read_csv(file_path)
    new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(file_path, index=False)
