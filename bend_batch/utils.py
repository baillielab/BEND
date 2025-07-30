import torch
import random
import numpy as np
import os
import time
import pandas as pd

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
    task: str, model: str, start_time: float, output_dir: str
) -> None:
    """
    Record the time taken for embedding in a CSV file.
    Parameters
    ----------
    start_time : float
        The start time of the embedding process.
    """

    end_time = time.time()
    print(f"Embedding completed in {end_time - start_time:.2f} seconds")

    file_path = os.path.join(output_dir, "embedding_times.csv")

    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data = data._append(
            {
                "task": task,
                "embedder": model,
                "time": end_time - start_time,
            },
            ignore_index=True,
        )
        data.to_csv(file_path, index=False)
    else:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(
            {
                "task": [task],
                "embedder": [model],
                "time": [end_time - start_time],
            }
        ).to_csv(file_path, index=False)
