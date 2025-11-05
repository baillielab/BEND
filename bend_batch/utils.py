"""
Utility functions for the BEND project.
"""

import os
import random
from contextlib import contextmanager
from time import process_time_ns

import numpy as np
import pandas as pd
import torch

import wandb

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


@contextmanager
def log_time(process_name: str, step: int | None = None, log_type: str = "log"):
    """
    Context manager to log the time taken by a task to wandb.

    Parameters
    ----------
    process_name : str
        Name of the process to log.
    step : int, optional
        Step number to log. The default is None.
    """

    start_time = process_time_ns()
    yield
    end_time = process_time_ns()
    if log_type == "log":
        wandb.log({f"{process_name}_ns": (end_time - start_time)}, step=step)
    elif log_type == "summary":
        wandb.summary[f"{process_name}_ns"] = end_time - start_time


def wandb_login():
    """
    Login to wandb using the WANDB_API_KEY environment variable.
    If the variable is not set, print a warning.
    """

    wandb_key = os.environ.get("WANDB_API_KEY", None)
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        print("No Wandb API key found")
