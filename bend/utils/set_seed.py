import torch
import random
import numpy as np

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
