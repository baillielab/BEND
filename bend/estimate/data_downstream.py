"""
data_downstream.py
==================
Data loading and processing utilities for training
downsteam tasks on embeddings saved in webdataset .tar.gz format.
"""

# create torch dataset & dataloader from webdataset
import torch
from functools import partial
import os
import glob
from typing import List, Tuple, Union
import webdataset as wds
from bend.utils.set_seed import seed_worker
import numpy as np
from bend.utils.data_downstream import return_dataloader


def get_data(
    data_dir: str,
    train_data: List[str] = None,
    valid_data: List[str] = None,
    test_data: List[str] = None,
    cross_validation: Union[bool, int] = False,
    batch_size: int = 8,
    num_workers: int = 32,
    padding_value=-100,
    shuffle: int = None,
    **kwargs,
):
    """
    Function to get data from tar files.

    Parameters
    ----------
    data_dir : str
        Path to data directory containing the tar files.
    train_data : List[str], optional
        List of paths to train tar files. The default is None.
        In case of cross validation can be simply the path to the data directory.
    valid_data : List[str], optional
        List of paths to valid tar files. The default is None.
    test_data : List[str], optional
        List of paths to test tar files. The default is None.
    cross_validation : Union[bool, int], optional
        If int, use the given partition as test set, +1 as valid set and the rest as train set.
        First split is 1. The default is False.
    batch_size : int, optional
        Batch size. The default is 8.
    num_workers : int, optional
        Number of workers for data loading. The default is 0.
    padding_value : int, optional
        Value to pad with. The default is -100.
    shuffle : int, optional
        Whether to shuffle the data. The default is None.

    Returns
    -------
    train_dataloader : torch.utils.data.DataLoader
        Dataloader for training data.
    valid_dataloader : torch.utils.data.DataLoader
        Dataloader for validation data.
    test_dataloader : torch.utils.data.DataLoader
        Dataloader for test data.
    """
    # check if data exists
    if not os.path.exists(data_dir):
        print(data_dir)
        raise SystemExit(
            f"The data directory {data_dir} does not exist\nExiting script"
        )
    tars = glob.glob(f"{data_dir}/*.tar.gz")
    # else:
    #     # join data_dir with each item in train_data, valid_data and test_data

    #     train_data = [f'{data_dir}/{x}' for x in train_data] if train_data else None
    #     valid_data = [f'{data_dir}/{x}' for x in valid_data] if valid_data else None
    #     test_data = [f'{data_dir}/{x}' for x in test_data] if test_data else None

    # get dataloaders
    # import ipdb; ipdb.set_trace()

    return return_dataloader(
        tars,
        batch_size=batch_size,
        num_workers=num_workers,
        padding_value=padding_value,
        shuffle=shuffle,
    )
