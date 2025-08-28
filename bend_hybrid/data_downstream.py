"""
data_downstream.py
==================
Data loading and processing utilities for training
downsteam tasks on embeddings saved in webdataset .tar.gz format.
"""

import glob
import os
from functools import partial
from typing import List, Union

# create torch dataset & dataloader from webdataset
import torch
import webdataset as wds

from bend_hybrid.utils import SEED, seed_worker


def pad_to_longest(sequences: List[torch.Tensor], padding_value=-100, batch_first=True):
    """Pad a list of sequences to the longest sequence in the list.
    Parameters
    ----------
    sequences : list[torch.Tensor]
        List of sequences to pad.
    padding_value : int, optional
        Value to pad with. The default is -100.
    batch_first : bool, optional
        Whether to return the batch dimension first. The default is True.
    Returns
    -------
    sequences : torch.Tensor
        Padded sequences.
    """

    sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, padding_value=padding_value, batch_first=batch_first
    )

    return sequences


def collate_fn_pad_to_longest(batch, padding_value=-100):
    """Collate function for dataloader that pads to the longest sequence in the batch.
    Parameters
    ----------
    batch : list
        List of samples to collate.
    padding_value : int, optional
        Value to pad with. The default is -100.
    Returns
    -------
    padded : Tuple[torch.Tensor]
        Padded batch.
    """

    if isinstance(batch, torch.Tensor):
        return batch

    batch = list(zip(*batch))
    padded = tuple(
        map(
            partial(pad_to_longest, padding_value=padding_value, batch_first=True),
            batch,
        )
    )

    if padding_value != 0:  # make sure features do no have padding value
        padded[0][padded[0] == padding_value] = 0

    return padded


def return_dataloader(
    data: Union[str, list],
    batch_size: int = 8,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    padding_value=-100,
    shuffle: int = None,
    shardshuffle: Union[bool, int] = False,
):
    """
    Function to return a dataloader from a list of tar files or a single one.

    Parameters
    ----------
    data : Union[str, list]
        Path to single tar file or list of paths to tar files.
    batch_size : int, optional
        Batch size. The default is 8.
    num_workers : int, optional
        Number of workers for data loading. The default is 0.
    padding_value : int, optional
        Value to pad with. The default is -100.
    shuffle : int, optional
        Whether to shuffle the data. The default is None.
    shardshuffle : Union[bool, int], optional
        Whether to shuffle the shards of the dataset.
        If an int, it will shuffle the first n shards. The default is False (no shuffling).
    """

    # '''Load data to dataloader from a list of paths or a single path'''
    if isinstance(data, str):
        data = [data]

    dataset = wds.WebDataset(data, shardshuffle=shardshuffle, seed=SEED)

    if shuffle is not None:
        # Each worker shuffles the top `shuffle` samples that it loads
        # https://github.com/webdataset/webdataset/issues/71
        dataset = dataset.shuffle(shuffle)

    dataset = (
        dataset.decode()
        .to_tuple("input.npy", "output.npy")
        .map_tuple(torch.from_numpy, torch.from_numpy)
        .map_tuple(torch.squeeze, torch.squeeze)
    )

    # Each worker load samples into batches from its assigned shards
    # Each batch is composed only of samples from the worker's assigned shards
    dataset = dataset.batched(batch_size, collation_fn=None).map(
        partial(collate_fn_pad_to_longest, padding_value=padding_value)
    )

    # number of workers has to be equal or less than the number of shards in the dataset, otherwise it will raise an error
    if num_workers > len(data):
        print(
            f"Number of workers ({num_workers}) is greater than the number of shards ({len(data)}). Setting num_workers to {len(data)}."
        )
        num_workers = len(data)

    dataloader = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        # https://github.com/webdataset/webdataset/issues/151
        prefetch_factor=(
            prefetch_factor if prefetch_factor > 0 and num_workers > 0 else None
        ),
        pin_memory=True if torch.cuda.is_available() and pin_memory else False,
        batch_size=None,
        worker_init_fn=seed_worker,
    )

    return dataloader


def get_data(
    data_dir: str,
    cross_validation: Union[bool, int] = False,
    batch_size: int = 8,
    num_workers: int = 32,
    prefetch_factor: int = 2,
    padding_value=-100,
    shuffle: int = None,
    shardshuffle: Union[bool, int] = False,
    **kwargs,
):
    """
    Function to get data from tar files.

    Parameters
    ----------
    data_dir : str
        Path to data directory containing the tar files.
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

    if cross_validation is not False:
        fold_names = list(
            set([os.path.split(shard)[-1].split("_")[0] for shard in tars])
        )
        fold_names = sorted(fold_names, key=lambda x: int(x.replace("part", "")))

        test_shards = [
            shard for shard in tars if f"{fold_names[cross_validation]}_" in shard
        ]

        val_idx = cross_validation + 1 if cross_validation + 1 < len(fold_names) else 0
        valid_shards = [shard for shard in tars if f"{fold_names[val_idx]}_" in shard]

        train_shards = [
            shard
            for shard in tars
            if shard not in test_shards and shard not in valid_shards
        ]

    else:
        train_shards = [x for x in tars if os.path.split(x)[-1].startswith("train")]
        valid_shards = [x for x in tars if os.path.split(x)[-1].startswith("valid")]
        test_shards = [x for x in tars if os.path.split(x)[-1].startswith("test")]

    dataloaders = {}
    for split, shards in zip(
        ["train", "valid", "test"], [train_shards, valid_shards, test_shards]
    ):
        dataloaders[split] = return_dataloader(
            shards,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            padding_value=padding_value,
            shuffle=shuffle if split == "train" else None,
            shardshuffle=shardshuffle if split == "train" else False,
        )

    return dataloaders
