"""
Data loading and processing for training downstream tasks on embeddings saved in webdataset .tar.gz format.
"""

import glob
import math
import os
from functools import partial
from typing import List, Union

import pandas as pd
import torch
import webdataset as wds
from omegaconf import DictConfig

from bend_batch.embedding.datasets import DEFAULT_SPLIT_COLUMN_IDX
from bend_batch.utils import SEED, seed_worker


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


def make_dataloader(
    shards: Union[str, list],
    batch_size: int = 8,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    padding_value=-100,
    shuffle: int = None,
    shardshuffle: Union[bool, int] = False,
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader from a list of tar files or a single one.

    Parameters
    ----------
    shards : Union[str, list]
        Path to single tar file or list of paths to tar files.
    batch_size : int, optional
        Batch size. The default is 8.
    num_workers : int, optional
        Number of workers for data loading. The default is 0.
    prefetch_factor : int, optional
        Number of batches to prefetch. The default is 2.
    padding_value : int, optional
        Value to pad with. The default is -100.
    shuffle : int, optional
        Whether to shuffle the data. The default is None.
    shardshuffle : Union[bool, int], optional
        Whether to shuffle the shards of the dataset.
        If an int, it will shuffle the first n shards. The default is False (no shuffling).

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        Dataloader for the tar dataset.
    """

    # '''Load data to dataloader from a list of paths or a single path'''
    if isinstance(shards, str):
        shards = [shards]

    dataset = wds.WebDataset(shards, shardshuffle=shardshuffle, seed=SEED)

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
    if num_workers > len(shards):
        print(
            f"Number of workers ({num_workers}) is greater than the number of shards ({len(shards)}). Setting num_workers to {len(shards)}."
        )
        num_workers = len(shards)

    dataloader = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        # https://github.com/webdataset/webdataset/issues/151
        prefetch_factor=(
            prefetch_factor if prefetch_factor > 0 and num_workers > 0 else None
        ),
        pin_memory=True if torch.cuda.is_available() else False,
        batch_size=None,
        worker_init_fn=seed_worker,
    )

    return dataloader


def get_dataloaders(
    shards_dir: str,
    batch_size: int = 8,
    num_workers: int = 32,
    prefetch_factor: int = 2,
    padding_value=-100,
    shuffle: int = None,
    shardshuffle: Union[bool, int] = False,
    test_valid_folds: tuple[str, str] = None,
    **kwargs,
) -> dict[str, torch.utils.data.DataLoader]:
    """
    Function to get data from tar files.

    Parameters
    ----------
    shards_dir : str
        Path to data directory containing the tar files.
    batch_size : int, optional
        Batch size. The default is 8.
    num_workers : int, optional
        Number of workers for data loading. The default is 0.
    prefetch_factor : int, optional
        Number of batches to prefetch. The default is 2.
    padding_value : int, optional
        Value to pad with. The default is -100.
    shuffle : int, optional
        Whether to shuffle the data. The default is None.
    shardshuffle : Union[bool, int], optional
        Whether to shuffle the shards of the dataset.
        If an int, it will shuffle the first n shards. The default is False (no shuffling).
    test_valid_folds : tuple[str, str], optional
        Tuple containing the fold names to use as test and validation sets. The default is None.
        Example: ('part0', 'part1') will use all tar files containing 'part0' as test set and all tar files containing 'part1' as validation set.

    Returns
    -------
    dict[str, torch.utils.data.DataLoader]
        A dictionary containing the dataloaders for each split.
    """

    if not os.path.exists(shards_dir):
        print(shards_dir)
        raise SystemExit(
            f"The shards directory {shards_dir} does not exist\nExiting script"
        )

    tars = glob.glob(f"{shards_dir}/*.tar.gz")

    if len(tars) == 0:
        raise SystemExit("No embedding shards found.\nExiting script")

    if test_valid_folds is not None:
        fold_test, fold_valid = test_valid_folds

        test_shards = [shard for shard in tars if f"{fold_test}_" in shard]
        valid_shards = [shard for shard in tars if f"{fold_valid}_" in shard]

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
        if len(shards) == 0:
            print(
                f"No {split} shards found in {shards_dir}, skipping {split} dataloader"
            )
            continue

        dataloaders[split] = make_dataloader(
            shards,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            padding_value=padding_value,
            shuffle=shuffle if split == "train" else None,
            shardshuffle=shardshuffle if split == "train" else False,
        )

    return dataloaders


def get_samples_idx_by_split(
    annotations_path: str, split_column_idx: int = DEFAULT_SPLIT_COLUMN_IDX
) -> dict[str, int]:
    """
    Get the number of samples by split from the annotations file.

    Parameters
    ----------
    annotations_path : str
        Path to the annotations file.
    split_column_idx : int
        Index of the column containing the split information.

    Returns
    -------
    dict[str, int]
        A dictionary containing the number of samples for each split.
    """

    if not os.path.exists(annotations_path):
        raise SystemExit(
            f"The annotations file {annotations_path} does not exist\nExiting script"
        )
    annotations = pd.read_csv(annotations_path, sep="\t", low_memory=False)

    samples_idx_by_split = {
        split: annotations[
            annotations.iloc[:, split_column_idx] == split
        ].index.tolist()
        for split in annotations.iloc[:, split_column_idx].unique()
    }

    return samples_idx_by_split


def undersample_dataloaders(
    cfg: DictConfig,
    dataloaders: dict[str, wds.DataPipeline],
    n_samples: int,
    test_valid_folds: tuple[str, str] = None,
) -> dict[str, wds.DataPipeline]:
    """
    Undersample the training and validation dataloaders to have at most n_samples in the training set.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    dataloaders : dict[str, wds.DataPipeline]
        Dictionary with split names as keys and dataloaders as values.
    n_samples : int
        Maximum number of samples to keep in the training set.
    test_valid_folds : tuple[str, str], optional
        Tuple containing the fold names to use as test and validation sets. The default is None.
        Example: ('part0', 'part1') will use all tar files containing 'part0' as test set and all tar files containing 'part1' as validation set.
    Returns
    -------
    dict[str, wds.DataPipeline]
        Undersampled dataloaders dictionary.
    """

    samples_idx_by_split = get_samples_idx_by_split(cfg.task.dataset.annotations_path)

    # calculate the number of samples for each split
    if test_valid_folds is not None:
        fold_test, fold_valid = test_valid_folds

        total_samples = sum(len(indices) for indices in samples_idx_by_split.values())

        samples_number_by_split = {
            "valid": len(samples_idx_by_split[fold_valid]),
            "test": len(samples_idx_by_split[fold_test]),
        }
        samples_number_by_split["train"] = (
            total_samples
            - samples_number_by_split["valid"]
            - samples_number_by_split["test"]
        )
    else:
        samples_number_by_split = {
            split: len(indices) for split, indices in samples_idx_by_split.items()
        }

    # undersample training and validation dataloaders
    undersampling_factor = n_samples / samples_number_by_split["train"]

    if undersampling_factor < 1.0:
        for split, dataloader in dataloaders.items():
            if split == "test" or split not in samples_number_by_split.keys():
                continue

            n_batches = math.ceil(
                int(samples_number_by_split[split] * undersampling_factor)
                / cfg.task.dataloaders.batch_size
            )
            print(
                f"Undersampling {split} dataloader from {samples_number_by_split[split]} to {int(samples_number_by_split[split] * undersampling_factor)} samples ({n_batches} batches)"
            )

            dataloaders[split] = dataloader.with_epoch(n_batches)

    return dataloaders
