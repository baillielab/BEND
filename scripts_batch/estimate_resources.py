"""
train_on_task.py
----------------
Train a model on a downstream task.
"""

import hydra
from bend.utils.task_trainer import (
    MSELoss,
    BCEWithLogitsLoss,
    PoissonLoss,
    CrossEntropyLoss,
)
from bend.estimate.task_trainer import EstimateTrainer
from bend_batch.utils import get_device, set_seed, record_embedding_time
import shutil
from omegaconf import DictConfig, OmegaConf
import torch
import os
import pandas as pd
import numpy as np
import webdataset as wds
from bend_batch.datasets import DataSupervised, DEFAULT_SPLIT_COLUMN_IDX, collate_fn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time


EPOCHS = 1
MAX_SAMPLES = 50000
set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


def embed(cfg: DictConfig) -> None:
    """
    Run a embedding of nucleotide sequences.
    This function is called by hydra.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    print(f"== Run embedding for task: {cfg.task.task} with model: {cfg.embedder} ===")

    os.makedirs(cfg.embeddings_output_dir, exist_ok=True)

    embedder = hydra.utils.instantiate(cfg.embedding[cfg.embedder])

    start_time = time.time()
    split = "train" if cfg.task.task != "enhancer_annotation" else "part5"

    print(f"=== Processing split: {split} ===")

    print("Loading dataset ...")
    dataset = DataSupervised(
        annotations_path=cfg.task.dataset.annotations_path,
        genome_path=cfg.task.dataset.genome_path,
        label_depth=(
            cfg.task.dataset.label_depth if "label_depth" in cfg.task.dataset else None
        ),
        hdf5_path=(
            cfg.task.dataset.hdf5_path if "hdf5_path" in cfg.task.dataset else None
        ),
        sequence_length=cfg.task.dataset.sequence_length,
        split=split,
    )

    dataset.sequences = (
        dataset.sequences[:MAX_SAMPLES]
        if MAX_SAMPLES is not None
        else dataset.sequences
    )
    dataset.labels = (
        dataset.labels[:MAX_SAMPLES] if MAX_SAMPLES is not None else dataset.labels
    )

    is_data_uneven = True if cfg.task.dataset.sequence_length is None else False

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.task.data.batch_size,
        num_workers=cfg.task.data.num_workers,
        shuffle=False,
        collate_fn=collate_fn if is_data_uneven else None,
    )

    with wds.ShardWriter(
        os.path.join(cfg.embeddings_output_dir, f"{split}_%06d.tar.gz"),
        verbose=0,
        compress="gz",
    ) as writer:
        for batch_idx, (sequences, labels) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Embedding {split}"
        ):
            embeddings = embedder(sequences, uneven_length=is_data_uneven)

            for sample_idx in tqdm(
                range(len(embeddings)), desc="Writing samples", leave=False
            ):
                sample_key = batch_idx * cfg.task.data.batch_size + sample_idx
                writer.write(
                    {
                        "__key__": f"sample{sample_key:08d}",
                        "input.npy": embeddings[sample_idx],
                        "output.npy": np.array(labels[sample_idx], dtype=np.int32),
                    }
                )

    record_embedding_time(cfg.task.task, cfg.embedder, start_time, cfg.output_dir)


def train_on_task(cfg: DictConfig) -> None:
    """
    Run a supervised task experiment.
    This function is called by hydra.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    device = get_device()

    os.makedirs(f"{cfg.output_dir}", exist_ok=True)
    print("output_dir", cfg.output_dir)

    model = hydra.utils.instantiate(cfg.task.model).to(device).float()

    optimizer = hydra.utils.instantiate(cfg.task.optimizer, params=model.parameters())

    # define criterion
    print(f"Use {cfg.task.params.criterion} loss function")
    match cfg.task.params.criterion:
        case "cross_entropy":
            criterion = CrossEntropyLoss(
                ignore_index=cfg.task.data.padding_value,
                weight=(
                    torch.tensor(cfg.task.params.class_weights).to(device)
                    if cfg.task.params.class_weights is not None
                    else None
                ),
            )
        case "poisson_nll":
            criterion = PoissonLoss()
        case "mse":
            criterion = MSELoss()
        case "bce":
            criterion = BCEWithLogitsLoss(
                class_weights=(
                    torch.tensor(cfg.task.params.class_weights).to(device)
                    if cfg.task.params.class_weights is not None
                    else None
                )
            )

    train_loader, _, _ = hydra.utils.instantiate(cfg.task.data)

    # instantiate trainer
    trainer = EstimateTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=cfg.task,
        overwrite_dir=True,
    )

    if cfg.task.params.mode == "train":
        trainer.train(
            train_loader,
            None,
            None,
            EPOCHS,
            False,
        )

    shutil.rmtree(cfg.task.data.data_dir, ignore_errors=True)


# load config
@hydra.main(config_path=f"../config", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    """
    Run the experiment.
    This function is called by hydra.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    embed(cfg)
    train_on_task(cfg)


if __name__ == "__main__":
    run_experiment()
