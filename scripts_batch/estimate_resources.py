"""
Script to estimate resources for embedding nucleotide sequences and
training the task's downstream model. This script is called by hydra.
"""

import os
import shutil
import time

import hydra
import numpy as np
import webdataset as wds
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from bend.estimate.task_trainer import EstimateTrainer
from bend_batch.datasets import DataSupervised, collate_fn
from bend_batch.utils import get_device, record_embedding_time, set_seed

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

    n_samples = min(len(dataset.sequences), MAX_SAMPLES)

    dataset.sequences = dataset.sequences[:n_samples]
    dataset.labels = dataset.labels[:n_samples]

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

    record_embedding_time(
        cfg.task.task,
        cfg.embedder,
        start_time,
        n_samples,
        cfg.embeddings_output_dir,
        cfg.output_dir,
    )


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

    train_loader, _, _ = hydra.utils.instantiate(cfg.task.data)

    # instantiate trainer
    trainer = EstimateTrainer(
        model=model,
        optimizer=optimizer,
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


# load config
@hydra.main(config_path="../config", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    """
    Run the experiment.
    This function is called by hydra.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    if "nt" in cfg.embedder and (
        cfg.task.task == "enhancer_annotation" or cfg.task.task == "gene_finding"
    ):
        cfg.task.data.batch_size = max(1, cfg.task.data.batch_size // 2)

    cfg.task.data._target_ = cfg.task.data._target_.replace("utils", "estimate")

    embed(cfg)
    train_on_task(cfg)

    # Remove generated embeddings and checkpoints
    shutil.rmtree(cfg.task.data.data_dir, ignore_errors=True)


if __name__ == "__main__":
    run_experiment()
