"""
Run a supervised task experiment by embedding DNA sequences, using compute_embeddings(),
and training a downstream model, using train_downstream().
Configuration is set through Hydra in config/config.yaml.
"""

import os
import time

import hydra
import numpy as np
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from bend_hybrid.downstream.trainer import BaseTrainer
from bend_hybrid.embedding.datasets import DataSupervised, collate_fn, get_splits
from bend_hybrid.utils import get_device, record_embedding_time, set_seed

set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


def compute_embeddings(cfg: DictConfig, annotations_splits: dict) -> None:
    """
    Embed all sequences in the dataset.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    split : str
        The dataset split to embed (e.g., 'train', 'valid', 'test').
    frac_annotations : float, optional
        Fraction of annotations to use for embedding. If None, use all annotations.
        Defaults to None.
    """

    print(
        f"=== Embedding sequences for task: {cfg.task.name} with model: {cfg.embedder} ==="
    )

    os.makedirs(cfg.embeddings_output_dir, exist_ok=True)
    embedder = hydra.utils.instantiate(cfg.embedding[cfg.embedder])

    start_time = time.time()

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
    )

    for split, annotations in annotations_splits.items():
        dataloader = DataLoader(
            Subset(dataset, annotations.index),
            batch_size=cfg.task.dataloader.annotations.batch_size,
            num_workers=cfg.task.dataloader.annotations.num_workers,
            prefetch_factor=cfg.task.dataloader.annotations.prefetch_factor,
            shuffle=True if split == "train" else False,
            collate_fn=collate_fn if dataset.is_uneven() else None,
        )

        with wds.ShardWriter(
            os.path.join(cfg.embeddings_output_dir, f"{split}_%06d.tar.gz"),
            verbose=0,
            compress="gz",
        ) as writer:
            for batch_idx, (sequences, labels) in tqdm(
                enumerate(dataloader), total=len(dataloader), desc=f"Embedding {split}"
            ):
                embeddings = embedder(sequences, uneven_length=dataset.is_uneven())

                for sample_idx in tqdm(
                    range(len(embeddings)), desc="Writing samples", leave=False
                ):
                    sample_key = (
                        batch_idx * cfg.task.dataloader.annotations.batch_size
                        + sample_idx
                    )
                    writer.write(
                        {
                            "__key__": f"sample{sample_key:08d}",
                            "input.npy": embeddings[sample_idx],
                            "output.npy": np.array(labels[sample_idx], dtype=np.int32),
                        }
                    )

    record_embedding_time(
        cfg.task.name,
        cfg.embedder,
        running_time=time.time() - start_time,
        n_samples=len(dataset),
        tar_path=cfg.embeddings_output_dir,
        output_dir=cfg.output_dir,
    )


def train_downstream(cfg: DictConfig) -> None:
    """
    Train the downstream model of a supervised task.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    cfg.output_dir = os.path.join(cfg.output_dir, "downstream")
    os.makedirs(f"{cfg.output_dir}/checkpoints/", exist_ok=True)
    print("output_dir", cfg.output_dir)

    device = get_device()
    model = hydra.utils.instantiate(cfg.task.model).to(device).float()
    optimizer = hydra.utils.instantiate(cfg.task.optimizer, params=model.parameters())

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        config=cfg.task,
        overwrite_dir=True,
    )

    OmegaConf.save(cfg, f"{cfg.output_dir}/config.yaml")

    dataloaders = hydra.utils.instantiate(cfg.task.dataloader.downstream)

    trainer.train(
        dataloaders["train"],
        dataloaders["valid"],
        cfg.task.params.epochs,
        cfg.task.params.load_checkpoint,
    )

    trainer.test(dataloaders["test"], overwrite=False)


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

    if "var" in cfg.task.name:
        print(
            "Skipping experiment for task variant_effects, as it is not a supervised task."
        )
        return

    cfg.embeddings_output_dir = os.path.join(
        cfg.embeddings_output_dir, cfg.task.name, cfg.embedder
    )
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.task.name, cfg.embedder)

    annotations_splits = get_splits(cfg.task.dataset.annotations_path)

    if cfg.compute_embeddings is True:
        compute_embeddings(cfg, annotations_splits)

    if cfg.train_downstream is True:
        print(
            f"=== Training model for task: {cfg.task.name} with embedder: {cfg.embedder} ==="
        )
        if (
            "fold_idx" in cfg.task.dataloader.downstream
            and cfg.task.dataloader.downstream.fold_idx is None
        ):

            output_dir = cfg.output_dir
            n_folds = len(annotations_splits.keys())

            for fold_idx in range(n_folds):
                print(f"=== Running fold {fold_idx + 1}/{n_folds} ===")
                cfg.task.dataloader.downstream.fold_idx = fold_idx
                cfg.output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
                train_downstream(cfg)

        # If not cross-validation, or only one fold specified, train once
        train_downstream(cfg)


if __name__ == "__main__":
    run_experiment()  # pylint: disable=E1120
