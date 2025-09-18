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

from bend_hybrid.downstream.dataloaders import (
    undersample_dataloaders,
    get_samples_idx_by_split,
)
from bend_hybrid.downstream.trainer import BaseTrainer
from bend_hybrid.embedding.datasets import collate_fn
from bend_hybrid.utils import get_device, record_embedding_time, set_seed

set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


def compute_embeddings(cfg: DictConfig) -> None:
    """
    Embed all sequences in the dataset.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    print(
        f"=== Embedding sequences for task: {cfg.task.name} with model: {cfg.embedder} ==="
    )

    os.makedirs(cfg.embeddings_output_dir, exist_ok=True)
    embedder = hydra.utils.instantiate(cfg.embedding[cfg.embedder])

    start_time = time.time()

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    samples_idx_by_split = dataset.get_samples_idx_by_split()

    for split, indices in samples_idx_by_split.items():

        split_dataset = Subset(dataset, indices)

        dataloader = DataLoader(
            split_dataset,
            batch_size=cfg.task.dataloaders.batch_size,
            num_workers=cfg.task.dataloaders.num_workers,
            prefetch_factor=(
                cfg.task.dataloaders.prefetch_factor
                if cfg.task.dataloaders.prefetch_factor > 0
                else None
            ),
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
                        batch_idx * cfg.task.dataloaders.batch_size + sample_idx
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


def train_downstream(
    cfg: DictConfig,
    test_valid_folds: tuple[str, str] | None = None,
) -> None:
    """
    Train the downstream model of a supervised task.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    test_valid_folds : tuple[str, str], optional
        Tuple containing the fold names to use as test and validation sets. The default is None.
        Example: ('part0', 'part1') will use all tar files containing 'part0' as test set
        and all tar files containing 'part1' as validation set.
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

    dataloaders = hydra.utils.instantiate(
        cfg.task.dataloaders, test_valid_folds=test_valid_folds
    )

    n_samples = cfg.task.dataset.get("n_samples", None)
    if n_samples is not None:
        dataloaders = undersample_dataloaders(
            cfg,
            dataloaders,
            n_samples,
            test_valid_folds=test_valid_folds,
        )

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

    cfg.embeddings_output_dir = os.path.join(
        cfg.embeddings_output_dir, cfg.task.name, cfg.embedder
    )
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.task.name, cfg.embedder)

    if cfg.compute_embeddings is True:
        compute_embeddings(cfg)

    if cfg.train_downstream is True:
        print(
            f"=== Training model for task: {cfg.task.name} with embedder: {cfg.embedder} ==="
        )

        if "fold_idx" in cfg.task.dataloaders:
            fold_names = list(
                get_samples_idx_by_split(cfg.task.dataset.annotations_path).keys()
            )

            if cfg.task.dataloaders.fold_idx is None:
                # When fold_idx is None, perform cross-validation across all folds
                output_dir = cfg.output_dir
                for fold_idx, fold_test in enumerate(fold_names):
                    print(f"=== Running fold {fold_idx + 1}/{len(fold_names)} ===")

                    cfg.output_dir = os.path.join(output_dir, fold_test)

                    val_idx = fold_idx + 1 if fold_idx + 1 < len(fold_names) else 0
                    fold_valid = fold_names[val_idx]

                    train_downstream(cfg, (fold_test, fold_valid))
            else:
                # When fold_idx is specified, use the specified fold for testing
                fold_idx = cfg.task.dataloaders.fold_idx
                if fold_idx < 0 or fold_idx >= len(fold_names):
                    raise ValueError(f"fold_idx {fold_idx} is out of range.")
                fold_test = fold_names[fold_idx]
                val_idx = fold_idx + 1 if fold_idx + 1 < len(fold_names) else 0
                fold_valid = fold_names[val_idx]

                train_downstream(cfg, (fold_test, fold_valid))

        # If not cross-validation train once
        train_downstream(cfg)


if __name__ == "__main__":
    run_experiment()  # pylint: disable=E1120
