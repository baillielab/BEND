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
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from bend_hybrid.downstream.dataloaders import (
    undersample_dataloaders,
    get_samples_idx_by_split,
)
from bend_hybrid.embedding.datasets import collate_fn
from bend_hybrid.utils import set_seed
import shutil

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

    os.makedirs(cfg.embeddings_output_dir, exist_ok=True)
    embedder = hydra.utils.instantiate(cfg.embedding[cfg.embedder])

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    samples_idx_by_split = dataset.get_samples_idx_by_split()

    split = "train"

    dataloader = DataLoader(
        Subset(dataset, samples_idx_by_split[split]),
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
        maxsize=cfg.get("max_shard_size", 3000000000),
    ) as writer:
        for batch_idx, (sequences, labels) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Embedding {split}"
        ):
            embeddings = embedder(sequences, uneven_length=dataset.is_uneven())

            for sample_idx in tqdm(
                range(len(embeddings)), desc="Writing samples", leave=False
            ):
                sample_key = batch_idx * cfg.task.dataloaders.batch_size + sample_idx
                writer.write(
                    {
                        "__key__": f"sample{sample_key:08d}",
                        "input.npy": embeddings[sample_idx],
                        "output.npy": np.array(labels[sample_idx], dtype=np.int32),
                    }
                )


def iterate_downstream(
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

    dataloaders = hydra.utils.instantiate(
        cfg.task.dataloaders,
        shuffle=None,
        shardshuffle=False,
        test_valid_folds=test_valid_folds,
    )

    n_samples = cfg.task.dataset.get("n_samples", None)
    if n_samples is not None:
        dataloaders = undersample_dataloaders(
            cfg,
            dataloaders,
            n_samples,
            test_valid_folds=test_valid_folds,
        )

    for _, _ in enumerate(dataloaders["train"]):
        pass


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

    start_time = time.time()

    cfg.embeddings_output_dir = os.path.join(
        cfg.embeddings_output_dir, "temp_sweep", cfg.task.name, cfg.embedder
    )
    cfg.output_dir = os.path.join(
        cfg.output_dir, "temp_sweep", cfg.task.name, cfg.embedder
    )

    if cfg.compute_embeddings is True:
        compute_embeddings(cfg)

    if cfg.train_downstream is True:

        if "fold_idx" in cfg.task.dataloaders:
            fold_names = list(
                get_samples_idx_by_split(cfg.task.dataset.annotations_path).keys()
            )

            fold_test = fold_names[0]
            fold_valid = fold_names[1]

            iterate_downstream(cfg, (fold_test, fold_valid))

        iterate_downstream(cfg)

    end_time = time.time()

    # delete embeddings and outputs to save space
    if os.path.exists(cfg.embeddings_output_dir):
        shutil.rmtree(cfg.embeddings_output_dir)
    if os.path.exists(cfg.output_dir):
        shutil.rmtree(cfg.output_dir)

    return end_time - start_time


if __name__ == "__main__":
    run_experiment()  # pylint: disable=E1120
