"""
Run a supervised task experiment by embedding DNA sequences, using compute_embeddings(),
and training a downstream model, using train_downstream().
Configuration is set through Hydra in config/config.yaml.
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import hydra
import numpy as np
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import wandb
from bend_batch.embedding.datasets import collate_fn
from bend_batch.embedding.embedders.abstract_class import BaseEmbedder
from bend_batch.utils import log_time, set_seed, wandb_login

set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


def write_batch(shard_dir, split, batch_idx, embeddings, labels):
    """
    Write a batch of embeddings and labels to a tar file using webdataset.
    Parameters
    ----------
    shard_dir : str
        Directory in which to store the shards.
    split : str
        Dataset split (train, val, test).
    batch_idx : int
        Batch index.
    embeddings : np.ndarray
        Batch embeddings to store.
    labels : list[int]
        Batch labels to store.
    """

    os.makedirs(os.path.join(shard_dir), exist_ok=True)

    writer = wds.TarWriter(
        os.path.join(
            shard_dir,
            f"{split}_%06d.tar.gz" % batch_idx,
        ),
        compress="gz",
    )

    for sample_idx, (embedding, label) in enumerate(zip(embeddings, labels)):

        sample_key = batch_idx * len(embeddings) + sample_idx

        writer.write(
            {
                "__key__": f"sample{sample_key:08d}",
                "input.npy": embedding,
                "output.npy": np.array(label, dtype=np.int32),
            }
        )

    writer.close()


def compute_embeddings(cfg: DictConfig, embedder: BaseEmbedder) -> None:
    """
    Embed all sequences in the dataset and stores them using all pooling modes.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    embedder : BaseEmbedder
        Embedder to use.
    """

    with log_time("dataset/init_time", log_type="summary"):
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        samples_idx_by_split = dataset.get_samples_idx_by_split()
    wandb.summary["dataset/n_samples"] = len(dataset)

    batch_step = 0

    for split, indices in samples_idx_by_split.items():

        dataloader = DataLoader(
            Subset(dataset, indices),
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

        with log_time(
            f"dataset/{split}/embedding_time", step=batch_step, log_type="summary"
        ):
            futures = []
            with ProcessPoolExecutor(
                max_workers=cfg.task.dataloaders.num_workers
            ) as executor:

                for batch_idx, (sequences, labels) in tqdm(
                    enumerate(dataloader),
                    total=len(dataloader),
                    desc=f"Embedding {split}",
                ):

                    embeddings = embedder(
                        sequences,
                        cfg.task.dataset.sequence_length,
                    )

                    futures.append(
                        executor.submit(
                            write_batch,
                            cfg.embeddings_output_dir,
                            split,
                            batch_idx,
                            embeddings,
                            labels,
                        )
                    )

                    batch_step += 1

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Completing writes"
                ):
                    future.result()

    tar_path = os.path.join(cfg.embeddings_output_dir)

    if os.path.exists(tar_path):
        tar_size = sum(
            os.path.getsize(os.path.join(tar_path, f))
            for f in os.listdir(tar_path)
            if f.endswith(".tar.gz")
        )
        wandb.summary["dataset/tar_size_bytes"] = tar_size
        wandb.summary["dataset/n_shards"] = len(os.listdir(tar_path))


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

    wandb_login()
    wandb.init(
        anonymous="allow",
        project=cfg.wandb.project,
        name=f"{cfg.task.name}_{cfg.embedder}_embeddings",
        config=OmegaConf.to_container(cfg),
    )
    wandb.summary["task"] = cfg.task.name
    wandb.summary["embedder"] = cfg.embedder

    cfg.embeddings_output_dir = os.path.join(
        cfg.embeddings_output_dir, cfg.task.name, cfg.embedder
    )
    os.makedirs(cfg.embeddings_output_dir, exist_ok=True)

    embedder = hydra.utils.instantiate(cfg.embedding[cfg.embedder])

    print(
        f"=== Embedding sequences for task: {cfg.task.name} with model: {cfg.embedder} ==="
    )

    compute_embeddings(cfg, embedder)

    wandb.finish()


if __name__ == "__main__":
    run_experiment()  # pylint: disable=E1120
