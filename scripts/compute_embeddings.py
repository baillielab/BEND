"""
Run a supervised task experiment by embedding DNA sequences, using compute_embeddings(),
and training a downstream model, using train_downstream().
Configuration is set through Hydra in config/config.yaml.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import hydra
import numpy as np
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import wandb
from bend_hybrid.embedding.datasets import collate_fn
from bend_hybrid.embedding.embedders import BaseEmbedder
from bend_hybrid.utils import log_time, set_seed, wandb_login

set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


def write_batch(shard_dir, split, mode, batch_idx, embeddings, labels):
    """
    Write a batch of embeddings and labels to a tar file using webdataset.
    Parameters
    ----------
    shard_dir : str
        Directory in which to store the shards.
    split : str
        Dataset split (train, val, test).
    mode : str
        Pooling mode used.
    batch_idx : int
        Batch index.
    embeddings : np.ndarray
        Batch embeddings to store.
    labels : list[int]
        Batch labels to store.
    """

    os.makedirs(os.path.join(shard_dir, mode), exist_ok=True)

    writer = wds.TarWriter(
        os.path.join(
            shard_dir,
            mode,
            f"{split}_%06d.tar.gz" % batch_idx,
        ),
        compress="gz",
    )

    # print(f"Writing batch {batch_idx} with mode {mode} at {time.time()}")

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

    embeddings_modes = set()

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

        futures = []
        with ProcessPoolExecutor(
            max_workers=cfg.task.dataloaders.num_workers
        ) as executor:

            for batch_idx, (sequences, labels) in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=f"Embedding {split}",
            ):

                with log_time("dataset/batch_embed_time", step=batch_step):
                    output = embedder(sequences, uneven_length=dataset.is_uneven())

                for embeddings, mode in output:
                    embeddings_modes.add(mode)

                    futures.append(
                        executor.submit(
                            write_batch,
                            cfg.embeddings_output_dir,
                            split,
                            mode,
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

    wandb.summary["dataset/embeddings_modes"] = list(embeddings_modes)

    for mode in wandb.summary["dataset/embeddings_modes"]:
        tar_path = os.path.join(cfg.embeddings_output_dir, mode)

        if os.path.exists(tar_path):
            tar_size = sum(
                os.path.getsize(os.path.join(tar_path, f))
                for f in os.listdir(tar_path)
                if f.endswith(".tar.gz")
            )
            wandb.summary[f"dataset/{mode}/tar_size_bytes"] = tar_size
            wandb.summary[f"dataset/{mode}/n_shards"] = len(os.listdir(tar_path))


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
