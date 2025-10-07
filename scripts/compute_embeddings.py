"""
Run a supervised task experiment by embedding DNA sequences, using compute_embeddings(),
and training a downstream model, using train_downstream().
Configuration is set through Hydra in config/config.yaml.
"""

import os

import hydra
import numpy as np
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import wandb
from bend_hybrid.embedding.datasets import collate_fn
from bend_hybrid.embedding.embedders import BaseEmbedder
from bend_hybrid.embedding.pooling import PoolingMode, pool_embeddings
from bend_hybrid.utils import log_time, set_seed, wandb_login

set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


def compute_embeddings(cfg: DictConfig, embedder: BaseEmbedder) -> None:
    """
    Embed all sequences in the dataset and stores them using all pooling modes.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    with log_time("dataset/init_time", log_type="summary"):
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        samples_idx_by_split = dataset.get_samples_idx_by_split()

    wandb.summary["dataset/n_samples"] = len(dataset)

    sample_step = 0
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

        writers = {}
        for mode in wandb.summary["dataset/pooling_modes"]:
            shard_dir = os.path.join(cfg.embeddings_output_dir, mode)
            os.makedirs(shard_dir, exist_ok=True)

            writers[mode] = wds.ShardWriter(
                os.path.join(shard_dir, f"{split}_%06d.tar.gz"),
                verbose=0,
                compress="gz",
            )

        for batch_idx, (sequences, labels) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Embedding {split}"
        ):
            with log_time("dataset/batch_embed_time", step=batch_step):
                embeddings = embedder(sequences, uneven_length=dataset.is_uneven())

            for mode in wandb.summary["dataset/pooling_modes"]:
                mode_sample_step = sample_step

                for sample_idx in tqdm(
                    range(len(embeddings)),
                    desc=f"Writing {mode} batch",
                    leave=False,
                ):
                    sample = pool_embeddings(embeddings[sample_idx], PoolingMode(mode))
                    sample_key = (
                        batch_idx * cfg.task.dataloaders.batch_size + sample_idx
                    )

                    with log_time(
                        f"dataset/{mode}/sample_store_time", step=mode_sample_step
                    ):
                        writers[mode].write(
                            {
                                "__key__": f"sample{sample_key:08d}",
                                "input.npy": sample,
                                "output.npy": np.array(
                                    labels[sample_idx], dtype=np.int32
                                ),
                            }
                        )
                    mode_sample_step += 1

            sample_step += len(embeddings)
            batch_step += 1

        for writer in writers.values():
            writer.close()

    for mode in wandb.summary["dataset/pooling_modes"]:
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
        project="bend-pooling",
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

    wandb.summary["dataset/pooling_modes"] = [
        mode.value
        for mode in PoolingMode
        if not (
            mode == PoolingMode.MEAN_NO_UPSAMPLE and not embedder.upsample_embeddings
        )
    ]

    print(
        f"=== Embedding sequences for task: {cfg.task.name} with model: {cfg.embedder} ==="
    )

    compute_embeddings(cfg, embedder)

    wandb.finish()


if __name__ == "__main__":
    run_experiment()  # pylint: disable=E1120
