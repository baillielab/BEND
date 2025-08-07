import glob
import os
import time

import hydra
import numpy as np
import pandas as pd
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from run_supervised_tasks import embed, train_on_task
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from bend.utils.task_trainer import BaseTrainer
from bend_hybrid.datasets import DEFAULT_SPLIT_COLUMN_IDX, DataSupervised, collate_fn
from bend_hybrid.utils import get_device, record_embedding_time, set_seed

set_seed()
os.environ["WDS_VERBOSE_CACHE"] = "1"


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

    print("Retrieving splits from annotations...")
    annotations = pd.read_csv(
        cfg.task.dataset.annotations_path, sep="\t", low_memory=False
    )
    splits = annotations.iloc[:, DEFAULT_SPLIT_COLUMN_IDX].unique().tolist()

    if cfg.compute_embeddings is True:

        existing_shards = [
            shard
            for shard in glob.glob(os.path.join(cfg.embeddings_output_dir, "*.tar.gz"))
        ]

        # Only embed train split if valid and test are already embedded
        if any("test" in shard for shard in existing_shards) and any(
            "valid" in shard for shard in existing_shards
        ):
            print("Valid and Test splits already embedded, skipping them.")
            splits = ["train"]

        # remove existing train shards
        if any("train" in shard for shard in existing_shards):
            train_shards = [shard for shard in existing_shards if "train" in shard]
            for shard in train_shards:
                os.remove(shard)

        print(
            f"=== Embedding sequences for task: {cfg.task.task} with model: {cfg.embedder} ==="
        )

        os.makedirs(cfg.embeddings_output_dir, exist_ok=True)
        embedder = hydra.utils.instantiate(cfg.embedding[cfg.embedder])

        end_time = start_time = time.time()

        for split in splits:
            if split == "train":
                start_time = time.time()

            print(f"=== Processing split: {split} ===")
            embed(cfg, embedder, split)

            if split == "train":
                end_time = time.time()

        record_embedding_time(
            cfg.task.task,
            cfg.embedder,
            running_time=end_time - start_time,
            n_samples=len(annotations),
            tar_path=cfg.embeddings_output_dir,
            output_dir=cfg.output_dir,
        )

    if (
        "cross_validation" in cfg.task.data.keys()
        and cfg.task.data.cross_validation is True
    ):
        output_dir = cfg.output_dir

        for fold in range(len(splits)):
            print(f"=== Running fold {fold + 1}/{len(splits)} ===")
            cfg.task.data.cross_validation = fold
            cfg.output_dir = os.path.join(output_dir, f"split_{fold + 1}")
            train_on_task(cfg)
    else:
        train_on_task(cfg)


if __name__ == "__main__":
    run_experiment()
