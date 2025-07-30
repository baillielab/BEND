import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import bend.io.sequtils as sequtils
import pandas as pd
import numpy as np
import sys
from bend_batch.utils import set_seed, record_embedding_time
import webdataset as wds
from bend_batch.datasets import DataSupervised, DEFAULT_SPLIT_COLUMN_IDX, collate_fn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time


set_seed()


@hydra.main(config_path="../config/", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    """
    Run a embedding of nucleotide sequences.
    This function is called by hydra.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    print(f"== Run embedding for task: {cfg.task} with model: {cfg.embedder} ===")

    cfg.embeddings_output_dir = os.path.join(
        cfg.embeddings_output_dir, cfg.task, cfg.embedder
    )
    os.makedirs(cfg.embeddings_output_dir, exist_ok=True)

    embedder = hydra.utils.instantiate(cfg.embedding[cfg.embedder])

    print("Retrieving splits from annotations...")
    annotations = pd.read_csv(
        cfg.tasks[cfg.task].dataset.annotations_path, sep="\t", low_memory=False
    )
    splits = annotations.iloc[:, DEFAULT_SPLIT_COLUMN_IDX].unique()

    start_time = time.time()
    for split in splits:

        print(f"=== Processing split: {split} ===")

        print("Loading dataset ...")
        dataset = DataSupervised(
            annotations_path=cfg.tasks[cfg.task].dataset.annotations_path,
            genome_path=cfg.tasks[cfg.task].dataset.genome_path,
            label_depth=(
                cfg.tasks[cfg.task].dataset.label_depth
                if "label_depth" in cfg.tasks[cfg.task].dataset
                else None
            ),
            hdf5_path=(
                cfg.tasks[cfg.task].dataset.hdf5_path
                if "hdf5_path" in cfg.tasks[cfg.task].dataset
                else None
            ),
            sequence_length=cfg.tasks[cfg.task].dataset.sequence_length,
            split=split,
        )

        is_data_uneven = (
            True if cfg.tasks[cfg.task].dataset.sequence_length is None else False
        )

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.tasks[cfg.task].dataloader.batch_size,
            num_workers=cfg.tasks[cfg.task].dataloader.num_workers,
            shuffle=True if split == "train" else False,
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
                    sample_key = (
                        batch_idx * cfg.tasks[cfg.task].dataloader.batch_size
                        + sample_idx
                    )
                    writer.write(
                        {
                            "__key__": f"sample{sample_key:08d}",
                            "input.npy": embeddings[sample_idx],
                            "output.npy": np.array(labels[sample_idx], dtype=np.int32),
                        }
                    )

    record_embedding_time(cfg, start_time)


if __name__ == "__main__":

    print("Run Embedding")

    run_experiment()
