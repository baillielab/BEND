import os
import sys
import time

import hydra
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import bend.io.sequtils as sequtils
from bend_batch.datasets import DEFAULT_SPLIT_COLUMN_IDX, DataSupervised, collate_fn
from bend_batch.utils import record_embedding_time, set_seed

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
    print(f"== Run embedding for task: {cfg.task.task} with model: {cfg.embedder} ===")

    os.makedirs(cfg.embeddings_output_dir, exist_ok=True)

    embedder = hydra.utils.instantiate(cfg.embedding[cfg.embedder])

    print("Retrieving splits from annotations...")
    annotations = pd.read_csv(
        cfg.task.dataset.annotations_path, sep="\t", low_memory=False
    )
    splits = annotations.iloc[:, DEFAULT_SPLIT_COLUMN_IDX].unique()

    start_time = time.time()
    for split in splits:

        print(f"=== Processing split: {split} ===")

        print("Loading dataset ...")
        dataset = DataSupervised(
            annotations_path=cfg.task.dataset.annotations_path,
            genome_path=cfg.task.dataset.genome_path,
            label_depth=(
                cfg.task.dataset.label_depth
                if "label_depth" in cfg.task.dataset
                else None
            ),
            hdf5_path=(
                cfg.task.dataset.hdf5_path if "hdf5_path" in cfg.task.dataset else None
            ),
            sequence_length=cfg.task.dataset.sequence_length,
            split=split,
        )

        is_data_uneven = True if cfg.task.dataset.sequence_length is None else False

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.task.dataloader.batch_size,
            num_workers=cfg.task.dataloader.num_workers,
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
                    sample_key = batch_idx * cfg.task.dataloader.batch_size + sample_idx
                    writer.write(
                        {
                            "__key__": f"sample{sample_key:08d}",
                            "input.npy": embeddings[sample_idx],
                            "output.npy": np.array(labels[sample_idx], dtype=np.int32),
                        }
                    )

    tar_size = sum(
        os.path.getsize(f)
        for f in os.listdir(cfg.embeddings_output_dir)
        if f.endswith(".tar.gz")
    )
    record_embedding_time(
        cfg.task.task,
        cfg.embedder,
        start_time,
        len(annotations),
        tar_size,
        cfg.output_dir,
    )


if __name__ == "__main__":

    print("Run Embedding")

    run_experiment()
