import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import bend.io.sequtils as sequtils
import pandas as pd
import numpy as np
import sys
from bend.utils.set_seed import set_seed, SEED
import time

set_seed()


def record_embedding_time(cfg, start_time: float):
    """
    Record the time taken for embedding in a CSV file.
    Parameters
    ----------
    start_time : float
        The start time of the embedding process.
    """

    end_time = time.time()
    print(f"Embedding completed in {end_time - start_time:.2f} seconds")

    file_path = os.path.join(cfg.output_dir, "embedding_times.csv")

    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data = data._append(
            {
                "task": cfg.task,
                "model": cfg.model,
                "time": end_time - start_time,
                "n_samples": cfg.chunk_size,
            },
            ignore_index=True,
        )
        data.to_csv(file_path, index=False)
    else:
        os.makedirs(cfg.output_dir, exist_ok=True)
        pd.DataFrame(
            {
                "task": [cfg.task],
                "model": [cfg.model],
                "time": [end_time - start_time],
                "n_samples": cfg.chunk_size,
            }
        ).to_csv(file_path, index=False)


# load config
@hydra.main(config_path="../conf/embedding/", config_name="embed", version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    """
    Run a embedding of nucleotide sequences.
    This function is called by hydra.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    print("Embedding data for", cfg.task)

    # read the bed file and get the splits :
    if not "splits" in cfg or cfg.splits is None:
        splits = sequtils.get_splits(cfg[cfg.task].bed)
    else:
        splits = cfg.splits
    print("Embedding with", cfg.model)
    # instatiante model
    embedder = hydra.utils.instantiate(cfg[cfg.model])

    start_time = time.time()

    for split in splits:
        print(f"Embedding split: {split}")
        output_dir = f"{cfg.output_dir}/{cfg.task}/{cfg.model}/"

        os.makedirs(output_dir, exist_ok=True)

        # embed in chunks
        # get length of bed file and divide by chunk size, if a spcific chunk is not set
        df = pd.read_csv(cfg[cfg.task].bed, sep="\t", low_memory=False)
        df = df[df.iloc[:, -1] == split] if split is not None else df
        possible_chunks = list(range(int(len(df) / cfg.chunk_size) + 1))
        if cfg.chunk is None:
            cfg.chunk = possible_chunks
        cfg.chunk = [cfg.chunk] if isinstance(cfg.chunk, int) else cfg.chunk
        # embed in chunks
        for n, chunk in enumerate(cfg.chunk):
            if chunk not in possible_chunks:
                print(
                    f"{chunk} is not a valid chunk id. {split} chunk ids are {possible_chunks}"
                )
                continue
            print(f"\t Embedding chunk {chunk} ({chunk +1}/{len(possible_chunks)})")
            sequtils.embed_from_bed(
                **cfg[cfg.task],
                embedder=embedder,
                output_path=f"{output_dir}/{split}_{chunk}.tar.gz",
                split=split,
                chunk=chunk,
                chunk_size=cfg.chunk_size,
                upsample_embeddings=(
                    cfg[cfg.model]["upsample_embeddings"]
                    if "upsample_embeddings" in cfg[cfg.model]
                    else False
                ),
            )

    record_embedding_time(cfg, start_time)


if __name__ == "__main__":

    print("Run Embedding")

    run_experiment()
