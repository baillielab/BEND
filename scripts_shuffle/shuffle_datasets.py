import numpy as np
import pandas as pd
import h5py
import hydra
import os
from bend.utils.set_seed import set_seed, SEED
from omegaconf import DictConfig

set_seed()


@hydra.main(config_path="../conf/embedding/", config_name="shuffle", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Shuffle the dataset specified in the configuration.
    This function is called by hydra.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """

    print("Shuffling data for", cfg.task)

    default_bed = cfg[cfg.task].bed
    default_h5 = cfg[cfg.task].hdf5_file if cfg[cfg.task].hdf5_file else None

    df = pd.read_csv(default_bed, sep="\t", low_memory=False)
    df = df.sample(frac=1, random_state=SEED, replace=False)

    shuffled_bed = default_bed.replace(".bed", "_shuffled.bed")
    df.to_csv(shuffled_bed, sep="\t", index=False)
    print(f"Shuffled BED file saved to: {shuffled_bed}")

    if default_h5 is not None:
        idxs = df.index.to_numpy()

        with h5py.File(default_h5, "r") as f:
            labels_data = np.array(f["labels"])
            labels_data = labels_data[idxs]

        shuffled_h5 = default_h5.replace(".hdf5", "_shuffled.hdf5")
        with h5py.File(shuffled_h5, "w") as f:
            f.create_dataset("labels", data=labels_data)
        print(f"Shuffled HDF5 file saved to: {shuffled_h5}")


if __name__ == "__main__":
    main()
