from torch.utils.data import DataLoader
from bend.utils.datasets import DatasetMultiHot
from bend.utils.set_seed import SEED
import numpy as np


def get_cpg_dataloader(
    annotations_path,
    reference_path,
    label_depth,
    batch_size=32,
    num_workers=0,
    **kwargs
):
    dataloaders = []

    for split in ["train", "valid", "test"]:

        dataset = DatasetMultiHot(
            annotations_path=annotations_path,
            genome_path=reference_path,
            label_depth=label_depth,
            split=split,
        )

        dataloaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True if split == "train" else False,
                num_workers=num_workers,
                worker_init_fn=lambda _: np.random.seed(SEED),
            )
        )

    return dataloaders
