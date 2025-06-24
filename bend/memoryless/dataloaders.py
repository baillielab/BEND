from torch.utils.data import DataLoader
from bend.memoryless.datasets import DatasetAnnotations
from bend.utils.set_seed import seed_worker
import numpy as np


def get_dataloaders(
    annotations_path,
    reference_path,
    label_depth,
    batch_size=32,
    num_workers=0,
    splits=["train", "valid", "test"],
    **kwargs
):
    dataloaders = []

    for split in splits:

        dataset = DatasetAnnotations(
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
                worker_init_fn=seed_worker,
            )
        )

    return dataloaders
