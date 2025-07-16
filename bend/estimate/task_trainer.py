"""
task_trainer.py
===============
Trainer class for training downstream models on supervised tasks.
"""

import torch
import torch.nn as nn
import os
import pandas as pd
from typing import Union, List
import numpy as np
import glob
import pandas as pd
from bend.utils.task_trainer import BaseTrainer
import time


class EstimateTrainer(BaseTrainer):
    """'Performs training steps for a given model and dataset.
    We use hydra to configure the trainer. The configuration is passed to the
    trainer as an OmegaConf object.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        config,
        overwrite_dir=False,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Get a BaseTrainer object that can be used to train a model.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.
        optimizer : torch.optim.Optimizer
            Optimizer to use for training.
        criterion : torch.nn.Module
            Loss function to use for training.
        device : torch.device
            Device to use for training.
        config : OmegaConf
            Configuration object.
        overwrite_dir : bool, optional
            Whether to overwrite the output directory. The default is False.
        gradient_accumulation_steps : int, optional
            Number of gradient accumulation steps. The default is 1.
        """

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.overwrite_dir = overwrite_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=True
        )  # init scaler for mixed precision training

    def _log_stats(self, epoch, start_time):
        csv_file = f"{self.config.output_dir}/downstream_stats.csv"
        if not os.path.exists(csv_file):
            os.makedirs(self.config.output_dir, exist_ok=True)
            pd.DataFrame(
                columns=["task", "model", "epoch", "time"],
            ).to_csv(csv_file, index=False)

        df = pd.read_csv(f"{self.config.output_dir}/downstream_stats.csv")
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        [
                            self.config.task,
                            self.config.embedder,
                            epoch,
                            time.time() - start_time,
                        ]
                    ],
                    columns=[
                        "task",
                        "model",
                        "epoch",
                        "time",
                    ],
                ),
            ],
            ignore_index=True,
        )
        df.to_csv(f"{self.config.output_dir}/downstream_stats.csv", index=False)
        return

    def train_epoch(self, train_loader):  # one epoch
        """
        Performs one epoch of training.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The training data loader.

        Returns
        -------
        train_loss : float
            The average training loss for the epoch.
        """
        from tqdm.auto import tqdm

        self.model.train()

        train_loss = 0
        # with torch.profiler.profile(schedule=torch.profiler.schedule(wait=10, warmup=2, active=10, repeat=1),
        #                            profile_memory=True,with_stack=True,
        #                            record_shapes=True,
        #                            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/fullwds')) as prof:

        for idx, batch in tqdm(enumerate(train_loader)):
            # with torch.profiler.record_function('h2d copy'):
            train_loss += self.train_step(batch, idx=idx)
            # prof.step()

        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        train_loss /= idx + 1

        return train_loss

    def train(
        self,
        train_loader,
        val_loader,
        test_loader,
        epochs,
        load_checkpoint: Union[bool, int] = True,
    ):
        """
        Performs the full training routine.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The training data loader.
        val_loader : torch.utils.data.DataLoader
            The validation data loader.
            epochs : int
            The number of epochs to train for.
        load_checkpoint : bool, optional
            If True, loads the latest checkpoint from the output directory and
            continues training. If an integer is provided, loads the checkpoint
            from that epoch and continues training.

        Returns
        -------
        None
        """
        print("Training")
        # if load checkpoint is true, then load latest model and continue training
        start_epoch = 0

        for epoch in range(1 + start_epoch, epochs + 1):

            print(f"Epoch {epoch}/{epochs}")
            start_time = time.time()
            train_loss = self.train_epoch(train_loader)

            self._log_stats(epoch, start_time)
        return

    def train_step(self, batch, idx=0):
        """
        Performs a single training step.

        Parameters
        ----------
        batch : tuple
            A tuple containing the batch of data and labels, as returned by the
            data loader.
        idx : int
            The index of the batch.

        Returns
        -------
        loss : float
            The loss for the batch.
        """
        self.model.train()

        data, target = batch
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = self.model(
                x=data.to(self.device, non_blocking=True),
                length=target.shape[-1],
                activation=self.config.params.activation,
            )

            if self.device == torch.device("mps"):
                target = target.to(
                    self.device, non_blocking=True, dtype=torch.float32
                ).long()
            else:
                target = target.to(self.device, non_blocking=True).long()

            loss = self.criterion(output, target)

            loss = loss / self.gradient_accumulation_steps
            # Accumulates scaled gradients.
            self.scaler.scale(loss).backward()
            if (
                idx + 1
            ) % self.gradient_accumulation_steps == 0:  # or (idx + 1 == len_dataloader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

        return loss.item()
