"""
downstream.py
====================================
This module contains the implementations of the supervised models used in the paper.

- :class:`~bend.models.downstream.ConvNetForSupervised`: a ResNet that we train as baseline model on one-hot encodings, if no dedicated baseline architecture is available for a task.
- :class:`~bend.models.downstream.CNN`: a two-layer CNN used for all downstream tasks.
"""

from typing import Union

import numpy as np
import torch.nn as nn

from bend_hybrid.downstream.pooling.default import DefaultPooling
from bend_hybrid.downstream.pooling.swe import SWE_Pooling
from bend_hybrid.downstream.pooling.upstream import UpstreamPooling


class Classifier(nn.Module):
    """A single linear layer classifier model with configurable pooling."""

    def __init__(
        self,
        input_size=5,
        output_size=2,
        hidden_size=64,
        pooling=None,
        kernel_size=3,
        upsample_factor: Union[bool, int] = False,
        encoder=None,
        output_downsample_window=None,
        freeze_swe=False,
        num_ref_points=512,
        *args,
        **kwargs,
    ):

        super(Classifier, self).__init__()

        match pooling:
            case "default" | None:
                self.pooling = DefaultPooling(
                    input_size=input_size,
                    output_size=output_size,
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    upsample_factor=upsample_factor,
                    output_downsample_window=output_downsample_window,
                    encoder=encoder,
                )
            case "swe":
                self.pooling = SWE_Pooling(
                    d_in=input_size,
                    num_slices=hidden_size,
                    num_ref_points=num_ref_points,
                    freeze_swe=freeze_swe,
                )

            case _:
                self.pooling = UpstreamPooling(
                    input_size=input_size, hidden_size=hidden_size
                )

        self.linear = nn.Sequential(
            nn.Linear(
                hidden_size,
                np.prod(output_size) if isinstance(output_size, tuple) else output_size,
            )
        )

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, activation="none", **kwargs):
        """Forward pass of the classifier."""

        # print(f"Input shape: {x.shape}")
        x = self.pooling(x, **kwargs)
        # print(f"After pooling shape: {x.shape}")

        x = self.linear(x)
        if activation == "softmax":
            x = self.softmax(x)
        elif activation == "softplus":
            x = self.softplus(x)
        elif activation == "sigmoid":
            x = self.sigmoid(x)

        return x
