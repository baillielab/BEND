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
import torch
from bend_batch.models.dilated_cnn import OneHotEmbedding


class TransposeLayer(nn.Module):
    """A layer that transposes the input."""

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        """
        Transpose the input.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transposed tensor.
        """
        x = torch.transpose(x, 1, 2)
        return x


class UpsampleLayer(nn.Module):
    """
    A layer that upsamples the input along the sequence dimension.
    This is useful when a position in the input sequence corresponds to
    multiple positions in the output sequence. The one-to-n mapping
    needs to be a fixed factor.
    """

    def __init__(self, scale_factor=6, input_size=2560):
        """
        Build an upsampling layer.

        Parameters
        ----------
        scale_factor: int
            The factor by which to upsample the input.

        input_size: int
            The embedding size of the input sequence.
        """
        super(UpsampleLayer, self).__init__()
        self.scale_factor = scale_factor
        self.input_size = input_size

        self.upsample = nn.Sequential(
            TransposeLayer(),
            nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=False),
            TransposeLayer(),
        )

    def forward(self, x):
        """
        Upsample the input.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor. Should have shape (batch_size, length, embedding_size).

        Returns
        -------
        torch.Tensor
            Upsampled tensor. Has shape (batch_size, length * scale_factor, embedding_size).
        """
        x = self.upsample(x)
        return x  # torch.reshape(x, (x.shape[0], -1, self.input_size))


class CNNPooling(nn.Module):
    """
    A two-layer CNN with step size 1, ReLU activation, and a linear layer.
    """

    def __init__(
        self,
        input_size=5,
        output_size=2,
        hidden_size=64,
        kernel_size=3,
        upsample_factor: Union[bool, int] = False,
        output_downsample_window=None,
        encoder=None,
        *args,
        **kwargs,
    ):
        """
        Build a two-layer CNN with step size 1, ReLU activation, and a linear layer.

        Parameters
        ----------
        input_size: int
            The embedding size of the input sequence.
        output_size: int
            The size of the output sequence.
        hidden_size: int
            The embedding size of the hidden layer.
        kernel_size: int
            The kernel size of the convolutional layers.
        upsample_factor: int
            The factor by which to upsample the input.
        output_downsample_window: int
            The window size for downsampling the output along the sequence dimension.
            This is done by taking the average of the output values in the window.
        """
        super(CNNPooling, self).__init__()
        self.encoder = encoder
        self.output_size = output_size
        self.onehot_embedding = OneHotEmbedding(input_size)
        if upsample_factor:
            self.upsample = UpsampleLayer(scale_factor=upsample_factor)

        self.conv1 = nn.Sequential(
            TransposeLayer(),
            nn.Conv1d(input_size, hidden_size, kernel_size, stride=1, padding=1),
            TransposeLayer(),
            nn.GELU(),
        )

        self.conv2 = nn.Sequential(
            TransposeLayer(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1, padding=1),
            TransposeLayer(),
            nn.GELU(),
        )

        self.downsample = (
            nn.Sequential(
                TransposeLayer(),
                nn.AvgPool1d(
                    kernel_size=output_downsample_window,
                    stride=output_downsample_window,
                ),
                TransposeLayer(),
            )
            if output_downsample_window is not None
            else None
        )

    def forward(self, x, **kwargs):
        """
        Forward pass of the CNN.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor. Should have shape (batch_size, length, embedding_size).
        length: int
            The actual length (in nucleotides) of the input sequence. Only required when embedding upsampling is used.
        Returns
        -------
        torch.Tensor
            Output tensor. Has shape (batch_size, output_length, output_size).
            output_length is determined by the input length, the upsampling factor, and the output downsampling window.

        """
        length = kwargs.pop("length", None)

        x = self.onehot_embedding(x)
        if hasattr(self, "upsample"):
            x = self.upsample(x)[:, :length]
        if self.encoder is not None:
            x = self.encoder(input_ids=x, **kwargs).last_hidden_state

        # 1st conv layer
        x = self.conv1(x)
        # 2nd conv layer
        x = self.conv2(x)

        # print(f"After conv layers shape: {x.shape}")

        if self.downsample is not None:
            x = self.downsample(x)
            # print(f"After downsample shape: {x.shape}")

        if self.output_size == 1 and x.dim() > 2 or self.downsample:
            x = torch.flatten(x, 1)
            # print(f"After flatten shape: {x.shape}")

        return x


class Classifier(nn.Module):
    """A single linear layer classifier model with configurable pooling."""

    def __init__(
        self,
        input_size=5,
        output_size=2,
        hidden_size=64,
        kernel_size=3,
        upsample_factor: Union[bool, int] = False,
        encoder=None,
        output_downsample_window=None,
        *args,
        **kwargs,
    ):

        super(Classifier, self).__init__()

        self.pooling = CNNPooling(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            upsample_factor=upsample_factor,
            output_downsample_window=output_downsample_window,
            encoder=encoder,
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
