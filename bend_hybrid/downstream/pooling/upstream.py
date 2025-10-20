import torch.nn as nn


class UpstreamPooling(nn.Module):
    """A single linear layer for resizing to the hidden size the already pooled embeddings."""

    def __init__(
        self,
        input_size=5,
        hidden_size=64,
        *args,
        **kwargs,
    ):
        """
        Build a two-layer CNN with step size 1, ReLU activation, and a linear layer.

        Parameters
        ----------
        input_size: int
            The embedding size of the input sequence.
        hidden_size: int
            The embedding size of the hidden layer.
        kernel_size: int
            The kernel size of the convolutional layers.
        """
        super(UpstreamPooling, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
        )

    def forward(self, x, **kwargs):
        """
        Forward pass of the CNN.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor. Should have shape (batch_size, length, embedding_size).
        Returns
        -------
        torch.Tensor
            Output tensor. Has shape (batch_size, output_length, output_size).
            output_length is determined by the input length, the upsampling factor, and the output downsampling window.

        """
        # print(f"Input shape: {x.shape}")

        # # if input is 2D (batch_size, embedding_size), add sequence dimension -> (batch_size, 1, embedding_size)
        # if x.ndim == 2:
        #     x = torch.unsqueeze(x, dim=1)
        #     # print(f"After unsqueeze shape: {x.shape}")

        x = self.linear(x)
        # print(f"After linear shape: {x.shape}")

        return x
