"""
Wrapper classes for embedding sequences with pretrained DNA language models using a common interface.
The wrapper classes handle loading the models and tokenizers, and embedding even or uneven length sequences.
As far as possible, models are downloaded automatically.
They also handle removal of special tokens, and optionally upsample the embeddings to the original sequence length.

- BaseEmbedder: Base class for all embedders.
- NucleotideTransformerEmbedder: Embed using the Nucleotide Transformer (NT) model
- AWDLSTMEmbedder: Embed using the AWD-LSTM model
- ConvNetEmbedder: Embed using the Dilated CNN model
- DNABert2Embedder: Embed using the DNABert2 model
- HyenaDNAModel: Embed using the Hyena-DNA model
"""

import os
from typing import List

import numpy as np
import torch
from transformers import logging

from bend_hybrid.embedding.pooling import PoolingMode
from bend_hybrid.utils import get_device

logging.set_verbosity_error()
DEVICE = get_device()

# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BaseEmbedder:
    """Base class for embedders.
    All embedders should inherit from this class.
    """

    def __init__(
        self, autoregressive, max_sequence_length, upsample_embeddings, *args, **kwargs
    ):
        """Initialize the embedder. Calls `load_model` with the given arguments.

        Parameters
        ----------
        *args
            Positional arguments. Passed to `load_model`.
        **kwargs
            Keyword arguments. Passed to `load_model`.
        """
        self.autoregressive = autoregressive
        self.max_tokens = max_sequence_length
        self.upsample_embeddings = upsample_embeddings

        self.tokenizer = None
        self.model = None

        self.load_model(*args, **kwargs)

        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer is not initialized. Please check the `load_model` method."
            )
        if self.model is None:
            raise ValueError(
                "Model is not initialized. Please check the `load_model` method."
            )

        self.max_tokens = (
            self.max_tokens - 1
            if self.tokenizer.eos_token_id is not None
            else self.max_tokens
        )
        self.max_tokens = (
            self.max_tokens - 1
            if self.tokenizer.cls_token_id is not None
            else self.max_tokens
        )

    def load_model(self, *args, **kwargs):
        """Load the model and tokenizer. Should be implemented by the inheriting class."""
        raise NotImplementedError

    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_ids: list[int] = None,
        **kwargs,
    ):
        # Embed the input sequences, get hidden layers

        # If chunking was used:
        # - concatenate chunks back into full sequences
        # - remove in-between special tokens and padding
        # - Add new padding -> after concat, different sequences lengths

        # Set pad embeddings to nan
        # log hidden layers metrics

        # Return last hidden layer
        raise NotImplementedError

    def _split_tokens_into_chunks(
        self,
        input_ids: List[int],
    ) -> List[List[int]]:
        """Split input_ids into chunks of max_model_length, adding special tokens as needed."""

        chunk_ids = []
        chunk_inputs = []

        for seq_idx, seq in enumerate(input_ids):
            for input_ids_chunk in [
                seq[i : i + self.max_tokens]
                for i in range(0, len(seq), self.max_tokens)
            ]:

                if self.tokenizer.cls_token_id is not None:
                    input_ids_chunk = [self.tokenizer.cls_token_id] + input_ids_chunk
                if self.tokenizer.eos_token_id is not None:
                    input_ids_chunk = input_ids_chunk + [self.tokenizer.eos_token_id]

                chunk_inputs.append(torch.tensor(input_ids_chunk))
                chunk_ids.append(seq_idx)

        chunked_input = torch.nn.utils.rnn.pad_sequence(
            chunk_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        return chunked_input, np.array(chunk_ids)

    def _concatenate_embeddings_chunks(
        self,
        embeddings: List[np.ndarray],
        tokens: np.ndarray,
        chunk_ids: np.ndarray,
    ) -> List[np.ndarray]:
        """Concatenate chunks back into full sequences.

        Parameters
        ----------
        embeddings : List[np.ndarray]
            List of hidden states for each chunk.
        chunk_ids : list[int]
            List of chunk ids indicating which chunk belongs to which sequence.

        Returns
        -------
        List[np.ndarray]
            List of concatenated hidden states for each layer.
        """

        unique_ids = sorted(np.unique(chunk_ids))

        all_concat_tokens = []
        all_concat_emb = []

        for chunk_id in unique_ids:

            concat_emb = np.concatenate(embeddings[chunk_ids == chunk_id], axis=0)
            concat_tokens = np.concatenate(tokens[chunk_ids == chunk_id], axis=0)

            # remove padding tokens
            mask_pad = concat_tokens != self.tokenizer.pad_token_id
            concat_emb = concat_emb[mask_pad]
            concat_tokens = concat_tokens[mask_pad]

            # average eos/cls embeddings if multiple chunks
            cls_emb = None
            if self.tokenizer.cls_token_id is not None:
                cls_emb = concat_emb[concat_tokens == self.tokenizer.cls_token_id].mean(
                    axis=0, keepdims=True
                )

            eos_emb = None
            if self.tokenizer.eos_token_id is not None:
                eos_emb = concat_emb[concat_tokens == self.tokenizer.eos_token_id].mean(
                    axis=0, keepdims=True
                )

            # remove in-between special tokens from embeddings and tokens
            mask_special = (concat_tokens != self.tokenizer.eos_token_id) & (
                concat_tokens != self.tokenizer.cls_token_id
            )
            concat_emb = concat_emb[mask_special]
            if cls_emb is not None:
                concat_emb = np.concatenate([cls_emb, concat_emb], axis=0)
            if eos_emb is not None:
                concat_emb = np.concatenate([concat_emb, eos_emb], axis=0)
            all_concat_emb.append(concat_emb)

            if self.tokenizer.cls_token_id is not None:
                mask_special[0] = True
            if self.tokenizer.eos_token_id is not None:
                mask_special[-1] = True
            concat_tokens = concat_tokens[mask_special]
            all_concat_tokens.append(concat_tokens)

        return all_concat_emb, all_concat_tokens

    def __call__(
        self,
        sequence: List[str],
        sequence_length: List[int],
        pooling: List[PoolingMode],
    ):
        """Embed a list of sequences. Calls `embed` with the given arguments.

        Parameters
        ----------
        sequence : List[str]
            The sequences to embed.
        pooling : List[PoolingMode]
            The pooling modes to use for the embeddings.
        Returns
        -------
        torch.Tensor
            The embeddings of the sequences.
        """

        return self.embed(
            sequence,
            sequence_length,
            pooling,
        )

    def _remove_cls_eos_embeddings(
        self, embeddings: np.ndarray | list[np.ndarray]
    ) -> np.ndarray:
        """
        Remove CLS and EOS embeddings from the input embeddings.
        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings to process.
        Returns
        -------
        np.ndarray
            Embeddings with CLS and EOS tokens removed.
        """

        if isinstance(embeddings, list):
            new_embeddings = []
            for emb in embeddings:
                if self.tokenizer.cls_token_id is not None:
                    emb = emb[1:, :]
                if self.tokenizer.eos_token_id is not None:
                    emb = emb[:-1, :]
                new_embeddings.append(emb)
            return new_embeddings

        if self.tokenizer.cls_token_id is not None:
            embeddings = embeddings[:, 1:, :]
        if self.tokenizer.eos_token_id is not None:
            embeddings = embeddings[:, :-1, :]

        return embeddings
