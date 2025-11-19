"""
Provides the BaseEmbedder class that all embedders should inherit from.
The BaseEmbedder class implements common functionality for all embedders, such as:
-  _split_tokens_into_chunks: chunking long sequences into smaller pieces
-  _concatenate_chunks: concatenating chunked sequences back into full sequences
-  _remove_padding: removing padding.
-  _remove_cls_eos_embeddings: removing CLS and EOS/SEP token embeddings, if present.
Specific embedders should implement the `get_start_end_token_ids`, `load_model` and `embed` methods.
"""

import os
from typing import List

import numpy as np
import torch
from transformers import logging

from bend_batch.utils import get_device

logging.set_verbosity_error()
DEVICE = get_device()

# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BaseEmbedder:
    """Base class for embedders.
    All embedders should inherit from this class.
    """

    def __init__(
        self,
        autoregressive,
        max_length,
        upsample_embeddings,
        task_input_length=None,
        *args,
        **kwargs,
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
        self.max_length = max_length
        self.upsample_embeddings = upsample_embeddings
        self.task_input_length = task_input_length

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

        self.start_token_id, self.end_token_id, self.pad_token_id = (
            self.get_special_tokens_ids()
        )

        self.max_length = (
            self.max_length - 1 if self.end_token_id is not None else self.max_length
        )
        self.max_length = (
            self.max_length - 1 if self.start_token_id is not None else self.max_length
        )

    def get_special_tokens_ids(self):
        """Get the start, end and pad token ids. Should be implemented by the inheriting class."""
        raise NotImplementedError

    def load_model(self, *args, **kwargs):
        """Load the model and tokenizer. Should be implemented by the inheriting class."""
        raise NotImplementedError

    def embed(self, sequences: List[str]):
        """Embed  the input sequences. Should be implemented by the inheriting class."""
        raise NotImplementedError

    def __call__(self, sequences: List[str]):
        """Embed a list of sequences. Calls `embed` with the given arguments.

        Parameters
        ----------
        sequence : List[str]
            The sequences to embed.
        Returns
        -------
        torch.Tensor
            The embeddings of the sequences.
        """

        return self.embed(sequences)

    def _split_sequences_into_chunks(self, sequences: List[str], max_length: int):
        """Split sequences into chunks of max_length.
        Parameters
        ----------
        sequences : List[str]
            List of sequences to split.
        max_length : int
            Maximum length of each chunk.
        Returns
        -------
        chunk_inputs : List[str]
            List of chunked sequences.
        chunks_ids : List[int]
            List of chunk ids indicating which chunk belongs to which sequence.
        """

        chunks_ids = []
        chunk_inputs = []

        for idx, seq in enumerate(sequences):
            chunks = [seq[i : i + max_length] for i in range(0, len(seq), max_length)]
            chunk_inputs.extend(chunks)
            chunks_ids.extend([idx] * len(chunks))
        return chunk_inputs, chunks_ids

    def _split_tokens_into_chunks(
        self,
        input_ids: List[int],
    ) -> List[List[int]]:
        """Split input_ids into chunks of max_model_length, adding special tokens as needed."""

        chunk_ids = []
        chunk_inputs = []

        for seq_idx, seq in enumerate(input_ids):
            for input_ids_chunk in [
                seq[i : i + self.max_length]
                for i in range(0, len(seq), self.max_length)
            ]:

                if self.start_token_id is not None:
                    input_ids_chunk = [self.start_token_id] + input_ids_chunk
                if self.end_token_id is not None:
                    input_ids_chunk = input_ids_chunk + [self.end_token_id]

                chunk_inputs.append(torch.tensor(input_ids_chunk))
                chunk_ids.append(seq_idx)

        chunked_input = torch.nn.utils.rnn.pad_sequence(
            chunk_inputs, batch_first=True, padding_value=self.pad_token_id
        )

        return chunked_input, np.array(chunk_ids)

    def _merge_embeddings(
        self,
        embeddings: List[np.ndarray],
        tokens: np.ndarray,
        chunk_ids: np.ndarray,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Concatenate chunks back into full sequences.

        Parameters
        ----------
        embeddings : List[np.ndarray]
            List of embeddings for each chunk.
        tokens : np.ndarray
            Array of token ids for each chunk.
        chunk_ids : list[int]
            List of chunk ids indicating which chunk belongs to which sequence.
        Returns
        -------
        Tuple[List[np.ndarray],List[np.ndarray]]
            List of concatenated embeddings for each sequence.
        """

        unique_ids = sorted(np.unique(chunk_ids))

        all_concat_tokens = []
        all_concat_emb = []

        for chunk_id in unique_ids:

            concat_emb = np.concatenate(embeddings[chunk_ids == chunk_id], axis=0)
            concat_tokens = np.concatenate(tokens[chunk_ids == chunk_id], axis=0)

            mask_special = (
                (concat_tokens != self.pad_token_id)
                & (concat_tokens != self.end_token_id)
                & (concat_tokens != self.start_token_id)
            )

            all_concat_emb.append(concat_emb[mask_special])
            all_concat_tokens.append(concat_tokens[mask_special])

        return all_concat_emb, all_concat_tokens

    def _remove_padding_tokens(
        self,
        embeddings: np.ndarray,
        input_ids: np.ndarray,
    ) -> tuple[List[np.ndarray], List[np.ndarray]] | List[np.ndarray]:
        """Remove padding embeddings and padding tokens.
        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings to process.
        input_ids : np.ndarray
            Input ids corresponding to the embeddings.
        Returns
        -------
        tuple[List[np.array], List[np.array]]
            Embeddings and input ids with padding removed. If input_ids is None, only embeddings are returned.
        """

        masked_embeddings = []
        masked_input_ids = []
        for emb, tokens_ids in zip(embeddings, input_ids):

            mask = tokens_ids != self.pad_token_id
            masked_embeddings.append(emb[mask])
            masked_input_ids.append(tokens_ids[mask])

        return masked_embeddings, masked_input_ids

    def _remove_special_tokens(self, embeddings, input_ids):
        all_embeddings = []
        all_tokens = []

        for emb, tokens in zip(embeddings, input_ids):
            mask_special = (
                (tokens != self.pad_token_id)
                & (tokens != self.end_token_id)
                & (tokens != self.start_token_id)
            )

            all_embeddings.append(emb[mask_special])
            all_tokens.append(tokens[mask_special])

        return all_embeddings, all_tokens

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
                if self.start_token_id is not None:
                    emb = emb[1:, :]
                if self.end_token_id is not None:
                    emb = emb[:-1, :]
                new_embeddings.append(emb)
            return new_embeddings

        if self.start_token_id is not None:
            embeddings = embeddings[:, 1:, :]
        if self.end_token_id is not None:
            embeddings = embeddings[:, :-1, :]

        return embeddings
