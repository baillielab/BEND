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
        self,
        autoregressive,
        max_length,
        upsample_embeddings,
        pooling_modes,
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

        self.start_token_id, self.end_token_id = self.get_start_end_token_ids()

        self.max_length = (
            self.max_length - 1 if self.end_token_id is not None else self.max_length
        )
        self.max_length = (
            self.max_length - 1 if self.start_token_id is not None else self.max_length
        )

        self.pooling_modes = self.filter_pooling_modes(pooling_modes)

    def filter_pooling_modes(
        self, pooling_modes: list[PoolingMode]
    ) -> list[PoolingMode]:
        """Filter the given pooling modes based on the embedder's capabilities.

        Parameters
        ----------
        pooling_modes : list[PoolingMode]
            List of pooling modes to filter.
        Returns
        -------
        list[PoolingMode]
            List of valid pooling modes.
        Raises
        -------
        ValueError
            If no valid pooling modes are found.
        """

        modes = []
        for mode in pooling_modes:
            if (
                (
                    mode is PoolingMode.EOS
                    and (not self.autoregressive or self.start_token_id is None)
                )
                or (mode is PoolingMode.MEAN_UPSAMPLE and not self.upsample_embeddings)
                or (
                    mode is PoolingMode.CLS
                    and (self.autoregressive or self.end_token_id is None)
                )
            ):
                continue
            modes.append(mode)

        if len(modes) == 0:
            raise ValueError("No valid pooling modes available for this embedder.")

        return modes

    def get_pooling_modes(self) -> list[PoolingMode]:
        """Get the valid pooling modes for this embedder.

        Returns
        -------
        list[PoolingMode]
            List of valid pooling modes.
        """

        return self.pooling_modes

    def get_start_end_token_ids(self):
        """Get the start and end token ids. Should be implemented by the inheriting class."""
        raise NotImplementedError

    def load_model(self, *args, **kwargs):
        """Load the model and tokenizer. Should be implemented by the inheriting class."""
        raise NotImplementedError

    def embed(
        self,
        sequence: List[str],
        sequence_length: int = None,
    ):
        """Embed and pools the input sequences. Should be implemented by the inheriting class."""
        raise NotImplementedError

    def __call__(
        self,
        sequence: List[str],
        sequence_length: int = None,
    ):
        """Embed and pools a list of sequences. Calls `embed` with the given arguments.

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
        )

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
            chunk_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        return chunked_input, np.array(chunk_ids)

    def _concatenate_chunks(
        self,
        embeddings: List[np.ndarray],
        tokens: np.ndarray,
        chunk_ids: np.ndarray,
    ) -> List[np.ndarray]:
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
        List[np.ndarray]
            List of concatenated embeddings for each sequence.
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
            if self.start_token_id is not None:
                cls_emb = concat_emb[concat_tokens == self.start_token_id].mean(
                    axis=0, keepdims=True
                )

            eos_emb = None
            if self.end_token_id is not None:
                eos_emb = concat_emb[concat_tokens == self.end_token_id].mean(
                    axis=0, keepdims=True
                )

            # remove in-between special tokens from embeddings and tokens
            # append average cls/eos embeddings at beginning/end
            mask_special = (concat_tokens != self.end_token_id) & (
                concat_tokens != self.start_token_id
            )
            concat_emb = concat_emb[mask_special]
            if cls_emb is not None:
                concat_emb = np.concatenate([cls_emb, concat_emb], axis=0)
            if eos_emb is not None:
                concat_emb = np.concatenate([concat_emb, eos_emb], axis=0)
            all_concat_emb.append(concat_emb)

            if cls_emb is not None:
                mask_special[0] = True
            if eos_emb is not None:
                mask_special[-1] = True
            concat_tokens = concat_tokens[mask_special]
            all_concat_tokens.append(concat_tokens)

        return all_concat_emb, all_concat_tokens

    def _remove_padding(
        self,
        embeddings: np.ndarray,
        attention_mask: np.ndarray,
        input_ids: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Remove padding from embeddings using the attention mask.
        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings to process.
        attention_mask : np.ndarray
            Attention mask indicating non-padded tokens.
        input_ids : np.ndarray, optional
            Input ids corresponding to the embeddings. If provided, will also be returned without padding.
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Embeddings and input ids with padding removed. If input_ids is None, only embeddings are returned.
        """

        if not isinstance(attention_mask, np.ndarray):
            attention_mask = attention_mask.numpy()
        if attention_mask.dtype != bool:
            attention_mask = attention_mask.astype(bool)

        if input_ids is not None:

            masked_embeddings = []
            masked_input_ids = []
            for emb, tokens, mask in zip(embeddings, input_ids, attention_mask):
                masked_embeddings.append(emb[mask])
                masked_input_ids.append(tokens[mask])

            return masked_embeddings, masked_input_ids

        masked_embeddings = []
        for emb, mask in zip(embeddings, attention_mask):
            masked_embeddings.append(emb[mask])
        return masked_embeddings

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
