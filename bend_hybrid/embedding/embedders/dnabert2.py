"""
Implements the BaseEmbedder interface, allowing embedding of sequences using the DNABERT2 model (https://arxiv.org/pdf/2306.15006.pdf).
Outputs a dictionary of pooled embeddings based on the specified pooling modes.
"""

import os
import warnings
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, logging

from bend_hybrid.embedding.pooling import PoolingMode, pool_name_to_function
from bend_hybrid.models.dnabert2 import BertForMaskedLM as DNABert2BertForMaskedLM
from bend_hybrid.utils import get_device

from .abstract_class import BaseEmbedder

logging.set_verbosity_error()
DEVICE = get_device()

# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DNABert2Embedder(BaseEmbedder):
    """
    Embed using the DNABERT2 model https://arxiv.org/pdf/2306.15006.pdf
    """

    def get_start_end_token_ids(self):
        """
        Get the start and end token IDs for the DNABERT2 model.
        Returns
        -------
        Tuple[int, int]
            The cls and sep token IDs.
        """
        return self.tokenizer.cls_token_id, self.tokenizer.sep_token_id

    def load_model(
        self,
        model_name="zhihan1996/DNABERT-2-117M",
        **kwargs,
    ):
        """
        Load the DNABERT2 model.
        Note that this model uses byte pair encoding (BPE) and upsample_embedding=True repeats BPE token embeddings so that each nucleotide has its own embedding.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to load. Defaults to "zhihan1996/DNABERT-2-117M".
            When providing a name, the model will be loaded from the HuggingFace model hub.
            Alternatively, you can provide a path to a local model directory.
        """

        # keep the source in this repo to avoid using flash attn.
        self.model = DNABert2BertForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.model.to(DEVICE)

    def embed(
        self,
        sequences: List[str],
        pooling: List[PoolingMode] = [PoolingMode.DEFAULT],
        sequence_length: int = None,
    ):
        """Embeds a list sequences using the DNABERT2 model.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        pooling : List[PoolingMode], optional
            List of pooling modes to apply. Defaults to [PoolingMode.DEFAULT].
        sequence_length : int, optional
            The length of the sequences.
        Returns
        -------
        embeddings : List[np.ndarray]
            List of embeddings.
        """

        # BPE tokeniser generates len(tokens)<len(sequence), hence do not check here if sequence length > max_tokens, set max_length=None
        input_ids = self.tokenizer(
            sequences,
            padding=False,
            max_length=None,
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]

        # Iterate over tokens, split if necessary, add special tokens, pad into tensor
        input_ids, chunk_ids = self._split_tokens_into_chunks(input_ids)
        attention_mask = input_ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            embeddings = (
                self.model(
                    input_ids.to(DEVICE),
                    attention_mask=attention_mask.to(DEVICE),
                    encoder_attention_mask=attention_mask.to(DEVICE),
                )["hidden_states"]
                .detach()
                .cpu()
                .numpy()
            )  # n_chunks or batch_size x seq_len x emb_dim
            input_ids = input_ids.numpy()

        if len(chunk_ids) != len(set(chunk_ids)):
            # concatenate chunks, remove pad and in-between special tokens
            embeddings, input_ids = self._concatenate_chunks(
                embeddings, input_ids, chunk_ids
            )  # batch_size x seq_len x emb_dim
        else:
            embeddings, input_ids = self._remove_padding(
                embeddings, attention_mask.numpy(), input_ids
            )

        # Pooling
        output = {}

        if PoolingMode.CLS in pooling:
            output[PoolingMode.CLS.value] = pool_name_to_function[PoolingMode.CLS](
                embeddings
            )
            pooling.remove(PoolingMode.CLS)

        if PoolingMode.EOS in pooling:
            warnings.warn(
                "EOS pooling is not supported for DNABERT2, as sequences do not have an EOS token."
            )
            pooling.remove(PoolingMode.EOS)

        embeddings = self._remove_cls_eos_embeddings(embeddings)

        if PoolingMode.MEAN_NO_UPSAMPLE in pooling:
            output[PoolingMode.MEAN_NO_UPSAMPLE.value] = pool_name_to_function[
                PoolingMode.MEAN_NO_UPSAMPLE
            ](embeddings)
            pooling.remove(PoolingMode.MEAN_NO_UPSAMPLE)

        if self.upsample_embeddings:
            upsampled_embeddings = []
            for token_ids, emb in zip(input_ids, embeddings):
                upsampled_embeddings.append(self._upsample(token_ids, emb))
            embeddings = np.stack(upsampled_embeddings)

        for mode in pooling:
            output[mode.value] = pool_name_to_function[mode](embeddings)

        return output

    def _upsample(self, token_ids: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        """
        Upsamples the embeddings based on the number of characters in each non-special token.
        CLS and SEP tokens are ignored, and the [UNK] token is repeated once.

        Parameters
        ----------
            token_ids (np.ndarray): The 1D array of token IDs.
            embedding (np.ndarray): The embeddings array to be upsampled.
        Returns
        -------
            np.ndarray: The upsampled embeddings array.
        Raises
        ------
            ValueError: If the tokenizer does not have a method `convert_ids_to_tokens`.
        """

        if not hasattr(self.tokenizer, "convert_ids_to_tokens"):
            raise ValueError(
                "Tokenizer does not have method `convert_ids_to_tokens`, cannot upsample embeddings."
            )

        tokens = self.tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=False
        )

        repetitions = []
        for token in tokens:
            if (
                token == self.tokenizer.cls_token
                or token == self.tokenizer.sep_token
                or token == self.tokenizer.pad_token
            ):
                continue

            if token == self.tokenizer.unk_token:
                repetitions.append(1)
            else:
                repetitions.append(len(token))

        repetitions = np.array(repetitions, dtype=np.int32)

        return np.repeat(embedding, repetitions, axis=0)
