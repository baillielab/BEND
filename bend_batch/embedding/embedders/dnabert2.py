"""
Implements the BaseEmbedder interface, allowing embedding of sequences using the DNABERT2 model (https://arxiv.org/pdf/2306.15006.pdf).
Outputs a dictionary of pooled embeddings based on the specified pooling modes.
"""

import os
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, logging

from bend_batch.models.dnabert2 import BertForMaskedLM as DNABert2BertForMaskedLM
from bend_batch.utils import get_device

from .abstract_class import BaseEmbedder

logging.set_verbosity_error()
DEVICE = get_device()

# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DNABert2Embedder(BaseEmbedder):
    """
    Embed using the DNABERT2 model https://arxiv.org/pdf/2306.15006.pdf
    """

    def get_special_tokens_ids(self):
        """
        Get the start, end and pad token IDs for the DNABERT2 model.
        Returns
        -------
        Tuple[int, int]
            The cls and sep token IDs.
        """
        return (
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        )

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
        sequence_length: int = None,
    ):
        """Embeds a list sequences using the DNABERT2 model.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        sequence_length : int, optional
            The length of the sequences.
        Returns
        -------
        embeddings : List[np.ndarray]
            List of embeddings.
        """

        # Tokenize input sequences
        chunk_ids = None
        if sequence_length is None or sequence_length > self.max_length:
            sequences, chunk_ids = self._split_sequences_into_chunks(
                sequences,
                self.max_length,
            )

        output = self.tokenizer(sequences, return_tensors="pt", padding="longest")
        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]

        with torch.no_grad():
            embeddings = (
                self.model(
                    input_ids.to(DEVICE),
                    attention_mask=attention_mask.to(DEVICE),
                    output_hidden_states=True,
                )["hidden_states"]
                .detach()
                .cpu()
                .numpy()
            )
            input_ids = input_ids.numpy()

        if chunk_ids is not None and len(chunk_ids) != len(set(chunk_ids)):
            # concatenate chunks, remove pad, cls and eos tokens
            embeddings, input_ids = self._merge_embeddings(
                embeddings, input_ids, chunk_ids
            )
        else:
            embeddings, input_ids = self._remove_special_tokens(embeddings, input_ids)

        if self.upsample_embeddings:
            embeddings = self._upsample(
                input_ids,
                embeddings,
                same_length_sequences=sequence_length is not None,
            )

        return embeddings

    def _upsample(
        self,
        input_ids: np.ndarray,
        embeddings: np.ndarray,
        same_length_sequences: bool = False,
    ) -> np.ndarray | List[np.ndarray]:
        """
        Upsamples the embeddings based on the number of characters in each non-special token.
        CLS and SEP tokens are ignored, and the [UNK] token is repeated once.

        Parameters
        ----------
        input_ids (np.ndarray):
            The 1D array of token IDs.
        embeddings (np.ndarray):
            The embeddings array to be upsampled.
        same_length_sequences (bool):
            Whether all sequences have the same length.
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

        upsampled_embeddings = []
        for token_ids, emb in zip(input_ids, embeddings):
            tokens = self.tokenizer.convert_ids_to_tokens(
                token_ids, skip_special_tokens=False
            )

            repetitions = []
            for token in tokens:
                if token in [
                    self.tokenizer.cls_token,
                    self.tokenizer.sep_token,
                    self.tokenizer.pad_token,
                ]:
                    continue

                if token == self.tokenizer.unk_token:
                    repetitions.append(1)
                else:
                    repetitions.append(len(token))

            repetitions = np.array(repetitions, dtype=np.int32)

            upsampled_embeddings.append(np.repeat(emb, repetitions, axis=0))

        if same_length_sequences:
            upsampled_embeddings = np.stack(upsampled_embeddings)

        return upsampled_embeddings
