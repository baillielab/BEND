"""
Implements the BaseEmbedder interface, allowing embedding of sequences using
the Nucleotide Transformer model (https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full).
Outputs a dictionary of pooled embeddings based on the specified pooling modes.
"""

import os
from typing import List

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging

from bend_batch.utils import get_device

from .abstract_class import BaseEmbedder

logging.set_verbosity_error()
DEVICE = get_device()

# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NucleotideTransformerEmbedder(BaseEmbedder):
    """
    Embed using the Nucleotide Transformer (NT) model (https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full).
    """

    def get_special_tokens_ids(self):
        """
        Get the start and end token IDs for the Nucleotide Transformer model.
        Returns
        -------
        Tuple[int, None]
            The cls token ID and None (no end token).
        """
        return self.tokenizer.cls_token_id, None, self.tokenizer.pad_token_id

    def load_model(
        self,
        model_name,
        **kwargs,
    ):
        """
        Load the Nuclieotide Transformer (NT) model.

        Parameters
        ----------
        model_name : str
            The name of the model to load.
            When providing a name, the model will be loaded from the HuggingFace model hub.
            Alternatively, you can provide a path to a local model directory. We check whether the model_name
            contains 'v2' to determine whether we need to follow the V2 model API or not.
        remove_special_tokens : bool, optional
            Whether to remove the CLS token from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to the length of the input sequence. Defaults to False.
        """

        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, max_length=None
        )  # manually handle max length

        self.is_v2 = True if "v2" in model_name else False

        self.model.to(DEVICE)
        self.model.eval()

    def embed(self, sequences: List[str]):
        """
        Embed sequences using the Nuclieotide Transformer (NT) model.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        Returns
        -------
        torch.Tensor
            The embeddings of the sequences.
        """

        input_ids = self.tokenizer(
            sequences,
            padding=False,
            max_length=None,
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]

        # Iterate over tokens, split if necessary, add special tokens, pad into tensor
        input_ids, chunk_ids = self._split_tokens_into_chunks(input_ids)
        attention_mask = input_ids != self.pad_token_id

        with torch.no_grad():
            embeddings = (
                self.model(
                    input_ids.to(DEVICE),
                    attention_mask=(
                        attention_mask.to(DEVICE)
                        if attention_mask is not None
                        else None
                    ),
                    output_hidden_states=True,
                )["hidden_states"][-1]
                .detach()
                .cpu()
                .numpy()
            )
            input_ids = input_ids.numpy()

        if len(chunk_ids) != len(set(chunk_ids)):
            # concatenate chunks, remove pad and in-between special tokens
            embeddings, input_ids = self._merge_embeddings(
                embeddings, input_ids, chunk_ids
            )
        else:
            embeddings, input_ids = self._remove_special_tokens(embeddings, input_ids)

        if self.upsample_embeddings:
            embeddings = self._upsample(
                input_ids,
                embeddings,
                same_length_sequences=self.task_input_length is not None,
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
        Skips repeating any special tokens, such as CLS, UNK or PAD.

        Parameters
        ----------
        token_ids (np.ndarray):
            The 1D arrays of token IDs.
        embeddings (np.ndarray):
            The embeddings arrays to be upsampled.
        same_length_sequences (bool):
            Whether all sequences have the same length.
        Returns
        -------
            np.ndarray: The upsampled embeddings arrays.
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
                token_ids, skip_special_tokens=True
            )

            repetitions = np.array([len(token) for token in tokens], dtype=np.int64)

            upsampled_embeddings.append(np.repeat(emb, repetitions, axis=0))

        if same_length_sequences:
            upsampled_embeddings = np.stack(upsampled_embeddings)

        return upsampled_embeddings
