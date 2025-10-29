"""
Implements the BaseEmbedder interface, allowing embedding of sequences using
the HyenaDNA model (https://arxiv.org/abs/2306.15794).
Outputs a dictionary of pooled embeddings based on the specified pooling modes.
"""

import os
import warnings
from typing import List

import torch
from transformers import logging

from bend_hybrid.embedding.pooling import PoolingMode, pool_name_to_function
from bend_hybrid.models.hyena_dna import CharacterTokenizer, HyenaDNAPreTrainedModel
from bend_hybrid.utils import get_device

from .abstract_class import BaseEmbedder

logging.set_verbosity_error()
DEVICE = get_device()

# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class HyenaDNAEmbedder(BaseEmbedder):
    """Embed using the HyenaDNA model https://arxiv.org/abs/2306.15794"""

    def get_start_end_token_ids(self):
        """
        Get the start and end token IDs for the HyenaDNA model.
        Returns
        -------
        Tuple[int, int]
            The cls and eos token IDs.
        """
        return self.tokenizer.cls_token_id, self.tokenizer.eos_token_id

    def load_model(
        self,
        model_path="pretrained_models/hyenadna/hyenadna-tiny-1k-seqlen",
        **kwargs,
    ):
        """
        Load the HyenaDNA model.

        Parameters
        ----------
        model_path : str, optional
            Path to the model checkpoint. Defaults to 'pretrained_models/hyenadna/hyenadna-tiny-1k-seqlen'.
            If the path does not exist, the model will be downloaded from HuggingFace. Rather than just downloading the model,
            HyenaDNA's `from_pretrained` method relies on cloning the HuggingFace-hosted repository, and using git lfs to download the model.
            This requires git lfs to be installed on your system, and will fail if it is not.
        """

        checkpoint_path, model_name = os.path.split(model_path)

        # you can override with your own backbone config here if you want,
        # otherwise we'll load the HF one in None
        backbone_cfg = None

        is_git_lfs_repo = os.path.exists(".git/hooks/pre-push")
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            checkpoint_path,
            model_name,
            download=not os.path.exists(model_path),
            config=backbone_cfg,
            device=DEVICE,
            use_head=False,
            use_lm_head=False,  # we don't use the LM head for embeddings
            n_classes=2,
        )
        model.eval()

        model.to(DEVICE)
        self.model = model

        # NOTE the git lfs download command will add this,
        # but we actually dont use LFS for BEND itself.
        if not is_git_lfs_repo:
            try:
                os.remove(".git/hooks/pre-push")
            except FileNotFoundError:
                pass

        # create tokenizer - NOTE this adds CLS and SEP tokens when add_special_tokens=False
        self.tokenizer = CharacterTokenizer(
            characters=["A", "C", "G", "T", "N"],  # add DNA characters, N is uncertain
            model_max_length=self.max_length,
            add_special_tokens=False,
            padding_side="right",  # as we are interested in the embeddings, and not in generating sequences, we pad on the right
        )

    def embed(
        self,
        sequences: List[str],
        sequence_length: int = None,
    ):
        """Embeds a list of sequences using the HyenaDNA model.
        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        sequence_length : int, optional
            The length of the sequences. If provided, the model will pad or truncate the sequences to this length.
        Returns
        -------
        torch.Tensor
            The embeddings of the sequences.
        """

        # Tokenize input sequences
        chunk_ids = None
        if sequence_length is None or sequence_length > self.max_length:
            sequences, chunk_ids = self._split_sequences_into_chunks(
                sequences,
                self.max_length,  # Single nucleotide tokenizer -> max tokens = max sequence length
            )

        input_ids = self.tokenizer(
            sequences,
            return_tensors="pt",
        )["input_ids"]
        input_ids = torch.LongTensor(input_ids)

        # Embed the input sequences
        with torch.no_grad():
            embeddings = (
                self.model(input_ids=input_ids.to(DEVICE)).detach().cpu().numpy()
            )

        if chunk_ids is not None:
            input_ids = input_ids.numpy()
            embeddings, _ = self._concatenate_chunks(embeddings, input_ids, chunk_ids)

        output = {}

        if PoolingMode.EOS in self.pooling_modes:
            output[PoolingMode.EOS.value] = pool_name_to_function[PoolingMode.EOS](
                embeddings
            )

        embeddings = self._remove_cls_eos_embeddings(embeddings)

        for mode in self.pooling_modes:
            if mode == PoolingMode.EOS:
                continue
            output[mode.value] = pool_name_to_function[mode](embeddings)

        return output
