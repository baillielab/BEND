"""
embedders.py
------------
Wrapper classes for embedding sequences with pretrained DNA language models using a common interface.
The wrapper classes handle loading the models and tokenizers, and embedding the sequences. As far as possible,
models are downloaded automatically.
They also handle removal of special tokens, and optionally upsample the embeddings to the original sequence length.

Embedders can be used as follows. Please check the individual classes for more details on the arguments.

``embedder = EmbedderClass(model_name, some_additional_config_argument=6)``

``embedding = embedder(sequence, remove_special_tokens=True, upsample_embeddings=True)``

"""

import torch
import numpy as np
from typing import List, Iterable
from functools import partial
import os

from bend.models.awd_lstm import AWDLSTMModelForInference
from bend.models.dilated_cnn import ConvNetModel
from bend.models.hyena_dna import HyenaDNAPreTrainedModel, CharacterTokenizer
from bend.models.dnabert2 import BertForMaskedLM as DNABert2BertForMaskedLM
from bend.utils.download import download_model
from bend_batch.utils import get_device

from tqdm.auto import tqdm
from transformers import (
    logging,
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from sklearn.preprocessing import LabelEncoder

logging.set_verbosity_error()
DEVICE = get_device()


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
        self.max_sequence_length = max_sequence_length
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

    def load_model(self, *args, **kwargs):
        """Load the model. Should be implemented by the inheriting class."""
        raise NotImplementedError

    def embed(self, sequences: List[str], uneven_length: bool = False, *args, **kwargs):
        """Embed a list of sequences. Should be implemented by the inheriting class.

        Parameters
        ----------
        sequences : List[str]
            The sequences to embed.
        uneven_length : bool
            Whether the sequences have uneven length. If True, the model should handle padding. Defaults to False.
        *args
            Positional arguments. Passed to the model's embedding method.
        **kwargs
            Keyword arguments. Passed to the model's embedding method.

        Returns
        -------
        torch.Tensor
            The embeddings of the sequences.
        """
        raise NotImplementedError

    def __call__(self, sequence: List[str], *args, **kwargs):
        """Embed a list of sequences. Calls `embed` with the given arguments.

        Parameters
        ----------
        sequence : List[str]
            The sequences to embed.
        *args
            Positional arguments. Passed to `embed`.
        **kwargs
            Keyword arguments. Passed to `embed`.

        Returns
        -------
        torch.Tensor
            The embeddings of the sequences.
        """
        return self.embed(sequence, *args, **kwargs)

    def chunkify_sequences(self, sequences: List[str]) -> tuple[List[str], np.ndarray]:
        """
        Chunkify the input sequences into smaller chunks, defined by `self.max_sequence_length`.

        Parameters
        ----------
        sequences : List[str]
            The input sequences to chunk.

        Returns
        -------
        (List[str], np.ndarray)
            A tuple containing the chunked sequences and their corresponding sequence indices.
        """

        chunks = []
        chunk_ids = []

        for seq_idx, seq in enumerate(sequences):
            chunked_sequence = [
                seq[i : i + self.max_sequence_length]
                for i in range(0, len(seq), self.max_sequence_length)
            ]
            chunks.extend(chunked_sequence)
            chunk_ids.extend([seq_idx] * len(chunked_sequence))

        return chunks, np.array(chunk_ids)

    def _upsample(self, token_ids: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        """
        Upsamples the embeddings based on the number of characters in each token.

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
                "Tokenizer does not have a method `convert_ids_to_tokens`. "
                "Please check the tokenizer implementation."
            )

        tokens = self.tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=True
        )
        repetitions = np.array([len(token) for token in tokens])

        return np.repeat(embedding, repetitions, axis=0)

    def _remove_special_tokens(
        self, token_ids: np.ndarray, embedding: np.ndarray
    ) -> np.ndarray:
        """
        Removes special tokens from the embeddings based on the tokenizer's special token mask.

        Parameters
        ----------
            token_ids (np.ndarray): The 1D array of token IDs.
            embedding (np.ndarray): The embeddings array from which to remove special tokens.
        Returns
        -------
            np.ndarray: The embeddings array with special tokens removed.
        Raises
        ------
            ValueError: If the tokenizer does not have a method `get_special_tokens_mask`.
        """

        if not hasattr(self.tokenizer, "get_special_tokens_mask"):
            raise ValueError(
                "Tokenizer does not have a method `get_special_tokens_mask`. "
                "Please check the tokenizer implementation."
            )

        mask_special_tokens = ~np.array(
            self.tokenizer.get_special_tokens_mask(
                token_ids, already_has_special_tokens=True
            ),
            dtype=bool,
        )

        return embedding[mask_special_tokens]

    def process_chunk_embeddings(
        self,
        chunk_embeddings: np.ndarray,
        chunk_input_ids: np.ndarray,
        sequences_ids: np.ndarray,
    ):
        """
        Processes chunk embeddings by removing special tokens and optionally upsampling the embeddings.

        Parameters
        ----------
            chunk_embeddings (np.ndarray): The embeddings for each chunk.
            chunk_input_ids (np.ndarray): The input IDs corresponding to each chunk.
            sequences_ids (np.ndarray): The sequence IDs corresponding to each chunk.
        Returns
        -------
            List[np.ndarray]: A list of processed embeddings for each sequence.
        Raises
        ------
            ValueError: If the tokenizer does not have a method `convert_ids_to_tokens`.
            ValueError: If the tokenizer does not have a method `get_special_tokens_mask`.
        """

        masked_embeddings = []

        for sequence_idx in np.unique(sequences_ids):

            mask_sequence = sequence_idx == sequences_ids
            concat_embeddings = np.concatenate(chunk_embeddings[mask_sequence], axis=0)
            concat_input_ids = np.concatenate(chunk_input_ids[mask_sequence], axis=0)

            concat_embeddings = self._remove_special_tokens(
                concat_input_ids, concat_embeddings
            )

            if self.upsample_embeddings:
                concat_embeddings = self._upsample(concat_input_ids, concat_embeddings)

            masked_embeddings.append(concat_embeddings)

        return masked_embeddings


# https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
class NucleotideTransformerEmbedder(BaseEmbedder):
    """
    Embed using the Nuclieotide Transformer (NT) model https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
    """

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

        # Get pretrained model
        if "v2" in model_name:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.is_v2 = True
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.is_v2 = False
        self.model.to(DEVICE)
        self.model.eval()

    def embed(
        self,
        sequences: List[str],
        uneven_length: bool = False,
    ):
        """
        Embed sequences using the Nuclieotide Transformer (NT) model.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        uneven_length : bool, optional
            Whether the sequences have uneven length. If True, the model should handle padding. Defaults to False.
        Returns
        -------
        torch.Tensor
            The embeddings of the sequences.
        """

        with torch.no_grad():

            sequences, chunk_ids = self.chunkify_sequences(sequences)

            output = self.tokenizer(
                sequences,
                return_tensors="pt",
                return_token_type_ids=False,
                padding="longest",
            )

            input_ids = output["input_ids"].int()
            attention_mask = output["attention_mask"]

            embeddings = (
                self.model(
                    input_ids.to(DEVICE),
                    attention_mask=attention_mask.to(DEVICE),
                    output_hidden_states=True,
                )["hidden_states"][-1]
                .detach()
                .cpu()
                .numpy()
            )
            input_ids = input_ids.numpy()

            embeddings = self.process_chunk_embeddings(embeddings, input_ids, chunk_ids)

            return embeddings

            # else:
            #     list_embeddings = []

            #     for sample_idx in range(embeddings.shape[0]):
            #         emb = embeddings[sample_idx]
            #         token_ids = input_ids[sample_idx]

            #         # Remove special tokens (CLS)
            #         emb = emb[1:, :]
            #         token_ids = token_ids[1:]

            #         if self.upsample_embeddings:
            #             emb = self._upsample(token_ids, emb)

            #         list_embeddings.append(emb)

            #     return np.array(list_embeddings)


class AWDLSTMEmbedder(BaseEmbedder):
    """
    Embed using the AWD-LSTM (https://arxiv.org/abs/1708.02182) baseline LM trained in BEND.
    """

    def load_model(self, model_path, **kwargs):
        """
        Load the AWD-LSTM baseline LM trained in BEND.

        Parameters
        ----------
        model_path : str
            The path to the model directory.
            If the model path does not exist, it will be downloaded from https://sid.erda.dk/cgi-sid/ls.py?share_id=dbQM0pgSlM&current_dir=pretrained_models&flags=f
        """

        # download model if not exists
        if not os.path.exists(model_path):
            print(
                f"Path {model_path} does not exists, model is downloaded from https://sid.erda.dk/cgi-sid/ls.py?share_id=dbQM0pgSlM&current_dir=pretrained_models&flags=f"
            )
            download_model(model="awd_lstm", destination_dir=model_path)
        # Get pretrained model
        self.model = AWDLSTMModelForInference.from_pretrained(model_path)
        self.model.to(DEVICE)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def embed(
        self,
        sequences: List[str],
        uneven_length: bool = False,
    ):
        """
        Embed sequences using the AWD-LSTM baseline LM trained in BEND.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        uneven_length : bool
            Whether the sequences have uneven length. If True, the model should handle padding. Defaults to False.
        Returns
        -------
        list or np.ndarray
            The embeddings of the sequences. If `uneven_length` is True, returns a list of embeddings, otherwise returns a numpy array of embeddings.
        """

        with torch.no_grad():
            output = self.tokenizer(
                sequences,
                return_tensors="pt",
                return_token_type_ids=False,
                padding="longest",
            )

            input_ids = output["input_ids"]

            embeddings = self.model(input_ids=input_ids.to(DEVICE)).last_hidden_state
            embeddings = embeddings.detach().cpu().numpy()

            if uneven_length:
                masked_embeddings = []
                attention_mask = output["attention_mask"].numpy().astype(bool)

                for idx in range(len(embeddings)):
                    # Remove padding from embeddings
                    masked_embeddings.append(embeddings[idx][attention_mask[idx]])

                # List of uneven length embeddings cannot be converted to a numpy array
                return masked_embeddings

        # If uneven_length is False, return a numpy array of embeddings
        return embeddings


class ConvNetEmbedder(BaseEmbedder):
    """
    Embed using the GPN-inspired ConvNet baseline LM trained in BEND.
    """

    def load_model(self, model_path, **kwargs):
        """
        Load the GPN-inspired ConvNet baseline LM trained in BEND.

        Parameters
        ----------
        model_path : str
            The path to the model directory.
            If the model path does not exist, it will be downloaded from https://sid.erda.dk/cgi-sid/ls.py?share_id=dbQM0pgSlM&current_dir=pretrained_models&flags=f
        """

        logging.set_verbosity_error()
        if not os.path.exists(model_path):
            print(
                f"Path {model_path} does not exists, model is downloaded from https://sid.erda.dk/cgi-sid/ls.py?share_id=dbQM0pgSlM&current_dir=pretrained_models&flags=f"
            )
            download_model(model="convnet", destination_dir=model_path)
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # load model
        self.model = ConvNetModel.from_pretrained(model_path).to(DEVICE).eval()

    def embed(
        self,
        sequences: List[str],
        uneven_length: bool = False,
    ):
        """
        Embed sequences using the GPN-inspired ConvNet baseline LM trained in BEND.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        uneven_length : bool, optional
            Whether the sequences have uneven length. If True, the model should handle padding. Defaults to False.
        Returns
        -------
        torch.Tensor
            The embeddings of the sequences.
        """

        with torch.no_grad():
            output = self.tokenizer(
                sequences,
                return_tensors="pt",
                return_token_type_ids=False,
                padding="longest",
            )

            input_ids = output["input_ids"]

            embeddings = self.model(input_ids=input_ids.to(DEVICE)).last_hidden_state
            embeddings = embeddings.detach().cpu().numpy()

            if uneven_length:
                masked_embeddings = []
                attention_mask = output["attention_mask"].numpy().astype(bool)

                for idx in range(len(embeddings)):
                    # Remove padding from embeddings
                    masked_embeddings.append(embeddings[idx][attention_mask[idx]])

                # List of uneven length embeddings cannot be converted to a numpy array
                return masked_embeddings

        # If uneven_length is False, return a numpy array of embeddings
        return embeddings


class HyenaDNAEmbedder(BaseEmbedder):
    """Embed using the HyenaDNA model https://arxiv.org/abs/2306.15794"""

    def load_model(
        self,
        model_path="pretrained_models/hyenadna/hyenadna-tiny-1k-seqlen",
        **kwargs,
    ):
        # '''Load the model from the checkpoint path
        # 'hyenadna-tiny-1k-seqlen'
        # 'hyenadna-small-32k-seqlen'
        # 'hyenadna-medium-160k-seqlen'
        # 'hyenadna-medium-450k-seqlen'
        # 'hyenadna-large-1m-seqlen'
        # '''
        # you only need to select which model to use here, we'll do the rest!
        """
        Load the HyenaDNA model.

        Parameters
        ----------
        model_path : str, optional
            Path to the model checkpoint. Defaults to 'pretrained_models/hyenadna/hyenadna-tiny-1k-seqlen'.
            If the path does not exist, the model will be downloaded from HuggingFace. Rather than just downloading the model,
            HyenaDNA's `from_pretrained` method relies on cloning the HuggingFace-hosted repository, and using git lfs to download the model.
            This requires git lfs to be installed on your system, and will fail if it is not.
        remove_special_tokens : bool, optional
            Whether to remove the CLS and SEP tokens from the embeddings. Defaults to True.
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
            model_max_length=self.max_sequence_length
            + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side="left",  # since HyenaDNA is causal, we pad on the left
        )

    def embed(
        self,
        sequences: List[str],
        uneven_length: bool = False,
    ):
        """Embeds a list of sequences using the HyenaDNA model.
        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        uneven_length : bool, optional
            Whether the sequences have uneven length. If True, the model should handle padding. Defaults to
        Returns
        -------
        torch.Tensor
            The embeddings of the sequences.
        """

        with torch.no_grad():
            if uneven_length:
                # If uneven length, we need to chunk the sequences
                chunked_sequences, chunk_ids = self.chunkify_sequences(sequences)

                input_ids = self.tokenizer(
                    chunked_sequences,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    return_attention_mask=False,  # HyenaDNA does not use attention masks
                    padding="longest",
                )["input_ids"]

                input_ids = torch.LongTensor(input_ids)
                embeddings = (
                    self.model(input_ids=input_ids.to(DEVICE)).detach().cpu().numpy()
                )
                input_ids = input_ids.numpy()

                return self.process_chunk_embeddings(embeddings, input_ids, chunk_ids)

            else:
                input_ids = self.tokenizer(
                    sequences,
                    return_tensors="pt",
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]

                input_ids = torch.LongTensor(input_ids)
                embeddings = (
                    self.model(input_ids=input_ids.to(DEVICE)).detach().cpu().numpy()
                )

                # Remove special tokens (CLS and SEP)
                return embeddings[:, 1:-1, :]


class DNABert2Embedder(BaseEmbedder):
    """
    Embed using the DNABERT2 model https://arxiv.org/pdf/2306.15006.pdf
    """

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
        uneven_length: bool = False,
    ):
        """Embeds a list sequences using the DNABERT2 model.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.

        Returns
        -------
        embeddings : List[np.ndarray]
            List of embeddings.
        """

        with torch.no_grad():

            sequences, chunk_ids = self.chunkify_sequences(sequences)

            output = self.tokenizer(
                sequences,
                return_tensors="pt",
                return_token_type_ids=False,
                padding="longest",
            )

            input_ids = output["input_ids"]
            attention_mask = output["attention_mask"]

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

            embeddings = self.process_chunk_embeddings(embeddings, input_ids, chunk_ids)

            return embeddings


# Class for one-hot encoding.
categories_4_letters_unknown = ["A", "C", "G", "N", "T"]


class EncodeSequence:
    def __init__(self, nucleotide_categories=categories_4_letters_unknown):

        self.nucleotide_categories = nucleotide_categories

        self.label_encoder = LabelEncoder().fit(self.nucleotide_categories)

    def transform_integer(
        self, sequence, return_onehot=False
    ):  # integer/onehot encode sequence
        if isinstance(sequence, np.ndarray):
            return sequence
        if isinstance(sequence[0], str):  # if input is str
            sequence = np.array(list(sequence))

        sequence = self.label_encoder.transform(sequence)

        if return_onehot:
            sequence = np.eye(len(self.nucleotide_categories))[sequence]
        return sequence

    def inverse_transform_integer(self, sequence):
        if isinstance(sequence, str):  # if input is str
            return sequence
        sequence = EncodeSequence.reduce_last_dim(sequence)  # reduce last dim
        sequence = self.label_encoder.inverse_transform(sequence)
        return ("").join(sequence)

    @staticmethod
    def reduce_last_dim(sequence):
        if isinstance(sequence, (str, list)):  # if input is str
            return sequence
        if len(sequence.shape) > 1:
            sequence = np.argmax(sequence, axis=-1)
        return sequence


def embed_nucleotide_transformer(sequences, model_name):
    return NucleotideTransformerEmbedder(model_name).embed(sequences)


def embed_awdlstm(sequences, model_path, disable_tqdm=False, **kwargs):
    return AWDLSTMEmbedder(model_path, **kwargs).embed(
        sequences, disable_tqdm=disable_tqdm
    )


def embed_convnet(sequences, model_path, disable_tqdm=False, **kwargs):
    return ConvNetEmbedder(model_path, **kwargs).embed(
        sequences, disable_tqdm=disable_tqdm
    )


def embed_sequence(sequences: List[str], embedding_type: str = "categorical", **kwargs):
    """
    sequences : list of sequences to embed
    """
    if not embedding_type:
        return sequences

    if embedding_type == "categorical" or embedding_type == "onehot":
        encode_seq = EncodeSequence()
        # embed to categorcal
        sequence = []
        for seq in sequences:
            sequence.append(torch.tensor(encode_seq.transform_integer(seq)))
            return sequence
    # embed with nt transformer:
    elif embedding_type == "nt_transformer":
        # model name "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
        sequences, cls_token = embed_nucleotide_transformer(sequences, **kwargs)
        return sequences, cls_token
    # embed with GPN
    # embed with DNAbert
    # embed with own models.
    elif embedding_type == "awdlstm":
        sequences = embed_awdlstm(sequences, disable_tqdm=True, **kwargs)
        return sequences
    elif embedding_type == "convnet":
        sequences = embed_convnet(sequences, disable_tqdm=True, **kwargs)
        return sequences

    return sequences
