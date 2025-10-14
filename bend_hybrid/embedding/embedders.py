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
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging

from bend_hybrid.embedding.pooling import PoolingMode, pool_embeddings
from bend_hybrid.models.awd_lstm import AWDLSTMModelForInference
from bend_hybrid.models.dilated_cnn import ConvNetModel
from bend_hybrid.models.dnabert2 import BertForMaskedLM as DNABert2BertForMaskedLM
from bend_hybrid.models.hyena_dna import CharacterTokenizer, HyenaDNAPreTrainedModel
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

    def chunk_sequence(self, sequence: str) -> List[str]:
        """
        Chunkify the input sequence into smaller chunks, defined by `self.max_sequence_length`.

        Parameters
        ----------
        sequence : str
            The input sequence to chunk.

        Returns
        -------
        List[str]
            A list containing the chunked sequence.
        """

        return [
            sequence[chunk : chunk + self.max_sequence_length]
            for chunk in range(0, len(sequence), self.max_sequence_length)
        ]

    def _remove_pad_tokens(
        self, embedding: np.ndarray, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove the PAD tokens from the embedding and input IDs based on the attention mask.

        Parameters
        ----------
        embedding : np.ndarray
            The embedding to process.
        input_ids : np.ndarray
            The input IDs to process.
        attention_mask : np.ndarray
            The attention mask to apply to the input IDs.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The processed embedding and input IDs.
        """

        attention_mask = attention_mask.astype(bool)
        return embedding[attention_mask, :], input_ids[attention_mask]


# https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
class NucleotideTransformerEmbedder(BaseEmbedder):
    """
    Embed using the Nuclieotide Transformer (NT) model https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
    """

    def load_model(
        self,
        model_name,
        max_tokens=1000,
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

        self.max_tokens = max_tokens

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

            if uneven_length:
                # If uneven length, we need to chunk the sequences

                embeddings = []

                for sequence in sequences:

                    chunks = self.chunk_sequence(sequence)

                    chunks_embeddings = []

                    for chunk in chunks:

                        input_ids = self.tokenizer(
                            chunk,
                            return_tensors="pt",
                        )["input_ids"].int()

                        if len(input_ids[0]) > self.max_tokens:
                            splits = torch.split(input_ids, self.max_tokens, dim=-1)
                            chunk_emb = [self.get_embedding(split) for split in splits]
                            chunk_emb = np.concatenate(chunk_emb, axis=1)
                        else:
                            chunk_emb = self.get_embedding(input_ids)

                        input_ids = input_ids.numpy()

                        # remove batch dimension
                        chunk_emb = chunk_emb[0, :, :]
                        input_ids = input_ids[0, :]

                        chunk_emb, input_ids = self._remove_cls_tokens(
                            chunk_emb, input_ids
                        )

                        if self.upsample_embeddings:
                            chunk_emb = self._upsample(input_ids, chunk_emb)

                        chunks_embeddings.append(chunk_emb)

                    embeddings.append(np.concatenate(chunks_embeddings, axis=0))

                return [pool_embeddings(embeddings, PoolingMode.DEFAULT)]

            # if sequences are of the same length, we can batch process without chunking

            output = self.tokenizer(
                sequences,
                return_tensors="pt",
                return_token_type_ids=False,
                padding="longest",
            )
            input_ids = output["input_ids"].int()
            attention_masks = output["attention_mask"]

            embeddings = self.get_embedding(input_ids, attention_masks)
            input_ids = input_ids.numpy()
            attention_masks = attention_masks.numpy().astype(bool)

            output = [pool_embeddings(embeddings, PoolingMode.CLS)]

            list_embeddings = []
            list_embeddings_no_upsample = []

            for idx, _ in enumerate(embeddings):
                emb = embeddings[idx]
                token_ids = input_ids[idx]

                emb, token_ids = self._remove_pad_tokens(
                    emb, token_ids, attention_masks[idx]
                )

                emb, token_ids = self._remove_cls_tokens(emb, token_ids)
                list_embeddings_no_upsample.append(emb)

                emb = self._upsample(token_ids, emb)
                list_embeddings.append(emb)

            embeddings = np.array(list_embeddings)
            embeddings_no_upsample = np.array(list_embeddings_no_upsample)

            output.extend(
                [
                    pool_embeddings(embeddings, mode)
                    for mode in [PoolingMode.DEFAULT, PoolingMode.MEAN, PoolingMode.MAX]
                ]
            )
            output.append(
                pool_embeddings(embeddings_no_upsample, PoolingMode.MEAN_NO_UPSAMPLE)
            )
            return output

    def get_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> np.ndarray:
        """
        Get the embedding of the given input IDs.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input IDs for which to get the embeddings.
        attention_mask : torch.Tensor, optional
            The attention mask to apply to the input IDs.

        Returns
        -------
        np.ndarray
            The embeddings for the input IDs.
        """

        embedding = (
            self.model(
                input_ids.to(DEVICE),
                attention_mask=(
                    attention_mask.to(DEVICE) if attention_mask is not None else None
                ),
                encoder_attention_mask=(
                    attention_mask.to(DEVICE) if attention_mask is not None else None
                ),
                output_hidden_states=True,
            )["hidden_states"][-1]
            .detach()
            .cpu()
            .numpy()
        )

        return embedding

    def _remove_cls_tokens(
        self,
        embedding: np.ndarray,
        input_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove the PAD and CLS token embedding.

        Parameters
        ----------
        embedding : np.ndarray
            The embedding to process.

        Returns
        -------
        np.ndarray
            The embedding with the CLS token removed.
        """
        return embedding[1:, :], input_ids[1:]

    def _upsample(self, token_ids: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        """
        Upsamples the embeddings based on the number of characters in each non-special token.
        Skips repeating any special tokens, such as CLS, UNK or PAD.

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

        repetitions = np.array([len(token) for token in tokens], dtype=np.int64)

        return np.repeat(embedding, repetitions, axis=0)


def download_model(
    model: str = "convnet",
    base_url: str = "https://sid.erda.dk/share_redirect/dbQM0pgSlM/pretrained_models/",
    destination_dir: str = "./pretrained_models/",  # pretrained_models
) -> None:
    """Download BEND pretrained model checkpoints from the ERDA URL.
    Uses wget to download the files.

    Parameters
    ----------
    model : str
        Model to download. Needs to be a directory name in base_url.
    base_url : str
        Base URL to download from.
        Default is BEND's pretrained models directory on ERDA.
    destination_dir : str
        Destination directory to download to.
        Default is ./pretrained_models/

    Returns
    -------
    None.
    """

    # """download model from url to destination directory"""
    # make destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    files = [
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for file in files:
        url = f"{base_url}{model}/{file}"
        os.system(f"wget {url} -P {destination_dir}/")

    return


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

    def embed(self, sequences: List[str], uneven_length: bool = False, **kwargs):
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

                # Remove padding from embeddings
                for idx, _ in enumerate(embeddings):
                    masked_embeddings.append(embeddings[idx][attention_mask[idx]])

                # List of uneven length embeddings cannot be converted to a numpy array
                return [pool_embeddings(masked_embeddings, PoolingMode.DEFAULT)]

        # If uneven_length is False, return a numpy array of embeddings
        return [
            pool_embeddings(embeddings, mode)
            for mode in [PoolingMode.DEFAULT, PoolingMode.MEAN, PoolingMode.MAX]
        ]


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

                # Remove padding from embeddings
                for idx, _ in enumerate(embeddings):
                    masked_embeddings.append(embeddings[idx][attention_mask[idx]])

                # List of uneven length embeddings cannot be converted to a numpy array
                return [pool_embeddings(masked_embeddings, PoolingMode.DEFAULT)]
        # If uneven_length is False, return a numpy array of embeddings
        return [
            pool_embeddings(embeddings, mode)
            for mode in [PoolingMode.DEFAULT, PoolingMode.MEAN, PoolingMode.MAX]
        ]


class HyenaDNAEmbedder(BaseEmbedder):
    """Embed using the HyenaDNA model https://arxiv.org/abs/2306.15794"""

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
            padding_side="right",  # as we are interested in the embeddings, and not in generating sequences, we pad on the right
        )

    def _remove_cls_sep_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Remove embeddings of the CLS and SEP tokens.

        Parameters
        ----------
        embeddings : np.ndarray
            The embeddings to process.

        Returns
        -------
        np.ndarray
            The embeddings with CLS and SEP tokens removed.
        """
        return embeddings[:, 1:-1, :]

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

                embeddings = []

                for sequence in sequences:
                    chunks = self.chunk_sequence(sequence)

                    chunks_embeddings = []

                    for chunk in chunks:

                        input_ids = self.tokenizer(
                            chunk,
                            return_tensors="pt",
                        )["input_ids"]
                        input_ids = torch.LongTensor(input_ids)

                        chunk_emb = (
                            self.model(input_ids=input_ids.to(DEVICE))
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        chunk_emb = self._remove_cls_sep_embeddings(chunk_emb)

                        # remove batch dimension
                        # (1, seq_len, emb_dim) -> (seq_len, emb_dim)
                        chunk_emb = chunk_emb[0, :, :]

                        chunks_embeddings.append(chunk_emb)

                    embeddings.append(np.concatenate(chunks_embeddings, axis=0))

                return [pool_embeddings(embeddings, PoolingMode.DEFAULT)]

            input_ids = self.tokenizer(
                sequences,
                return_tensors="pt",
            )["input_ids"]
            input_ids = torch.LongTensor(input_ids)

            embeddings = (
                self.model(input_ids=input_ids.to(DEVICE)).detach().cpu().numpy()
            )

            output = [pool_embeddings(embeddings, PoolingMode.EOS)]

            embeddings = self._remove_cls_sep_embeddings(embeddings)

            output.extend(
                [
                    pool_embeddings(embeddings, mode)
                    for mode in [PoolingMode.DEFAULT, PoolingMode.MEAN, PoolingMode.MAX]
                ]
            )
            return output


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

            if uneven_length:
                # If uneven length, we need to chunk the sequences

                embeddings = []
                for sequence in sequences:
                    chunks = self.chunk_sequence(sequence)
                    chunks_embeddings = []

                    for chunk in chunks:
                        input_ids = self.tokenizer(chunk, return_tensors="pt")[
                            "input_ids"
                        ]
                        chunk_emb = (
                            self.model(
                                input_ids.to(DEVICE),
                                output_hidden_states=True,
                            )["hidden_states"]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        input_ids = input_ids.numpy()

                        # remove batch dimension
                        chunk_emb = chunk_emb[0, :, :]
                        input_ids = input_ids[0, :]

                        chunk_emb, input_ids = self._remove_cls_sep_tokens(
                            chunk_emb, input_ids
                        )

                        if self.upsample_embeddings:
                            chunk_emb = self._upsample(input_ids, chunk_emb)

                        chunks_embeddings.append(chunk_emb)

                    embeddings.append(np.concatenate(chunks_embeddings, axis=0))

                return [pool_embeddings(embeddings, PoolingMode.DEFAULT)]

            output = self.tokenizer(sequences, return_tensors="pt", padding="longest")
            input_ids = output["input_ids"]
            attention_mask = output["attention_mask"]

            embeddings = (
                self.model(
                    input_ids.to(DEVICE),
                    attention_mask=attention_mask.to(DEVICE),
                    encoder_attention_mask=attention_mask.to(DEVICE),
                    output_hidden_states=True,
                )["hidden_states"]
                .detach()
                .cpu()
                .numpy()
            )
            input_ids = input_ids.numpy()
            attention_mask = attention_mask.numpy().astype(bool)

            output = [pool_embeddings(embeddings, PoolingMode.CLS)]

            list_embeddings = []
            list_upsampled = []

            for idx, _ in enumerate(embeddings):
                emb = embeddings[idx]
                token_ids = input_ids[idx]

                emb, token_ids = self._remove_pad_tokens(
                    emb, token_ids, attention_mask=attention_mask[idx]
                )

                emb, token_ids = self._remove_cls_sep_tokens(emb, token_ids)
                list_embeddings.append(emb)

                emb = self._upsample(token_ids, emb)
                list_upsampled.append(emb)

            embeddings = np.array(list_embeddings)
            upsampled = np.array(list_upsampled)

            output.extend(
                [
                    pool_embeddings(embeddings, mode)
                    for mode in [PoolingMode.DEFAULT, PoolingMode.MEAN, PoolingMode.MAX]
                ]
            )
            output.append(pool_embeddings(upsampled, PoolingMode.MEAN_NO_UPSAMPLE))

            return output

    def _remove_cls_sep_tokens(
        self, embeddings: np.ndarray, input_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove embeddings of the CLS and SEP tokens.

        Parameters
        ----------
        embeddings : np.ndarray
            The embeddings to process.
        input_ids : np.ndarray
            The input IDs to process.
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The embeddings and input IDs with special tokens removed.
        """
        return embeddings[1:-1, :], input_ids[1:-1]

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
                "Tokenizer does not have a method `convert_ids_to_tokens`. "
                "Please check the tokenizer implementation."
            )

        tokens = self.tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=False
        )

        repetitions = []
        for token in tokens:
            if token == "[CLS]" or token == "[SEP]":
                continue

            if token == "[UNK]":
                repetitions.append(1)
            else:
                repetitions.append(len(token))

        repetitions = np.array(repetitions, dtype=np.int32)

        return np.repeat(embedding, repetitions, axis=0)
