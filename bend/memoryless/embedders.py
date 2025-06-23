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
from bend.models.gena_lm import BertModel as GenaLMBertModel
from bend.models.hyena_dna import HyenaDNAPreTrainedModel, CharacterTokenizer
from bend.models.dnabert2 import BertModel as DNABert2BertModel
from bend.models.dnabert2 import BertForMaskedLM as DNABert2BertForMaskedLM
from bend.utils.download import download_model, download_model_zenodo

from tqdm.auto import tqdm
from transformers import (
    logging,
    BertModel,
    BertConfig,
    BertTokenizer,
    AutoModel,
    AutoTokenizer,
    BigBirdModel,
    AutoModelForMaskedLM,
)
from sklearn.preprocessing import LabelEncoder

logging.set_verbosity_error()


# TODO graceful auto downloading solution for everything that is hosted in a nice way
# https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseEmbedder:
    """Base class for embedders.
    All embedders should inherit from this class.
    """

    def __init__(
        self,
        max_seq_len,
        upsample_embeddings=False,
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
        self.upsample_embeddings = upsample_embeddings
        self.max_seq_len = max_seq_len
        self.load_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        """Load the model. Should be implemented by the inheriting class."""
        raise NotImplementedError

    def embed(self, sequences: List[str], *args, **kwargs):
        """Embed a sequence. Should be implemented by the inheriting class.

        Parameters
        ----------
        sequences : str
            The sequences to embed.
        """
        raise NotImplementedError

    def remove_special_tokens(self, embedding):
        raise NotImplementedError

    def __call__(self, sequences: List[str], *args, **kwargs):
        """Embed a single sequence. Calls `embed` with the given arguments.

        Parameters
        ----------
        sequence : str
            The sequence to embed.
        *args
            Positional arguments. Passed to `embed`.
        **kwargs
            Keyword arguments. Passed to `embed`.

        Returns
        -------
        np.ndarray
            The embedding of the sequence.
        """

        embeddings = []

        print("MAX SEQ LEN", self.max_seq_len)

        if self.max_seq_len:
            for s in sequences:
                chunks = [
                    s[chunk : chunk + self.max_seq_len]
                    for chunk in range(0, len(s), self.max_seq_len)
                ]
                print(
                    f"Embedding {len(chunks)} chunks of length {[len(c) for c in chunks]}"
                )

                input_ids, attention_mask = self.tokenize(chunks)

                print(f"Input IDs shape: {input_ids.shape}")

                chunks_emb = self.embed(
                    input_ids, attention_mask, *args, disable_tqdm=True, **kwargs
                )
                print(f"Chunks embedding shape: {chunks_emb.shape}")

                embeddings.append(
                    self.concatenate_chunks(chunks_emb, attention_mask, input_ids)
                )

        else:
            embeddings = self.embed(sequences, *args, **kwargs)

        return embeddings

    def concatenate_chunks(self, chunks: List, attention_mask, input_ids):
        embedding = []
        for idx_chunk, chunk in enumerate(chunks):

            # remove padding
            chunk = chunk[attention_mask[idx_chunk].bool()]

            chunk = self.remove_special_tokens(chunk)

            print(f"Chunk {idx_chunk} shape: {chunk.shape}")
            if self.upsample_embeddings:
                chunk = self._upsample(input_ids[idx_chunk], embedding=chunk)

            embedding.append(chunk)

        embedding = np.concatenate(embedding, axis=0)
        return embedding

    def tokenize(self, sequences):
        """
        Tokenize the sequences using the provided tokenizer.
        Returns a list of tokenized sequences.
        """

        if not hasattr(self, "tokenizer"):
            raise ValueError(
                "Embedder does not have a tokenizer. Please load the model first."
            )

        output = self.tokenizer(
            sequences,
            return_tensors="pt",
            return_token_type_ids=False,
            padding="longest",
        )

        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]

        return input_ids, attention_mask

    def _upsample(
        self,
        token_ids: torch.Tensor,
        embedding: np.ndarray,
    ):
        """
        Upsample the embeddings to match the length of the input sequences.
        This is done by repeating the embedding vectors for each letter in the token.
        """

        tokens = self.tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        repetitions = [len(token) if token != "[UNK]" else 1 for token in tokens]

        upsampled_embedding = np.repeat(embedding, repetitions, axis=0)
        return upsampled_embedding



# https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
class NucleotideTransformerEmbedder(BaseEmbedder):
    """
    Embed using the Nuclieotide Transformer (NT) model https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
    """

    def load_model(
        self,
        model_name,
        max_tokens_len: int,
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
        return_logits : bool, optional
            Whether to return the logits. Note that we do not apply any masking. Defaults to False.
        return_loss : bool, optional
            Whether to return the loss. Note that we do not apply any masking. ``remove_special_tokens`` also ignores these dimensions when
            computing the loss.
            Defaults to False.
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

        self.max_tokens = max_tokens_len
        self.upsample_embeddings = True

        self.model.to(device)
        self.model.eval()

    def embed(
        self,
        input_ids,
        attention_mask,
        disable_tqdm: bool = False,
    ):
        """
        Embed sequences using the Nuclieotide Transformer (NT) model.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
             Whether to remove the special tokens from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to the length of the input sequence. Defaults to False.

        Returns
        -------
        List[np.ndarray]
            List of embeddings.
        """

        with torch.no_grad():
            input_ids = input_ids.int().to(device)
            attention_mask = attention_mask.to(device)

            outs = (
                self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    encoder_attention_mask=attention_mask,
                    output_hidden_states=True,
                )["hidden_states"][-1]
                .detach()
                .cpu()
                .numpy()
            )

            return outs

    def remove_special_tokens(self, embedding):
        return embedding[1:]


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
        self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def embed(
        self,
        sequences: List[str],
        disable_tqdm: bool = False,
    ):
        """
        Embed sequences using the AWD-LSTM baseline LM trained in BEND.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to the length of the input sequence. Defaults to False.
            Only provided for compatibility with other embedders. GPN embeddings are already the same length as the input sequence.

        Returns
        -------
        List[np.ndarray]
            List of embeddings.
        """

        with torch.no_grad():

            input_ids, _ = self.tokenize(sequences)
            input_ids = input_ids.to(device)
            embeddings = self.model(input_ids=input_ids).last_hidden_state

            embeddings = embeddings.detach().cpu().numpy()

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
        self.model = ConvNetModel.from_pretrained(model_path).to(device).eval()

    def embed(
        self,
        sequences: List[str],
        disable_tqdm: bool = False,
    ):
        """
        Embed sequences using the GPN-inspired ConvNet baseline LM trained in BEND.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to the length of the input sequence. Defaults to False.
            Only provided for compatibility with other embedders. GPN embeddings are already the same length as the input sequence.

        Returns
        -------
        List[np.ndarray]
            List of embeddings.
        """

        with torch.no_grad():

            input_ids, _ = self.tokenize(sequences)
            input_ids = input_ids.to(device)
            embeddings = self.model(input_ids=input_ids).last_hidden_state

            embeddings = embeddings.detach().cpu().numpy()

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
        return_logits : bool, optional
            If True, returns logits instead of embeddings. Defaults to False.
        return_loss : bool, optional
            If True, returns the unreduced next token prediction loss. Incompatible with return_logits. We trim special tokens from the
            output so that the loss is only computed on the ACTGN vocabulary.
              Defaults to False.


        """

        checkpoint_path, model_name = os.path.split(model_path)

        # all these settings are copied directly from huggingface.py

        # data settings:
        use_padding = True
        rc_aug = False  # reverse complement augmentation
        add_eos = False  # add end of sentence token

        # we need these for the decoder head, if using
        use_head = False
        n_classes = 2  # not used for embeddings only

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
            device=device,
            use_head=use_head,
            use_lm_head=False,  # we don't use the LM head for embeddings
            n_classes=n_classes,
        )
        model.eval()

        model.to(device)
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
            model_max_length=self.max_seq_len
            + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side="right",
        )

    def embed(
        self,
        input_ids,
        attention_mask,
        disable_tqdm: bool = True,
    ):
        """Embeds a list of sequences using the HyenaDNA model.
        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Whether to remove the CLS and SEP tokens from the embeddings. Defaults to True. Cannot be set to False if
            the return_loss option of the embedder is True (autoregression forces us to discard the BOS token position either way).
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to match the length of the input sequences. Defaults to False.
            Only provided for compatibility with other embedders. HyenaDNA embeddings are already the same length as the input sequence.
        Returns
        -------

        embeddings : List[np.ndarray]
            List of embeddings.
        """

        with torch.inference_mode():

            # place on device, convert to tensor
            input_ids = torch.LongTensor(input_ids).to(device)

            output = (
                self.model(input_ids)
                .detach()
                .cpu()
                .numpy()
            )

            return output

    def remove_special_tokens(self, embedding):
        return embedding[1:-1]


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

        Parameters
        ----------
        model_name : str, optional
            The name of the model to load. Defaults to "zhihan1996/DNABERT-2-117M".
            When providing a name, the model will be loaded from the HuggingFace model hub.
            Alternatively, you can provide a path to a local model directory.
        return_logits : bool, optional
            If True, returns logits instead of embeddings. Defaults to False.
        return_loss : bool, optional
            If True, returns the unreduced next token prediction loss. Incompatible with return_logits. If ``remove_special_tokens`` is True,
            the loss is only computed on the BPE vocabulary without the special tokens.
            Defaults to False.
        """

        # keep the source in this repo to avoid using flash attn.
        self.model = DNABert2BertForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.model.to(device)

        self.upsample_embeddings = True

    def embed(
        self,
        input_ids,
        attention_mask,
        disable_tqdm: bool = False,
    ):
        """Embeds a list sequences using the DNABERT2 model.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Whether to remove the CLS and SEP tokens from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to match the length of the input sequences. Defaults to False.

        Returns
        -------
        embeddings : List[np.ndarray]
            List of embeddings.
        """
        # '''
        # Note that this model uses byte pair encoding.
        # upsample_embedding repeats BPE token embeddings so that each nucleotide has its own embedding.
        # The [CLS] and [SEP] tokens are removed from the output if remove_special_tokens is True.
        # '''

        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            output = (
                self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )["hidden_states"]
                .detach()
                .cpu()
                .numpy()
            )

            return output

    def remove_special_tokens(self, embedding):
        return embedding[1:-1]


class CaduceusEmbedder(BaseEmbedder):

    def load_model(
        self,
        model_name: str = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        return_logits: bool = False,
        return_loss: bool = False,
        **kwargs,
    ):
        """
        Load the Caduceus model (https://arxiv.org/abs/2403.03234).

        Parameters
        ----------
        model_name : str, optional
            The name of the model to load. Defaults to "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16".
            When providing a name, the model will be loaded from the HuggingFace model hub.
            Alternatively, you can provide a path to a local model directory.
        return_logits : bool, optional
            If True, returns logits instead of embeddings. Defaults to False.
        return_loss : bool, optional
            If True, returns the unreduced next token prediction loss. Incompatible with return_logits.
            We trim special tokens from the output so that the loss is only computed on the ACTGN vocabulary.
              Defaults to False.


        """
        # check that we have mamba-ssm==1.2.0.post1
        try:
            import mamba_ssm
        except ImportError:
            raise ImportError(
                "Caduceus requires mamba-ssm==1.2.0.post1. Please install it with `pip install mamba-ssm==1.2.0.post1`."
            )
        if mamba_ssm.__version__ != "1.2.0.post1":
            raise ImportError(
                "Caduceus requires mamba-ssm==1.2.0.post1. Please install it with `pip install mamba-ssm==1.2.0.post1`."
            )

        if return_logits and return_loss:
            raise ValueError("Only one of return_logits and return_loss can be True")

        self.max_length = 131072
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.model.to(device)

        self.return_logits = return_logits
        self.return_loss = return_loss

    def embed(
        self,
        sequences: List[str],
        disable_tqdm: bool = False,
        remove_special_tokens: bool = True,
        upsample_embeddings: bool = False,
    ):
        """
        Embed sequences using the Caduceus model.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Whether to remove the CLS and SEP tokens from the embeddings. Defaults to True. Only provided for compatibility with other embedders.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to match the length of the input sequences. Defaults to False.
            Only provided for compatibility with other embedders. Caduceus embeddings are already the same length as the input sequence.

        Returns
        -------
        List[np.ndarray]
            List of embeddings.
        """
        ref_tokenized = self.tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=self.max_length,
            truncation=True,
        )
        embeddings = []
        with torch.no_grad():
            for sequence in tqdm(sequences, disable=disable_tqdm):
                chunks = [
                    sequence[chunk : chunk + self.max_length]
                    for chunk in range(0, len(sequence), self.max_length)
                ]
                embedded_chunks = []
                for n_chunk, chunk in enumerate(chunks):
                    input_ids = self.tokenizer(
                        chunk,
                        return_tensors="pt",
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        add_special_tokens=False,
                    )["input_ids"]

                    if self.return_logits:
                        out = (
                            self.model(
                                input_ids=input_ids.to(device),
                                output_hidden_states=False,
                                return_dict=True,
                            )["logits"]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                    elif self.return_loss:
                        out = self.model(
                            input_ids=input_ids.to(device),
                            output_hidden_states=False,
                            return_dict=True,
                        )[
                            "logits"
                        ]  # (1, seq_len, 16)
                        out = out[
                            :, :, 7:12
                        ]  # 0-6 are special tokens. vocab_size is only 12 so last 4 dimensions are dead.
                        targets = input_ids - 7  # shift to 0-indexed
                        out = torch.nn.functional.cross_entropy(
                            out.view(-1, out.size(-1)),
                            targets.view(-1).to(device),
                            reduction="none",
                        )
                        out = (
                            out.unsqueeze(0).detach().cpu().numpy()
                        )  # dim 0 gets lost because of view

                    else:
                        out = (
                            self.model(
                                input_ids=input_ids.to(device),
                                output_hidden_states=True,
                            )["hidden_states"][-1]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                    embedded_chunks.append(out)

                embedding = np.concatenate(embedded_chunks, axis=1)
                embeddings.append(embedding)

        return embeddings


# Class for one-hot encoding.
categories_4_letters_unknown = ["A", "C", "G", "N", "T"]


class OneHotEmbedder(BaseEmbedder):
    """Onehot encode sequences"""

    def __init__(self, nucleotide_categories=categories_4_letters_unknown):
        """Get an onehot encoder for nucleotide sequences.

        Parameters
        ----------
        nucleotide_categories : List[str], optional
            List of nucleotides in the alphabet. Defaults to ['A', 'C', 'G', 'N', 'T'].
        """

        self.nucleotide_categories = nucleotide_categories

        self.label_encoder = LabelEncoder().fit(self.nucleotide_categories)

    def embed(
        self,
        sequences: List[str],
        disable_tqdm: bool = False,
        return_onehot: bool = False,
        upsample_embeddings: bool = False,
    ):
        """Onehot encode sequences.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        return_onehot : bool, optional
            Whether to return onehot encoded sequences. Defaults to False.
            If false, returns integer encoded sequences.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to match the length of the input sequences. Defaults to False.

        Returns
        -------
        embeddings : List[np.ndarray]
            List of one-hot encodings or integer encodings, depending on return_onehot.
        """
        # """Onehot endode sequences"""
        embeddings = []
        for s in tqdm(sequences, disable=disable_tqdm):
            s = self._transform_integer(s, return_onehot=return_onehot)
            s = s[None, :]  # dummy batch dim, as customary for embeddings
            embeddings.append(s)
        return embeddings

    def _transform_integer(
        self, sequence: str, return_onehot=False
    ):  # integer/onehot encode sequence
        sequence = np.array(list(sequence))

        sequence = self.label_encoder.transform(sequence)
        if return_onehot:
            sequence = np.eye(len(self.nucleotide_categories))[sequence]
        return sequence


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


# backward compatibility
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
