"""
Implements the BaseEmbedder interface, allowing embedding of sequences using
the ConvNet baseline LM trained in BEND.
Outputs a dictionary of pooled embeddings based on the specified pooling modes.
"""

import os
import warnings
from typing import List

import torch
from transformers import AutoTokenizer, logging

from bend_hybrid.embedding.pooling import PoolingMode, pool_name_to_function
from bend_hybrid.models.dilated_cnn import ConvNetModel
from bend_hybrid.utils import get_device

from .abstract_class import BaseEmbedder

logging.set_verbosity_error()
DEVICE = get_device()

# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


class ConvNetEmbedder(BaseEmbedder):
    """
    Embed using the GPN-inspired ConvNet baseline LM trained in BEND.
    """

    def get_start_end_token_ids(self):
        """
        Get the start and end token IDs for the ConvNet model.
        Returns
        -------
        Tuple[None, None]
            None, None as ConvNet does not use special start/end tokens.
        """
        return None, None

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
        pooling: List[PoolingMode] = [PoolingMode.DEFAULT],
        sequence_length: int = None,
    ):
        """
        Embed sequences using the GPN-inspired ConvNet baseline LM trained in BEND.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        pooling : List[PoolingMode], optional
            List of pooling modes to apply to the embeddings. Default is [PoolingMode.DEFAULT].
        sequence_length : int, optional
            The length of the sequences.
        Returns
        -------
        torch.Tensor
            The embeddings of the sequences.
        """

        output = self.tokenizer(
            sequences,
            return_tensors="pt",
            return_token_type_ids=False,
            padding="longest",
        )

        input_ids = output["input_ids"]

        with torch.no_grad():
            embeddings = (
                self.model(input_ids=input_ids.to(DEVICE))
                .last_hidden_state.detach()
                .cpu()
                .numpy()
            )

        # Remove padding embeddings
        if self.tokenizer.pad_token_id in input_ids:
            attention_mask = output["attention_mask"]
            attention_mask = attention_mask.numpy().astype(bool)

            embeddings = self._remove_padding(embeddings, attention_mask)

        # Pooling
        output = {}
        for mode in pooling:
            if (
                mode is PoolingMode.CLS
                or mode is PoolingMode.EOS
                or mode is PoolingMode.MEAN_UPSAMPLE
            ):
                warnings.warn(
                    f"Pooling mode {mode.value} not supported for ResNetLM. Skipping."
                )
                continue

            output[mode.value] = pool_name_to_function[mode](embeddings)

        return output
