import random
import torch
import numpy as np

PADDING_VALUE = -100


def get_device():
    """Returns the device to be used for PyTorch operations."""

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def generate_random_dna_sequence(min_length: int = 10, max_length: int = 25):
    """Generate a random DNA sequence of a random length between min_length and max_length.

    Args:
        min_length (int): Minimum length of the DNA sequence.
        max_length (int): Maximum length of the DNA sequence.
    Returns:
        str: A random DNA sequence consisting of characters A, C, G, T.
    """

    length = random.randint(min_length, max_length)
    return "".join(random.choice(["A", "C", "G", "T"]) for _ in range(length))


def pad_embeddings(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    padding_value: int = PADDING_VALUE,
):
    """
    Pads the embeddings based on the attention mask.

    Args:
        embeddings (torch.Tensor): The embeddings tensor to be padded.
        attention_mask (torch.Tensor): The attention mask indicating which tokens are valid.
        padding_value (int, optional): The value to use for padding. Defaults to PADDING_VALUE.
    Returns:
        torch.Tensor: The padded embeddings tensor.
    """
    embeddings[~attention_mask] = padding_value
    return embeddings


def chunkify_sequences(sequences, max_length):
    chunks = []
    chunk_ids = []

    for s_idx, s in enumerate(sequences):
        chunked_sequence = [s[i : i + max_length] for i in range(0, len(s), max_length)]
        chunks.extend(chunked_sequence)
        chunk_ids.extend([s_idx] * len(chunked_sequence))

    return chunks, np.array(chunk_ids)


def remove_special_tokens(tokenizer, token_ids: list, embedding: np.ndarray):
    """
    Removes special tokens from the embeddings based on the tokenizer's special token mask.

    This function assumes that the tokenizer has a method `get_special_tokens_mask` that returns a mask indicating which tokens are special tokens.

    Args:
        tokenizer: The tokenizer used to process the token IDs.
        token_ids (list): The list of token IDs.
        embedding (np.ndarray): The embeddings tensor from which to remove special tokens.
    Returns:
        np.ndarray: The embeddings tensor with special tokens removed.

    """

    mask_special_tokens = ~np.array(
        tokenizer.get_special_tokens_mask(token_ids, already_has_special_tokens=True),
        dtype=bool,
    )

    return embedding[mask_special_tokens]


def _upsample(tokenizer, token_ids: np.ndarray, embedding: np.ndarray):
    """
    Upsamples the embeddings based on the number of characters in each token.
    Args:
        tokenizer: The tokenizer used to process the token IDs.
        token_ids (np.ndarray): The array of token IDs.
        embedding (np.ndarray): The embeddings tensor to be upsampled.
    Returns:
        np.ndarray: The upsampled embeddings tensor.
    """

    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
    repetitions = np.array([len(token) for token in tokens])

    return np.repeat(embedding, repetitions, axis=0)


def process_chunk_embeddings(
    tokenizer,
    chunk_embeddings: np.ndarray,
    input_ids: np.ndarray,
    chunk_ids: np.ndarray,
    upsample: bool = False,
):
    """
    Processes the embeddings by concatenating them based on their chunk IDs and optionally upsampling.
    Args:
        chunk_embeddings (np.ndarray): The embeddings for each chunk.
        chunk_ids (np.ndarray): The IDs corresponding to each chunk.
        upsample (bool): Whether to upsample the embeddings based on the number of characters in each token.
    Returns:
        list: A list of processed embeddings, where each entry corresponds to a unique chunk ID.
    """
    masked_embeddings = []
    masked_tokens = []

    for sequence_idx in np.unique(chunk_ids):

        mask_sequence = sequence_idx == chunk_ids
        concat_embeddings = np.concatenate(chunk_embeddings[mask_sequence], axis=0)
        concat_input_ids = np.concatenate(input_ids[mask_sequence], axis=0)

        concat_embeddings = remove_special_tokens(
            tokenizer, concat_input_ids, concat_embeddings
        )

        if upsample:
            concat_embeddings = _upsample(
                tokenizer, concat_input_ids, concat_embeddings
            )

        masked_embeddings.append(concat_embeddings)
        masked_tokens.append(
            tokenizer.convert_ids_to_tokens(concat_input_ids, skip_special_tokens=True)
        )

    return masked_embeddings, masked_tokens
