import random
import torch

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


def remove_special_tokens_and_padding(
    tokenizer, token_ids: list, embedding: torch.Tensor
):
    """
    Removes special tokens from the embeddings based on the tokenizer's special token mask.

    This function assumes that the tokenizer has a method `get_special_tokens_mask` that returns a mask indicating which tokens are special tokens.


    Args:
        tokenizer: The tokenizer used to process the token IDs.
        token_ids (list): The list of token IDs.
        embedding (torch.Tensor): The embeddings tensor from which to remove special tokens.
    Returns:
        torch.Tensor: The embeddings tensor with special tokens removed.

    """

    mask_special_tokens = torch.tensor(
        tokenizer.get_special_tokens_mask(token_ids, already_has_special_tokens=True),
        dtype=torch.bool,
    )

    return embedding[~mask_special_tokens]


def upsample(tokenizer, token_ids: list, embedding: torch.Tensor):
    """
    Upsamples the embeddings based on the number of characters in each token.
    Args:
        tokenizer: The tokenizer used to process the token IDs.
        token_ids (list): The list of token IDs.
        embedding (torch.Tensor): The embeddings tensor to be upsampled.
    Returns:
        torch.Tensor: The upsampled embeddings tensor.
    """

    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
    repetitions = torch.tensor([len(token) for token in tokens])

    return torch.repeat_interleave(embedding, repetitions, dim=0)
