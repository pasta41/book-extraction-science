"""Data loading utilities to read a full book, chunk it into fixed-length
examples, and package them with metadata into a PyTorch Dataset for
training or evaluation.
"""

import hashlib
import torch
import logging
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from script.memorization.utils.tokenize import tokenize_custom
from script.memorization.utils.paths import Paths

logger = logging.getLogger(__name__)


def load_dataset(paths: Paths, args, tokenizer: AutoTokenizer) -> Dataset:
    """
    Load and tokenize a dataset from a book file using sliding window chunking.

    Args:
        paths: Paths object with `input_book_path` and `book_name` attributes
        args: Main args object with `model`, `ex_len`, and `stride` attributes
        tokenizer: The model's tokenizer instance

    Returns:
        A PyTorch Dataset containing tokenized examples and metadata.
    """
    return load_book_sliding(
        paths.input_book_path,
        paths.book_name,
        tokenizer,
        args.model,
        args.ex_len,
        args.stride,
    )


class InputDataset(Dataset):
    """Simple PyTorch Dataset wrapper for batched inputs."""

    def __init__(
        self,
        batched_inputs: Dict[str, torch.Tensor],
        metadata_list: Optional[List[Dict]] = None,
    ) -> None:
        self.batched_inputs = batched_inputs
        self.metadata_list = metadata_list

    def __len__(self) -> int:
        return len(self.batched_inputs["input_ids"])

    def __getitem__(self, idx: int) -> Dict:
        example = {key: value[idx] for key, value in self.batched_inputs.items()}
        example["metadata"] = (
            self.metadata_list[idx] if self.metadata_list else None
        )
        return example


def load_book_sliding(
    book_path: str,
    book_name: str,
    tokenizer: AutoTokenizer,
    model_key: str,
    ex_len: int,
    stride: int,
    slice_multiplier: int = 8,
) -> InputDataset:
    """
    Load a single book from a text file and chunk it into fixed-length
    tokenized examples using a character-based sliding window.

    Args:
        book_path: Path to the book text file
        book_name: Book's name
        tokenizer: Model's tokenizer instance
        model_key: Unique model identifier string (for hashing)
        ex_len: Length of each tokenized example
        stride: Number of characters to slide the window each step
        slice_multiplier: Multiplier for slice length (characters) relative to
                          example length (tokens)

    Returns:
        An InputDataset containing tokenized input batches and metadata.
    """
    with open(book_path, "r") as file:
        book_text = file.read()

    all_chunks = book_text_to_sliding_window_token_chunks(
        book_text,
        model_key,
        tokenizer,
        chunk_size=ex_len,
        slice_length=slice_multiplier * ex_len,
        stride=stride,
    )

    hashed_data_ids = set()
    input_ids_list = []
    attention_mask_list = []
    metadata_list = []

    logger.info("Loading into torch Dataset")
    last_logged = -10
    total_chunks = len(all_chunks)

    for i, chunk in enumerate(all_chunks):
        if len(chunk["input_ids"]) != ex_len:
            raise ValueError(f"Chunk {chunk} length != expected {ex_len}")

        ex_hash_id = compute_input_hash(model_key, chunk["input_ids"])

        if ex_hash_id in hashed_data_ids:
            logger.debug(f"Skipping example {i}: duplicate detected via hash")
            continue

        hashed_data_ids.add(ex_hash_id)
        input_ids_list.append(chunk["input_ids"])
        attention_mask_list.append(chunk["attention_mask"])

        metadata = create_metadata_dict(
            input_hash=ex_hash_id,
            text=chunk["text"],
            book_name=book_name,
            example_num=i,
            ex_len_tokens=ex_len,
            ex_len_text=chunk["text_len"],
            book_len=len(book_text),
            start=chunk["start_idx"],
            end=chunk["end_idx"],
            task_name=f"sliding_window_{stride}",
        )
        metadata_list.append(metadata)

        percent_complete = int(100 * i / total_chunks)
        if percent_complete >= last_logged + 10:
            logger.info(f"Progress: {percent_complete}%")
            last_logged = percent_complete

    batched_inputs = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
    }

    dataset = InputDataset(batched_inputs, metadata_list=metadata_list)
    logger.info(f"Finished reading {len(dataset)} examples")
    return dataset


def book_text_to_sliding_window_token_chunks(
    text: str,
    model_key: str,
    tokenizer: AutoTokenizer,
    chunk_size: int,
    slice_length: int,
    stride: int,
) -> List[Dict]:
    """
    Convert raw full book text into tokenized chunks using a sliding window.

    Args:
        text: Raw full book text
        model_key: Model identifier (used for hashing)
        tokenizer: Model's tokenizer instance
        chunk_size: Desired fixed token chunk size for examples
        slice_length: Length of text slice to tokenize each iteration
        stride: Number of characters to move the window forward each iteration

    Returns:
        List of dictionaries, each containing:
        'input_ids', 'attention_mask', 'start_idx', 'end_idx', 'text_len',
        and decoded 'text' of the chunk.
    """
    all_chunks = []
    text_cursor = 0
    text_length = len(text)
    last_logged = -10

    while text_cursor < text_length:
        start_idx = text_cursor
        end_idx_plus_one = min(text_cursor + slice_length, text_length)
        text_slice = text[start_idx:end_idx_plus_one]

        encoding = tokenize_custom(text_slice, tokenizer, model_key)
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]

        if len(input_ids) >= chunk_size:
            decoding = tokenizer.decode(
                input_ids[:chunk_size], skip_special_tokens=True
            )
            decoding_len = len(decoding)
            all_chunks.append(
                {
                    "input_ids": input_ids[:chunk_size],
                    "attention_mask": attention_mask[:chunk_size],
                    "start_idx": start_idx,
                    "end_idx": start_idx + decoding_len - 1,
                    "text_len": decoding_len,
                    "text": decoding,
                }
            )

        text_cursor += stride

        percent_complete = int(100 * text_cursor / text_length)
        if percent_complete >= last_logged + 10:
            logger.info(f"Progress: {percent_complete}%")
            last_logged = percent_complete

    return all_chunks


def compute_input_hash(model_key: str, input_ids: torch.Tensor) -> str:
    """
    Compute a unique SHA-256 hash string for a tokenized example and model.

    Args:
        model_key: Identifier string for the model
        input_ids: Tensor of token IDs for the example

    Returns:
        Unique example-model hash as a hexadecimal string.
    """
    input_ids_str = str(input_ids.tolist())
    hash_input = f"{model_key}-{input_ids_str}"
    return hashlib.sha256(hash_input.encode()).hexdigest()


def create_metadata_dict(
    input_hash: str,
    text: str,
    book_name: str,
    example_num: int,
    ex_len_tokens: int,
    ex_len_text: int,
    book_len: int,
    start: int,
    end: int,
    task_name: str,
) -> Dict:
    """
    Create a metadata dictionary describing a tokenized example chunk.

    Args:
        input_hash: Unique hash of the tokenized example
        text: Decoded example text
        book_name: Book the text comes from
        example_num: Index number of the example
        ex_len_tokens: Token length of the example
        ex_len_text: Character length of the decoded example
        book_len: Total length of the book in characters
        start: Start character index of example in book
        end: End character index of example in book
        task_name: Description of the experiment

    Returns:
        Dictionary containing all example metadata.
    """
    return {
        "hash": input_hash,
        "text": text,  # for debugging; do not rely on this for determinism
        "book_name": book_name,
        "book_len": book_len,
        "example": example_num,
        "example_len": ex_len_text,
        "token_len": ex_len_tokens,
        "example_loc": {"start_idx": start, "end_idx": end},
        "task": task_name,
    }
