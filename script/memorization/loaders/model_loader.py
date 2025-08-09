"""
Model and tokenizer loading utilities.

Supports precision selection, model-specific quirks, and tokenizer
initialization with proper padding and decoding options.
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from script.memorization.constants import MODEL_INFO

logger = logging.getLogger(__name__)


def load_model(model_key: str, precision: int = 16) -> AutoModelForCausalLM:
    """
    Load a pretrained language model with specified precision.

    Args:
        model_key: Identifier for the model (must exist in MODEL_INFO)
        precision: Precision level; supports 16 (default) or 32

    Returns:
        AutoModelForCausalLM: The loaded and initialized model.
    """
    weights_path = MODEL_INFO[model_key].path

    if precision == 16:
        if any(m in model_key for m in ("Qwen", "gemma", "Llama-3")):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    logger.info(
        "Loading model with HuggingFace Transformers (device_map='auto')\n"
        f"Datatype: {torch_dtype}"
    )

    trust_remote_code = "Qwen" in model_key
    model = AutoModelForCausalLM.from_pretrained(
        weights_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )

    # Qwen prints erroneous warnings; just use this to shut them off
    if "Qwen" in model_key:
        model.config.use_sliding_window = False

    model.eval()
    return model


def load_tokenizer(model_key: str) -> AutoTokenizer:
    """
    Load a tokenizer corresponding to the given model key

    Args:
        model_key: Identifier for the model (must exist in MODEL_INFO)

    Returns:
        AutoTokenizer: The loaded tokenizer instance
    """
    tokenizer_path = MODEL_INFO[model_key].path
    tokenizer_kwargs = {}

    if "huggyllama" in model_key:
        tokenizer_kwargs["legacy"] = False

    if "Qwen" in model_key:
        tokenizer_kwargs["trust_remote_code"] = True
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    return tokenizer
