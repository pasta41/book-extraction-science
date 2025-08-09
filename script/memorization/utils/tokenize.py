import torch
import logging

logger = logging.getLogger(__name__)

# Tokenization utility that wraps model-specific logic

def needs_manual_bos(model_name: str) -> bool:
    """
    Returns True if the model requires manual BOS token prepending.
    Currently applies to Pythia models.
    """
    return "pythia" in model_name.lower()

def tokenize_custom(data, tokenizer, model_name, padding=False):
    """
    Tokenizes input data using the given tokenizer.
    If the model requires manual BOS tokens, they are prepended.
    """
    if needs_manual_bos(model_name):
        logger.debug("Adding BOS token to start of sequence (manual mode)")
        inputs = tokenizer(
            data,
            padding=padding,
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cpu")

        bos_tokens = torch.full(
            # batch_size, 1 token to pretend to each
            (inputs["input_ids"].size(0), 1),
            tokenizer.bos_token_id,
            dtype=inputs["input_ids"].dtype,
            device=inputs["input_ids"].device,
        )

        input_ids = torch.cat([bos_tokens, inputs["input_ids"]], dim=1)
        attention_mask = torch.cat(
            [torch.ones_like(bos_tokens), inputs["attention_mask"]], dim=1)

        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask

    else:
        inputs = tokenizer(
            data,
            padding=padding,
            return_tensors="pt",
            add_special_tokens=True,
        ).to("cpu")

    return inputs

