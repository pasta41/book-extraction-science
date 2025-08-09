"""
Utilities for computing logprobs of suffixes conditioned on prefixes
within a batch of tokenized inputs and logits.
"""

from typing import List, Tuple, Optional, Dict

import torch


def compute_logprob_batch(
    inputs_batch: Dict[str, torch.Tensor],
    logits_batch: torch.Tensor,
    suffix_start_indexes: List[int],
    temperature: float = 1.0,
    top_k: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Takes a batch of examples (inputs_batch), the associated logits tensor
    (logits_batch), and suffix_start_indexes for processing.

    Computes the logprobs of each suffix (starting at suffix_start_index)
    conditioned on the corresponding prefix.

    Arguments:
        inputs_batch: A dict with "input_ids" and "attention_mask", both
                      shaped [batch_size, seq_len]
        logits_batch: A tensor of shape [batch_size, seq_len, vocab_size], 
                      raw logits from the model
        suffix_start_indexes: A list of start indexes for suffixes, one 
                              per example in the batch. They're all the same;
                              clean this up later
        temperature: temparature scaling factor for logits; defaults to 1.0
                     (no scaling)
        top_k: Optional int. If set, zeroes out all logits except the top_k
               highest per position (normalization happens inside log_softmax)

    Returns:
        log_probs_list: List of total logprobs (one per example)
        prob_list: List of total probabilities (exp(logprob), one per example)
    """
    
    import torch.nn.functional as F
    import math

    # shape: [batch_size, seq_len]
    input_ids = inputs_batch["input_ids"]  
    attention_mask = inputs_batch["attention_mask"]

    assert input_ids.device.type == 'cpu'
    assert attention_mask.device.type == 'cpu'

    assert len(suffix_start_indexes) == input_ids.shape[0]

    if temperature != 1.0 or top_k is not None:
        logits_batch = apply_temperature_and_topk(
            logits_batch,
            temperature,
            top_k
        )
    
    # dim=-1, across the vocab, all possible next-token preds at each seq position
    # log_probs[b, i, j]: for example at batch index b, log probs of predicting 
    # token j at position i in seq for
    # shape: [batch_size, seq_len, vocab_size]
    log_probs = F.log_softmax(logits_batch, dim=-1)

    log_probs_list = []
    prob_list = []
    batch_size = input_ids.shape[0]

    # for each example
    for batch_b in range(batch_size):
        suffix_start = suffix_start_indexes[batch_b]
        input_ids_b = input_ids[batch_b]

        # shape [seq_len, vocab_size]
        log_probs_b = log_probs[batch_b]
        
        real_seq_len = attention_mask[batch_b].sum().item()
        
        # Compute log prob for each token in suffix (conditioned on prefix)
        total_log_prob = 0.0
        # this can happen from top-k processing; instead of just ignoring
        # infs, we are doing this so we can track data about stuff that
        # was extractable under temperature sampling but perhaps became
        # non-extractable under top-k
        has_impossible_token = False
        for i in range(suffix_start - 1, real_seq_len - 1):
            # get actual next token in suffix
            actual_next_token_id = input_ids_b[i + 1]
            # get log prob for actual_next_token (at i+1), based on context up to i
            log_prob = log_probs_b[i, actual_next_token_id].item()

            # -inf from top-k filtering
            if not math.isfinite(log_prob):
                has_impossible_token = True
                break

            total_log_prob += log_prob

        if has_impossible_token:
            total_log_prob = float('-inf')
            prob = 0.0
        else:
            prob = math.exp(total_log_prob)

        log_probs_list.append(total_log_prob)
        prob_list.append(prob)
        
    return log_probs_list, prob_list


def apply_temperature_and_topk(
    logits_batch: torch.Tensor,
    temperature: float,
    top_k: Optional[int]
) -> torch.Tensor:
    """
    Post-processes logits with temperature scaling and optional top-k filtering

    Arguments:
        logits_batch: Tensor of shape [batch_size, seq_len, vocab_size]
        temperature: Temperature to scale logits by. If 1.0, no effect
        top_k: If set, only the top_k logits per position are kept; others set to -inf

    Returns:
        A new tensor of same shape with modified logits.
    """

    import torch
    if temperature != 1.0:
        logits_batch = logits_batch / temperature

    if top_k is not None:
        #         [batch, seq_len, top_k]   selects along vocab [batch, seq_len] -> [batch, seq_len, 1]
        kth_vals = torch.topk(logits_batch, top_k, dim=-1).values[:, :, -1].unsqueeze(-1)
        mask = logits_batch < kth_vals
        logits_batch = logits_batch.masked_fill(mask, float('-inf'))

    return logits_batch
