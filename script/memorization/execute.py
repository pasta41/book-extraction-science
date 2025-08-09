import torch
import os
import numpy as np
from tqdm import tqdm
import logging
import datetime
import threading
import queue
import json

from script.memorization import inference

logger = logging.getLogger(__name__)

###########################################################################
#
# Wrapper for calling inference and post-processing logic
#
###########################################################################

"""
Takes a dataloader and a model, and performs inference. 
Saves logprobs, returns a manifest.
"""
def compute_and_save_all_pz(
    loader,
    model,
    model_key,
    configs,
    paths,
    num_workers=4,
):
    return inference.run_and_save(
        loader,
        model,
        model_key,
        configs,
        paths,
        save_worker_fn=pz_save_worker_fn,
        num_workers=num_workers,
    )

"""
Queue logic for saving logit files async
"""
def pz_save_worker_fn(save_queue):
    import datetime, logging
    from script.memorization import pz
    
    while True:
        item = save_queue.get()
        if item is None:
            break

        model_key, paths, config_item, batch, logits_batch, hashes, batch_idx = item

        metadata_path = paths.metadata_batch_path(batch_idx)

        start_time = datetime.datetime.now()
        batch_config = f"batch-config {batch_idx}-{config_item}"
        logger.debug(f"Computing and saving logprobs for {batch_config}: start")

        suffix_start_indexes = [config_item.suffix_start_index] * len(hashes)
        logprobs_list, probs_list = pz.compute_logprob_batch(
                batch,
                logits_batch,
                suffix_start_indexes,
                config_item.temperature,
                config_item.k
            )

        try:
            # TODO should be able to simplify this?
            # a lot of data that saving redundantly? For later
            _save_probs(
                model_key,
                paths,
                config_item,
                batch_idx,
                hashes,
                batch,
                logprobs_list,
                probs_list,
                metadata_path
            )
        except Exception as e:
            logger.exception(f"Error saving for {batch_config}: {e}")

        duration = datetime.datetime.now() - start_time
        logger.debug(
            f"Computing logprobs for {batch_config}: end (duration: {duration})"
        )

"""
save outputs
"""
def _save_probs(
    model_key,
    paths,
    config_item,
    batch_idx,
    expected_hashes,
    inputs,
    logprobs,
    probs,
    metadata_path,
):
    import os, json
    from script.memorization.loaders.model_loader import load_tokenizer

    probs_dir = paths.probs_dir()

    save_path = paths.probs_path_from_config(config_item, batch_idx)
    results_batch_record = {}

    results_batch_record['metadata_path'] = metadata_path
    results_batch_record['batch_idx'] = batch_idx

    results_batch_record['config'] = {}
    results_batch_record['config']['suffix_start_index'] = config_item.suffix_start_index
    results_batch_record['config']['temperature'] = config_item.temperature
    results_batch_record['config']['k'] = config_item.k

    results_batch_record['results'] = {}
    assert len(expected_hashes) == len(logprobs) and len(expected_hashes) == len(probs)
    # everything should be in the same order for the batch; just reorganizing

    tokenizer = load_tokenizer(model_key)

    for idx, (ids, mask) in enumerate(zip(inputs["input_ids"], inputs["attention_mask"])):
        h = expected_hashes[idx]

        results_batch_record['results'][h] = {}
        results_batch_record['results'][h]['prob'] = probs[idx]

        # check if defined (which might not be because of top-k
        logprob = logprobs[idx]
        if logprob == float('-inf'):
            logprob = None
        results_batch_record['results'][h]['logprob'] = logprob

        # we're not doing anything with padding, so this should be a no-op
        input_ids = ids[mask.bool()]
        prefix_ids = input_ids[:config_item.suffix_start_index]
        suffix_ids = input_ids[config_item.suffix_start_index:]
        prefix_decoded = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        suffix_decoded = tokenizer.decode(suffix_ids, skip_special_tokens=True)

        results_batch_record['results'][h]['prefix_ids'] = prefix_ids.tolist()
        results_batch_record['results'][h]['prefix_text'] = prefix_decoded
        results_batch_record['results'][h]['suffix_ids'] = suffix_ids.tolist()
        results_batch_record['results'][h]['suffix_text'] = suffix_decoded

    with open(save_path, "w") as f:
        json.dump(results_batch_record, f, indent=2)
    logger.info(f"Results for batch {batch_idx} written to {save_path}")

