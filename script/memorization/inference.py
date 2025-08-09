import threading
import datetime
import json
import queue
from tqdm import tqdm
import torch
import logging
import pickle

logger = logging.getLogger(__name__)

# TODO maybe refactor with run.py 

###########################################################################
#
# Utilities for doing inference and saving different outputs and metadata
#
###########################################################################
"""
Run model inference with device-aware input handling.
batch is a dict of tensors with keys "input_ids", "attention_mask"
Return model output.
"""
def run_model_inference(model, batch):
    # assumes device_map being used for model parallelism
    # hardware resources were too limited to get benefits of data parallelism
    inputs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")}
    inputs = {k: v.to(model.device) for k,v in inputs.items()}
    # shape: [batch_size, seq_len, vocab_size]
    return model(**inputs).logits.detach().cpu()
    
"""
Runner logic that takes a DataLoader (loader), causal model (model), and
args to run inference.

Saves metadata (using save_metadata_fn) and logprobs asynchronously (using the 
# specified num_workers, save_worker_fn,
enqueue_fn). 
"""
def run_and_save(
    loader,
    model,
    model_key,
    configs,
    paths,
    save_worker_fn,
    num_workers=4
):
    manifest = []

    # create the queue and workers to do asynchronous file saving
    save_queue = queue.Queue()
    workers = [
        threading.Thread(target=save_worker_fn, args=(save_queue,), daemon=True)
        for _ in range(num_workers)
    ]
    for w in workers:
        w.start()

    with torch.inference_mode():
        for batch_idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Running inference"):
            batch_start_time = datetime.datetime.now()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            example_metadata_batch = batch["metadata"]

            logits_tensor_batch = run_model_inference(model, batch)

            if len(example_metadata_batch['hash']) != logits_tensor_batch.size(0):
                logger.error("Mismatch: metadata hashes and logits batch size")
                continue

            batch_idx_meta = batch_idx

            # remove data from gpu to do data and metadata processing on cpu
            input_ids = input_ids.detach().cpu()
            attention_mask = attention_mask.detach().cpu()
            batch["input_ids"] = input_ids
            batch["attention_mask"] = attention_mask

            hashes = example_metadata_batch['hash']

            # cheap; can do this in a blocking fashion
            meta_path = _save_metadata(
                paths,
                batch_idx_meta,
                example_metadata_batch,
                input_ids,
                attention_mask
            )

            # dispatch to asynchronous computation of logprobs
            _logprobs_enqueue_fn(
                save_queue, 
                configs,
                model_key,
                paths,
                batch_idx_meta, 
                batch, 
                logits_tensor_batch, 
                hashes
            )
            
            del input_ids, attention_mask, logits_tensor_batch

            manifest_record = {
                "batch_idx": batch_idx_meta,
                "metadata_path": meta_path,
                "hashes": hashes
            }
            manifest.append(manifest_record)

    for _ in range(num_workers):
        save_queue.put(None)
    for w in workers:
        w.join()

    manifest_path = paths.manifest_path()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")

    return manifest

def _logprobs_enqueue_fn(
    queue, configs, model_key, paths, batch_idx, batch, logits, hashes
):
    for config_item in configs:
        queue.put((model_key, paths, config_item, batch, logits.clone(), hashes, batch_idx))

"""
Utility for saving batch metadata
"""
def _save_metadata(
    paths,
    batch_idx,
    batch_metadata,
    input_ids,
    attention_mask,
):
    logger.debug(f"Saving batch {batch_idx} dataset metadata")
    save_path = paths.metadata_batch_path(batch_idx)
    # reorg metadata into a dictionary keyed by hash values, also include 
    # tensor of tokenized examples. This is gross, but we need it because the
    # Pythia tokenizer is non-deterministic, so can't just go off of re-
    # tokenizing full_text
    reorganized_dict = {}

    # TODO, in the future save logits or last layer

    reorganized_dict['inputs'] = {
        "input_ids": input_ids, 
        "attention_mask": attention_mask
    }
    reorganized_dict['metadata'] = {}
    for i, h in enumerate(batch_metadata['hash']):
        reorganized_dict['metadata'][h] = {
            # CAUTION; DO NOT RELY ON full_text;
            # Seeing non-deterministic decodings with Pythia tokenizer
            'text': batch_metadata['text'][i],
            'batch_inputs_idx': i,
            # TODO rename later; these are old names/ plots depend on them
            'dataset': batch_metadata['book_name'][i],
            'document': batch_metadata['book_name'][i],
            'document_len': batch_metadata['book_len'][i].item(),
            'example': batch_metadata['example'][i].item(),
            'example_len': batch_metadata['example_len'][i].item(),
            'token_len': int(batch_metadata['token_len'][i]),
            'example_loc': {
                'start_idx': int(batch_metadata['example_loc']['start_idx'][i]),
                'end_idx': int(batch_metadata['example_loc']['end_idx'][i]),
            },
            'task': batch_metadata['task'][i]
        }

    with open(save_path, "wb") as f:
        pickle.dump(reorganized_dict, f)

    logger.debug(f"Saved to {save_path}")
    return save_path
