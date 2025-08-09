import json, pickle
import os
import io
import base64
import numpy as np
import math

from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter, MaxNLocator
import matplotlib.ticker as ticker
import seaborn as sns

from script import utils, constants
from script.plots import np_extraction

###########################################################################
#
# Plotting utilities
#
###########################################################################

"""
Useful for taking list of dicts version of results and sorting by probability,
filtering out 0 probability
"""
def sort_dict_list_by_prob(data, descending=False):
    return sorted(data, key=lambda x: x["prob"], reverse=descending)

def filter_out_zero_prob(data):
    return [item for item in data if item["prob"] != 0]

"""
Load pz list for the specific config
"""
def load_config_pz_list(model, dataset, temperature, k, suffix_start):
    return load_config_out_list(model, dataset, temperature, k, suffix_start, get_batch_pz_list)

def load_greedy_pz_list(model, dataset, suffix_start):
    return load_config_pz_list(model, dataset, 1.0, 1, suffix_start)

def load_config_pz_meta_list(model, dataset, temperature, k, suffix_start):
    return load_config_out_list(model, dataset, temperature, k, suffix_start, get_batch_pz_meta_list)

def load_greedy_pz_meta_list(model, dataset, suffix_start):
    return load_config_pz_meta_list(model, dataset, 1.0, 1, suffix_start)

def load_config_out_list(model, dataset, temperature, k, suffix_start, getter): 
    # load manifest and list config dir; confirm that number of batches consisent
    manifest_path = utils.get_manifest_path_full(False, model, dataset)
    with open(manifest_path, 'r') as f:
        manifest_len = len(json.load(f))

    config_dir =  utils.get_config_dir_full(
        False,
        model,
        dataset,
        temperature,
        k,
        suffix_start
    )
    batch_files = [
        os.path.join(config_dir, f) 
        for f in os.listdir(config_dir) 
        if os.path.isfile(os.path.join(config_dir, f))
    ]

    assert len(batch_files) == manifest_len

    pz_list = []
    for bf in batch_files:
         batch_pz_list = getter(bf)
         pz_list.extend(batch_pz_list)
    return pz_list

"""
Helper to load saved probabilities for a batch at the specified path
"""
def get_batch_pz_list(batch_file):
    with open(batch_file, 'r') as f:
        batch_results = json.load(f)

    probs = []
    for val in batch_results['results'].values():
        probs.append(val['prob'])

    return probs

def get_batch_pz_meta_list(batch_file):
    with open(batch_file, 'r') as f:
        batch_results = json.load(f)

    # TODO get meta file associated with batch, read off there
    probs_meta = []
    for key, val in batch_results['results'].items():
        probs_meta.append({
            "hash_id": key,
            "prob": val['prob'],
            "prefix_text": val['prefix_text'],
            "suffix_text": val['suffix_text']
        })
    return probs_meta

"""
Helper for plotting and saving
"""
def save_and_show_plot(filename, fig=None, as_png=False):
    if fig is None:
        fig = plt.gcf()
    

    # Prepare buffer
    buf = io.BytesIO()
    if as_png:
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        mime = "image/png"
        filename = filename.replace(".pdf", ".png")
        fig.savefig(filename, bbox_inches="tight", dpi=300)
    else:
        fig.savefig(buf, format="pdf", bbox_inches="tight")
        mime = "application/pdf"
        fig.savefig(filename, bbox_inches="tight")

    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    download_link = f'<a href="data:{mime};base64,{image_base64}" download="{os.path.basename(filename)}">Download Plot</a>'
    
    display(HTML(download_link))
    plt.show()

def get_plot_name_base(model, dataset, temperature, k, suffix_start):
    plot_dir = utils.get_plot_dir(model, dataset)
    config_name = utils.get_config_name(temperature, k, suffix_start)
    d = dataset.replace(" ", "_")
    d = d.replace("'", "_")
    d = d.replace(".", "")
    base_filename = f"{plot_dir}/{model}-{d}-{config_name}"
    return base_filename

###############################################
# For heatmaps
    
# for the model, dataset, temperature,k, and suffix_start, get the
# probs for every example processed, as well as the start and end index
# within the book
def get_book_probs_by_idx(model, dataset, temperature, k, suffix_start):
    manifest_path = utils.get_manifest_path_full(False, model, dataset)
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    progress = {i for i in range(10, 101, 10)}
    sequences = []
    doc_len = None
    for i, batch in enumerate(manifest):
        percent_complete = int(100 * i / len(manifest))
        if percent_complete in progress:
            print(f"Progress: {percent_complete}%")
            progress.remove(percent_complete)
    
        batch_idx = batch['batch_idx']
        metadata_batch_path = utils.get_metadata_path_full(False, model, dataset, batch_idx)
        with open(metadata_batch_path, 'rb') as f:
            batch_metadata = pickle.load(f)
        
        manifest_batch_hashes = batch['hashes']
        
        assert list(batch_metadata['metadata'].keys()) == manifest_batch_hashes
        
        probs_batch_path = utils.get_probs_path_full(
            False, model, dataset, temperature, k, suffix_start, batch_idx)
        
        with open(probs_batch_path, 'r') as f:
            probs_batch = json.load(f)
        
        assert list(probs_batch['results'].keys()) == manifest_batch_hashes
        
        for h in manifest_batch_hashes:
            if doc_len == None:
                doc_len = batch_metadata['metadata'][h]['document_len']
            ex_metadata = batch_metadata['metadata'][h]
            start = ex_metadata['example_loc']['start_idx']
            end = ex_metadata['example_loc']['end_idx']
            prob = probs_batch['results'][h]['prob']
            sequences.append({'hash': h, 'start': start, 'end': end, 'prob': prob})

    return sequences, doc_len

def max_variable_length_probs(sequences, suffix_start, doc_length):
    max_probs = np.zeros(doc_length)

    for seq in sequences:
        start, end = seq['start'], seq['end']
        prob = seq['prob']
        
        end = min(end, doc_length)

        extract_start = start + suffix_start
        # keep track of max probs at different indexes
        max_probs[extract_start:end] = np.maximum(max_probs[extract_start:end], prob)

    return max_probs

def get_heatmap_fn(model, dataset, temperature, k, suffix_start):
    base_name = get_base_fn(model, dataset, temperature, k, suffix_start)
    base_name = f"{base_name}-window-heatmap.pdf"
    return base_name

# plot 1d heatmap of max probs
def plot_1d_heatmap(
    filename, max_probs, model, book, cmap='Blues', min_idx=None, max_idx=None, nbins=10
):
    if min_idx is None:
        min_idx = 0
    if max_idx is None:
        max_idx = len(max_probs)

    max_probs_slice = max_probs[min_idx:max_idx]

    plt.figure(figsize=(15, 1.7))
    im = plt.imshow([max_probs_slice], cmap=cmap, aspect='auto', interpolation='none', vmin=0, vmax=1)

    cbar = plt.colorbar(im, fraction=0.45, pad=0.02)
    cbar.set_label("Max. probability", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.xlabel("Book position (character)")
    # so that 0 idx and end idx are visible in the heatmap
    pad = max(1, max_idx // 100)
    plt.xlim(-pad, max_idx + pad)
    plt.yticks([])

    # Compute nice tick locations
    nice_step = int(np.ceil((max_idx - min_idx) / nbins / 1000.0)) * 1000  # round to nearest 1000
    ticks = list(range(((min_idx + nice_step - 1) // nice_step) * nice_step, max_idx, nice_step))
    if ticks[0] > min_idx:
        ticks = [min_idx] + ticks
    if ticks[-1] < max_idx:
        ticks.append(max_idx)

    tick_positions = [t - min_idx for t in ticks]
    plt.xticks(tick_positions, [str(t) for t in ticks])

    plt.title(f"{book}: Maximum per-character probability for {model}")
    plt.tight_layout()
    save_and_show_plot(filename, as_png=True)

def plot_1d_heatmap_science(
    filename, max_probs, model, book, cmap='Blues', min_idx=None, max_idx=None, nbins=10
):
    if min_idx is None:
        min_idx = 0
    if max_idx is None:
        max_idx = len(max_probs)

    max_probs_slice = max_probs[min_idx:max_idx]

    plt.figure(figsize=(15, 1.7))
    im = plt.imshow([max_probs_slice], cmap=cmap, aspect='auto', interpolation='none', vmin=0, vmax=1)

    #cbar = plt.colorbar(im, fraction=0.45, pad=0.02)
    #cbar.set_label("Max. probability", fontsize=12)
    #cbar.ax.tick_params(labelsize=12)

    #plt.xlabel("Book position (character)")
    # so that 0 idx and end idx are visible in the heatmap
    pad = max(1, max_idx // 100)
    plt.xlim(-pad, max_idx + pad)
    plt.yticks([])

    # Compute nice tick locations
    nice_step = int(np.ceil((max_idx - min_idx) / nbins / 1000.0)) * 1000  # round to nearest 1000
    ticks = list(range(((min_idx + nice_step - 1) // nice_step) * nice_step, max_idx, nice_step))
    if ticks[0] > min_idx:
        ticks = [min_idx] + ticks
    if ticks[-1] < max_idx:
        ticks.append(max_idx)

    tick_positions = [t - min_idx for t in ticks]
    plt.xticks(tick_positions, [str(t) for t in ticks], fontsize=16)

    #plt.title(f"{book}: Maximum per-character probability for {model}")
    plt.tight_layout()
    save_and_show_plot(filename, as_png=True)
