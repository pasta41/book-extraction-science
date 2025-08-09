import logging
from torch.utils.data import DataLoader
import datetime
import os
import json

from script.memorization.loaders import data_loader
from script.memorization.loaders import model_loader

from script.memorization import execute

logger = logging.getLogger(__name__)
script_name = os.path.basename(__file__)

def run(args, paths):
    logger.info(f"Running {script_name}\n\n")
    start_time = datetime.datetime.now()
    logger.info(f"START TIME: {start_time}\n\n")

    if needs_force_to_run(paths, args.force):
        return

    model, loader = load_model_and_dataloader(paths, args)

    execute.compute_and_save_all_pz(
        loader, model, args.model, args.parsed_configs, paths
    )

    end_time = datetime.datetime.now()
    logger.info(f"END TIME: {end_time}\n\n")
    duration = end_time - start_time

    save_runtime(paths, duration)
    logger.info(f"TOTAL RUN TIME: {duration}")

def load_model_and_dataloader(paths, args):
    logger.info(f"Loading tokenizer for model: {args.model}")
    tokenizer = model_loader.load_tokenizer(args.model)
    logger.info(f"Loading book from: {paths.input_book_path}")
    book_dataset = data_loader.load_dataset(paths, args, tokenizer)
    logger.info(f"Loading {args.model}")
    model = model_loader.load_model(args.model, args.precision)
    loader = DataLoader(book_dataset, batch_size=args.batch_size, shuffle=False)
    return model, loader

def needs_force_to_run(paths, force_flag):
    manifest_file = paths.manifest_path()
    logger.info(f"Checking for manifest file at: {manifest_file}")
    if os.path.exists(manifest_file):
        logger.error(f"Manifest already exists at {manifest_file}.")
        if force_flag:
            logger.info("--force provided; clearing output directories and re-running")
            clear_output_dirs(paths)
            return False
        else:
            logger.error("Re-run with --force to re-compute. Exiting.")
            return True
    logger.info("Manifest does not exist.")
    return False

def clear_output_dirs(paths):
    import shutil
    dirs = [
        paths.manifest_dir(),
        paths.probs_dir(),
        paths.metadata_dir(),
        paths.runtime_dir()
    ]
    for directory in dirs:
        if os.path.exists(directory):
            logger.info(f"Removing directory: {directory}")
            shutil.rmtree(directory)

def save_runtime(paths, duration):
    path = paths.runtime_path()
    runtime_dict = {"runtime_seconds": duration.total_seconds()}
    with open(path, "w") as f:
        json.dump(runtime_dict, f, indent=2)
    logger.info(f"Saved runtime info to {path}")
