import logging
import tyro
import os
from dataclasses import dataclass, field
from typing import Optional

from script.memorization.config import ConfigListInput
from script.memorization.runner import run
from script.memorization.utils.paths import Paths

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###########################################################################
#
# Entry point for running probabilistic extraction procedure with a sliding
# window for a specified model on a specified book.
#
###########################################################################

# Default project root directory
DEFAULT_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

@dataclass
class Args:
    root_path: Optional[str] = field(
        default=DEFAULT_ROOT_DIR,
        metadata={"help": "Root directory containing input and output folders"}
    )
    in_path: str = "data"
    out_path: str = "outputs"
    book: str = "Down and Out in the Magic Kingdom.txt"
    model: str = "Llama-3.1-8b"
    precision: int = field(
        default=16,
        metadata={"help": "Precision must be 16 or 32"}
    )
    ex_len: int = 100
    batch_size: int = 1000
    stride: int = 10
    force: bool = False

    configs: str = field(
        default="top40-basic",
        metadata={"help": "yaml path or comma/space-separated preset names"}
    )

    parsed_configs: ConfigListInput = field(init=False)

    def __post_init__(self):
        if self.precision not in {16, 32}:
            raise ValueError(f"Precision must be 16 or 32. Got {self.precision}")
        self.parsed_configs = ConfigListInput(self.configs)


def main():
    args = tyro.cli(Args)

    paths = Paths(
        root_dir=args.root_path,
        in_path=args.in_path,
        out_path=args.out_path,
        model=args.model,
        book=args.book,
    )

    logger.info("Run configuration:")
    logger.info(f"  Root directory      : '{args.root_path}'")
    logger.info(f"  Input directory     : '{paths.input_root}'")
    logger.info(f"  Output directory    : '{paths.output_root}'")
    logger.info(f"  Book (filename)     : '{args.book}'")
    logger.info(f"  Input book path     : '{paths.input_book_path}'")
    logger.info(f"  Model               : {args.model}")
    logger.info(f"  Precision           : {args.precision}")
    logger.info(f"  Example length      : {args.ex_len}")
    logger.info(f"  Batch size          : {args.batch_size}")
    logger.info(f"  Stride              : {args.stride}")
    logger.info(f"  Force recompute     : {args.force}")
    logger.info(f"  Configs:")
    for config in args.parsed_configs:
        logger.info(f"    - {config}")

    logger.info(f"Manifest directory   : {paths.manifest_dir()}")
    logger.info(f"Probs directory      : {paths.probs_dir()}")
    logger.info(f"Metadata directory   : {paths.metadata_dir()}")
    logger.info(f"Runtime directory    : {paths.runtime_dir()}")

    run(args, paths)


if __name__ == "__main__":
    main()
