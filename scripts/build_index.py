"""Scan the prepared-batch directories once and cache the batch index.

Listing training/ (4.9M files) and validation/ (2.2M files) takes a few minutes
on Lustre, so every other entry point reads the cached index instead.

    python -m scripts.build_index
"""

from __future__ import annotations

import argparse
import time

from common.config import add_common_args, load_config, resolve
from common.utils import setup_logging
from data.index import build_index


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Build the batch-file index."))
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    index_dir = resolve(cfg.data.index_dir)
    logger = setup_logging(index_dir / "build_index.log")
    logger.info("root: %s", cfg.data.root)

    started = time.time()
    frames = build_index(cfg.data.root, index_dir, logger=logger)
    logger.info("done in %.0f s", time.time() - started)
    for split, frame in frames.items():
        logger.info("%s: %d batches, %d distinct stations",
                    split, len(frame), frame.loc[frame["kind"] == "regular", "station"].nunique())


if __name__ == "__main__":
    main()
