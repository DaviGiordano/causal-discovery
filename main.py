#!/usr/bin/env python
import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import pandas as pd
from src.causal_discovery import CausalDiscovery
from src.load_parse import load_yaml, parse_args
from src.logging_config import setup_logging
from src.pytetrad.TetradSearch import TetradSearch


def main() -> None:
    args = parse_args()
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    log_file = args.output.with_suffix(".log")

    setup_logging(str(log_file))
    logger = logging.getLogger("main")

    MAX_RETRIES = 2
    retry_count = 0

    while retry_count <= MAX_RETRIES:
        try:
            pipeline = CausalDiscovery(
                args.config,
                args.data,
                args.output,
                args.knowledge,
                args.metadata,
            )
            pipeline.run()
            break

        except Exception as err:
            retry_count += 1
            if retry_count <= MAX_RETRIES:
                logger.warning(
                    f"Pipeline failed (attempt {retry_count}/{MAX_RETRIES}): {err}. Retrying..."
                )
                time.sleep(2)  # Add a small delay before retrying
            else:
                logger.exception(f"Pipeline failed after {MAX_RETRIES} retries: {err}")
                sys.exit(1)


if __name__ == "__main__":
    main()
