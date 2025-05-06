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
from src.default_params import BOOTSTRAP_PRESETS, SCORE_METHODS, TEST_METHODS


def main() -> None:
    args = parse_args()
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    log_file = args.output.with_suffix(".log")

    setup_logging(str(log_file))
    logger = logging.getLogger("main")

    try:
        pipeline = CausalDiscovery(
            args.config,
            args.data,
            args.output,
            args.knowledge,
            args.metadata,
        )
        pipeline.run()
    except Exception as err:
        logger.exception("Pipeline failed: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
