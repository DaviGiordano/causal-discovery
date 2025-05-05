#!/usr/bin/env python
"""run_all.py
Utility to batch‑run *main.py* over every YAML configuration file in a folder.

Example
-------
python run_all.py \
    --configs configs/ \
    --data data/ruta/synth_normal_1000/train.csv \
    --output_root output/

For each `*.yaml` file found below *configs*, this script creates

    <output_root>/<config_stem>/graph.txt

and invokes

    python main.py --config <config> --data <data> --output <graph.txt>

A log file produced by *main.py* is automatically placed next to each graph.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

LOGGER = logging.getLogger(Path(__file__).stem)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_configs(root: Path) -> List[Path]:
    """Return a sorted list of *.yaml files under *root* (recursively)."""
    return sorted(root.rglob("*.yaml"))


def _run_single(config: Path, data: Path, output_root: Path, python_exe: str) -> None:
    """Execute *main.py* for a single algorithm config."""
    out_dir = output_root / config.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "graph.txt"

    cmd = [
        python_exe,
        "main.py",
        "--config",
        str(config),
        "--data",
        str(data),
        "--output",
        str(graph_path),
    ]

    LOGGER.info(
        "Running %s",
        " \
    ".join(
            cmd
        ),
    )
    completed = subprocess.run(cmd, capture_output=True, text=True)

    if completed.returncode != 0:
        LOGGER.error(
            "❌ %s failed (exit %s)\n%s",
            config.name,
            completed.returncode,
            completed.stderr,
        )
    else:
        LOGGER.info("✅ %s finished – graph saved to %s", config.name, graph_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch‑run main.py for every YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--configs",
        type=Path,
        required=True,
        help="Folder containing YAML configs (searched recursively)",
    )
    p.add_argument(
        "--data", type=Path, required=True, help="CSV dataset to feed all runs"
    )
    p.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Root folder where per‑config sub‑directories are created",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to invoke main.py with",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )

    configs = _discover_configs(args.configs)
    if not configs:
        LOGGER.error("No YAML configs found under %s", args.configs)
        sys.exit(1)

    LOGGER.info("Discovered %d configurations", len(configs))

    for cfg in configs:
        try:
            _run_single(cfg, args.data, args.output_root, args.python)
        except Exception as exc:
            LOGGER.exception("Unhandled exception while processing %s: %s", cfg, exc)


if __name__ == "__main__":
    main()
