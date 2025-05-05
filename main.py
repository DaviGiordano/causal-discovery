#!/usr/bin/env python
"""main.py - run causal discovery with Tetrad on a CSV dataset.

Usage
-----
python main.py --config CONFIG.yaml --data DATA.csv --output graph.txt [--knowledge knowledge.txt]

Flat YAML example::

    algorithm_name: fges
    score_name: conditional_gaussian_score
    score_params:
        penalty_discount: 2
    bootstrap_strategy: bootstrap100
    num_threads: 4

Any section may be omitted; hard-coded defaults will be applied. The script logs
all resolved parameters as well as the total runtime of the search. If a
knowledge file is supplied it is loaded into Tetrad **before** the search and
its contents are echoed in the log.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Project-local imports (package layout enforced by src/)
# ---------------------------------------------------------------------------
try:
    from src.logging_config import setup_logging
    from src.pytetrad.TetradSearch import TetradSearch
except ImportError as exc:  # pragma: no cover - helpful error
    print(
        "Error: project modules not importable - have you installed the package?", exc
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Canonical defaults (tests, scores, bootstrapping)
# ---------------------------------------------------------------------------
TEST_METHODS: Dict[str, Dict[str, Any]] = {
    "fisherz": {
        "func_name": "use_fisher_z",
        "default_params": {
            "alpha": 0.01,
            "use_for_mc": False,
            "singularity_lambda": 0.0,
        },
    },
    "conditional_gaussian_test": {
        "func_name": "use_conditional_gaussian_test",
        "default_params": {
            "alpha": 0.01,
            "discretize": True,
            "use_for_mc": False,
        },
    },
}

SCORE_METHODS: Dict[str, Dict[str, Any]] = {
    "conditional_gaussian_score": {
        "func_name": "use_conditional_gaussian_score",
        "default_params": {
            "penalty_discount": 1,
            "discretize": True,
            "num_categories_to_discretize": 3,
            "structure_prior": 0,
        },
    },
}

BOOTSTRAP_PRESETS: Dict[str, Dict[str, Any]] = {
    "bootstrap100": {
        "numberResampling": 10,
        "percent_resample_size": 100,
        "with_replacement": True,
        "add_original": True,
        "resampling_ensemble": 1,
        "seed": 42,
    },
    "jackknife90": {
        "numberResampling": 10,
        "percent_resample_size": 90,
        "with_replacement": False,
        "add_original": True,
        "resampling_ensemble": 1,
        "seed": 42,
    },
}

__all__ = ["CausalDiscoveryPipeline", "main"]

# ---------------------------------------------------------------------------
# YAML helper - flat configs only
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)
    if "algorithm_name" not in cfg:
        raise ValueError("Config YAML must be flat and include 'algorithm_name'.")
    return cfg


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class CausalDiscoveryPipeline:
    """End-to-end orchestration: load data, configure Tetrad, run, and save."""

    def __init__(
        self,
        cfg_path: Path,
        data_path: Path,
        output_path: Path,
        knowledge_path: Optional[Path] = None,
    ):
        self.cfg = _load_yaml(cfg_path)
        self.data = self._read_data(data_path)
        self.output_path = output_path
        self.knowledge_path = knowledge_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search: Optional[TetradSearch] = None

        # Number of worker threads (default 1)
        self.num_threads: int = int(self.cfg.get("num_threads", 1))

        # Store resolved config components for logging
        self.final_test: Optional[Dict[str, Any]] = None
        self.final_score: Optional[Dict[str, Any]] = None
        self.final_bootstrap: Optional[Dict[str, Any]] = None
        self.final_knowledge: Optional[Dict[str, str]] = None
        self.elapsed_seconds: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._build_and_execute()
        self._save_graph()
        if self.elapsed_seconds is not None:
            self.logger.info("Total search time: %.2f s", self.elapsed_seconds)
        self.logger.info("Graph written to %s", self.output_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_data(path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except Exception as err:
            raise RuntimeError(f"Unable to read dataset '{path}': {err}") from err
        if df.empty:
            raise ValueError("Dataset contains no rows.")
        return df

    @staticmethod
    def _configure_component(
        search: TetradSearch,
        name: Optional[str],
        user_params: Optional[Dict[str, Any]],
        registry: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not name:
            return None
        key = name.lower()
        if key not in registry:
            raise ValueError(f"Unsupported method '{name}'. Choices: {list(registry)}")
        meta = registry[key]
        final_params = {**meta["default_params"], **(user_params or {})}
        getattr(search, meta["func_name"])(**final_params)
        return {"name": key, **final_params}

    @staticmethod
    def _configure_bootstrap(
        search: TetradSearch, strategy: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not strategy:
            return None
        key = strategy.lower()
        if key not in BOOTSTRAP_PRESETS:
            raise ValueError(
                f"Bootstrap strategy '{strategy}' not implemented. Choices: {list(BOOTSTRAP_PRESETS)}"
            )
        params = BOOTSTRAP_PRESETS[key]
        search.set_bootstrapping(**params)
        return {"strategy": key, **params}

    # ..................................................................
    def _apply_knowledge(self) -> None:
        """Load domain background knowledge from file and inject into the search."""
        if self.knowledge_path is None:
            return
        if not self.knowledge_path.exists():
            raise FileNotFoundError(
                f"Knowledge file '{self.knowledge_path}' does not exist."
            )
        try:
            # Populate the Knowledge object inside TetradSearch
            self.search.load_knowledge(str(self.knowledge_path))  # type: ignore[attr-defined]
        except Exception as err:
            raise RuntimeError(
                f"Unable to load knowledge file '{self.knowledge_path}': {err}"
            ) from err

        # Store the raw file contents for logging
        try:
            content = self.knowledge_path.read_text(encoding="utf-8").strip()
        except Exception:
            content = "<unavailable>"
        self.final_knowledge = {
            "path": str(self.knowledge_path),
            "content": content,
        }

    # ..................................................................
    def _apply_threading_parameter(self) -> None:
        """Set NUM_THREADS in the underlying Java Params object when available."""
        try:
            from edu.cmu.tetrad.util import Params  # JVM already loaded

            self.search.params.set(  # type: ignore[attr-defined]
                Params.NUM_THREADS, self.num_threads
            )
        except Exception as err:
            # Warn but carry on; the algorithm will just use Tetrad's default
            self.logger.warning("Setting NUM_THREADS failed or unsupported: %s", err)

    # ..................................................................
    def _build_and_execute(self) -> None:
        self.search = TetradSearch(self.data)
        self._apply_threading_parameter()

        # --------------------------------------------------------------
        # 1. Knowledge (must be set *before* algorithm configuration)
        # --------------------------------------------------------------
        self._apply_knowledge()

        # 2. Optional building blocks (test / score / bootstrap)
        self.final_test = self._configure_component(
            self.search,
            self.cfg.get("test_name"),
            self.cfg.get("test_params"),
            TEST_METHODS,
        )
        self.final_score = self._configure_component(
            self.search,
            self.cfg.get("score_name"),
            self.cfg.get("score_params"),
            SCORE_METHODS,
        )
        self.final_bootstrap = self._configure_bootstrap(
            self.search, self.cfg.get("bootstrap_strategy")
        )

        # Log full configuration
        self.logger.info("\n===== Final configuration =====")
        self.logger.info("Algorithm: %s", self.cfg["algorithm_name"].lower())
        self.logger.info("Threads: %s", self.num_threads)
        if self.final_knowledge:
            self.logger.info("Knowledge file: %s", self.final_knowledge["path"])
            self.logger.info("Knowledge contents:\n%s", self.final_knowledge["content"])
        if self.final_test:
            self.logger.info("Test: %s", self.final_test)
        if self.final_score:
            self.logger.info("Score: %s", self.final_score)
        if self.final_bootstrap:
            self.logger.info("Bootstrap: %s", self.final_bootstrap)
        if not any([self.final_test, self.final_score]):
            self.logger.info(
                "(no test/score component configured - algorithm handles it internally)"
            )

        algo = self.cfg["algorithm_name"].lower()
        algo_params = self.cfg.get("algorithm_params", {})
        self.logger.info(
            "Running algorithm '%s' with params: %s", algo, algo_params or "<defaults>"
        )

        # Time the execution
        start = time.perf_counter()
        self._run_algorithm(self.search, algo, algo_params)
        self.elapsed_seconds = time.perf_counter() - start
        self.logger.info("Search finished in %.2f s", self.elapsed_seconds)

    # ..................................................................
    @staticmethod
    def _run_algorithm(search: TetradSearch, name: str, params: Dict[str, Any]) -> None:
        if name == "pc":
            search.run_pc(**params)
        elif name == "fges":
            search.run_fges(**params)
        elif name == "fci":
            search.run_fci(**params)
        elif name == "grasp":
            search.run_grasp(**params)
        elif name == "boss":
            search.run_boss(**params)
        elif name == "dagma":
            search.run_dagma(**params)
        elif name == "directlingam":
            search.run_direct_lingam(**params)
        else:
            raise ValueError(f"Unsupported algorithm_name '{name}'.")

    # ..................................................................

    def _save_graph(self) -> None:
        """Write both Tetrad’s native graph string **and** a plain DOT file."""
        graph_str = str(self.search.java)  # Tetrad multi-section format
        dot_str = self.search.get_dot()  # Clean DOT for GUI tools

        dot_path = self.output_path.with_suffix(".dot")

        try:
            with self.output_path.open("w", encoding="utf-8") as fh:
                fh.write(graph_str)
            with dot_path.open("w", encoding="utf-8") as fh_dot:
                fh_dot.write(dot_str)
        except Exception as err:
            raise RuntimeError(
                f"Unable to write graph outputs ('{self.output_path}', '{dot_path}'): {err}"
            ) from err

        # Record where the extra file went
        self.logger.info("DOT graph written to %s", dot_path)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run causal discovery with Tetrad",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="YAML configuration file"
    )
    parser.add_argument(
        "--data", required=True, type=Path, help="CSV dataset (header row required)"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="File to write the resulting graph string",
    )
    parser.add_argument(
        "--knowledge",
        required=False,
        type=Path,
        help="Optional Tetrad knowledge file (.txt) with tiers / forbidden / required edges",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Ensure output directory exists before configuring logging so the log file can be created there
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    log_file = args.output.with_suffix(".log")
    try:
        setup_logging(str(log_file))
    except Exception:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
        )

    logger = logging.getLogger("main")
    try:
        pipeline = CausalDiscoveryPipeline(
            args.config, args.data, args.output, args.knowledge
        )
        pipeline.run()
    except Exception as err:
        logger.exception("Pipeline failed: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
