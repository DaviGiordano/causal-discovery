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
from src.load_parse import load_data, load_yaml
from src.logging_config import setup_logging
from src.pytetrad.TetradSearch import TetradSearch
from src.default_params import BOOTSTRAP_PRESETS, SCORE_METHODS, TEST_METHODS

logger = logging.getLogger(__name__)


class CausalDiscovery:
    """End-to-end orchestration: load data, configure Tetrad, run, and save."""

    def __init__(
        self,
        configuration_path: Path,
        data_path: Path,
        output_path: Path,
        knowledge_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        logger.info("Initializing CausalDiscovery")
        logger.info("Configuration: %s", configuration_path)
        logger.info("Data: %s", data_path)
        logger.info("Output: %s", output_path)
        if knowledge_path:
            logger.info("Knowledge: %s", knowledge_path)
        if metadata_path:
            logger.info("Metadata: %s", metadata_path)

        self.configuration = load_yaml(configuration_path)
        self.data = load_data(data_path, metadata_path)
        self.output_path = output_path
        self.knowledge_path = knowledge_path
        self.search: Optional[TetradSearch] = None

        self.num_threads: Optional[int] = None
        if "num_threads" in self.configuration:
            self.num_threads = int(self.configuration["num_threads"])

        self.final_test: Optional[Dict[str, Any]] = None
        self.final_score: Optional[Dict[str, Any]] = None
        self.final_bootstrap: Optional[Dict[str, Any]] = None
        self.final_knowledge: Optional[Dict[str, str]] = None
        self.elapsed_seconds: Optional[float] = None

    def run(self) -> None:
        """Run the causal discovery process end-to-end."""
        start_time = time.perf_counter()

        self._build_and_execute()

        total_time = time.perf_counter() - start_time
        self.elapsed_seconds = total_time

        logger.info("Analysis complete in %.2f seconds", total_time)
        logger.info("Output written to %s", self.output_path)
        logger.info("DOT graph written to %s", self.output_path.with_suffix(".dot"))

    def _build_and_execute(self) -> None:
        """Configure the search and execute the algorithm."""
        self.search = TetradSearch(self.data)

        self._configure_search()

        # self._log_configuration()

        logger.info("Running algorithm: %s", self.configuration["algorithm_name"])
        start = time.perf_counter()
        self._run_algorithm(
            self.search,
            self.configuration["algorithm_name"].lower(),
            self.configuration.get("algorithm_params", {}),
        )
        self.elapsed_seconds = time.perf_counter() - start
        logger.info(
            "Algorithm execution completed in %.2f seconds", self.elapsed_seconds
        )

        # Save results
        logger.info("Saving graph results")
        self._save_graph(self.search, self.output_path)

    def _configure_search(self) -> None:
        """Apply all configurations to the search instance."""
        # Apply threading parameter
        if self.num_threads is not None:
            self._apply_threading_parameter(self.search, self.num_threads)

        # Configure knowledge
        if self.knowledge_path:
            self.final_knowledge = self._configure_knowledge(
                self.search, self.knowledge_path
            )

        # Configure test component
        test_name = self.configuration.get("test_name")
        if test_name:
            self.final_test = self._configure_component(
                self.search,
                test_name,
                self.configuration.get("test_params"),
                TEST_METHODS,
            )

        # Configure score component
        score_name = self.configuration.get("score_name")
        if score_name:
            logger.info("Configuring score component: %s", score_name)
            self.final_score = self._configure_component(
                self.search,
                score_name,
                self.configuration.get("score_params"),
                SCORE_METHODS,
            )

        # Configure bootstrap
        bootstrap_strategy = self.configuration.get("bootstrap_strategy")
        if bootstrap_strategy:
            logger.info("Configuring bootstrap strategy: %s", bootstrap_strategy)
            self.final_bootstrap = self._configure_bootstrap(
                self.search,
                bootstrap_strategy,
            )

    @staticmethod
    def _apply_threading_parameter(
        search: TetradSearch,
        num_threads: Optional[int] = None,
    ) -> None:
        """Set NUM_THREADS in the underlying Java Params object when available."""
        if num_threads is None:
            return  # Don't set if not explicitly configured

        try:
            from edu.cmu.tetrad.util import Params

            search.params.set(Params.NUM_THREADS, num_threads)
            logger.info("Successfully set NUM_THREADS to %d", num_threads)
        except Exception as err:
            logger.warning("Setting NUM_THREADS failed or unsupported: %s", err)

    @staticmethod
    def _configure_knowledge(
        search: TetradSearch,
        knowledge_path: Optional[Path],
    ) -> Optional[Dict[str, str]]:
        """Load domain background knowledge from file and inject into the search."""
        if knowledge_path is None:
            return None

        if not knowledge_path.exists():
            logger.error("Knowledge file not found: %s", knowledge_path)
            raise FileNotFoundError(
                f"Knowledge file '{knowledge_path}' does not exist."
            )

        try:
            search.load_knowledge(str(knowledge_path))
            content = knowledge_path.read_text(encoding="utf-8").strip()
            logger.info("Successfully loaded knowledge from %s", knowledge_path)
            logger.info("Knowledge content:\n%s", content)
        except Exception as err:
            logger.error("Failed to load knowledge: %s", err)
            raise RuntimeError(
                f"Unable to load knowledge file '{knowledge_path}': {err}"
            ) from err

        final_knowledge = {
            "path": str(knowledge_path),
            "content": content,
        }
        return final_knowledge

    @staticmethod
    def _configure_component(
        search: TetradSearch,
        name: Optional[str],
        user_params: Optional[Dict[str, Any]],
        registry: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Configure a test or score component with the given parameters."""
        if not name:
            return None

        key = name.lower()
        if key not in registry:
            logger.error("Invalid component: %s not in %s", key, list(registry.keys()))
            raise ValueError(f"Unsupported method '{name}'. Choices: {list(registry)}")

        meta = registry[key]
        final_params = {**meta["default_params"], **(user_params or {})}

        logger.info("Configuring %s with params: %s", meta["func_name"], final_params)
        getattr(search, meta["func_name"])(**final_params)

        return {"name": key, **final_params}

    @staticmethod
    def _configure_bootstrap(
        search: TetradSearch, strategy: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Configure bootstrap settings based on predefined strategies."""
        if not strategy:
            return None

        key = strategy.lower()
        if key not in BOOTSTRAP_PRESETS:
            logger.error("Invalid bootstrap strategy: %s", key)
            raise ValueError(
                f"Bootstrap strategy '{strategy}' not implemented. Choices: {list(BOOTSTRAP_PRESETS)}"
            )

        params = BOOTSTRAP_PRESETS[key]
        logger.info("Configuring bootstrap with params: %s", params)
        search.set_bootstrapping(**params)

        return {"strategy": key, **params}

    @staticmethod
    def _run_algorithm(search: TetradSearch, name: str, params: Dict[str, Any]) -> None:
        """Execute the selected causal discovery algorithm."""
        logger.info("Running algorithm %s with params: %s", name, params)

        try:
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
                logger.error("Unsupported algorithm: %s", name)
                raise ValueError(f"Unsupported algorithm_name '{name}'.")

            logger.info("Algorithm %s completed successfully", name)
        except Exception as err:
            logger.error("Algorithm %s failed: %s", name, err)
            raise

    @staticmethod
    def _save_graph(
        search: TetradSearch,
        output_path: Path,
    ) -> None:
        """Write both Tetrad's native graph string **and** a plain DOT file."""
        logger.info("Preparing graph output")
        graph_str = str(search.java)  # Tetrad multi-section format
        dot_str = search.get_dot()  # Clean DOT for GUI tools

        dot_path = output_path.with_suffix(".dot")

        try:
            logger.info("Writing Tetrad graph to %s", output_path)
            with output_path.open("w", encoding="utf-8") as fh:
                fh.write(graph_str)

            logger.info("Writing DOT graph to %s", dot_path)
            with dot_path.open("w", encoding="utf-8") as fh_dot:
                fh_dot.write(dot_str)

            logger.info("Successfully saved graph output")
        except Exception as err:
            logger.error("Failed to write graph output: %s", err)
            raise RuntimeError(
                f"Unable to write graph outputs ('{output_path}', '{dot_path}'): {err}"
            ) from err
