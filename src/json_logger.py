import json
import pathlib
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def log_experiment_results(
    output_dir: pathlib.Path,
    algorithm_params: Dict[str, Any],
    data_params: Dict[str, Any],
    metrics_results: Dict[str, Any],
) -> None:
    """
    Log experiment parameters and results as JSON files in the output directory.

    Args:
        output_dir: Directory where to save the log files
        algorithm_params: Dictionary containing algorithm parameters
        data_params: Dictionary containing data parameters
        metrics_results: Dictionary containing metrics results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log files
    log_files = {
        "algorithm_params.json": algorithm_params,
        "data_params.json": data_params,
        "metrics_results.json": metrics_results,
    }

    # Save each log file
    for filename, data in log_files.items():
        file_path = output_dir / filename
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Logged {filename} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to log {filename}: {str(e)}")
