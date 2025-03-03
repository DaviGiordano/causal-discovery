import json
import pathlib
import logging
from typing import Any, Dict
from flatten_dict import flatten

logger = logging.getLogger(__name__)


def log_experiment_results(
    output_dir: pathlib.Path,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
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
        "params.json": flatten(params, reducer="dot"),
        "metrics.json": flatten(metrics, reducer="dot"),
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
