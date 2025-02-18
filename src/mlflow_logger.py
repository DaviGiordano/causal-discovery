import mlflow
import pathlib
from typing import Any, Dict
from flatten_dict import flatten
import logging

logger = logging.getLogger(__name__)
LOG_PATH = "./mlruns"


class MLflowLogger:
    def __init__(self, experiment_name: str):
        """
        Initialize MLflow logger with experiment name.

        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        try:
            mlflow.set_tracking_uri(LOG_PATH)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.error(f"Failed to set MLflow experiment: {str(e)}")
            raise

    def log_run(
        self, run_name: str, params: Dict, metrics: Dict, artifacts_dir: str
    ) -> None:
        """Log a single experiment run to MLflow."""
        try:
            with mlflow.start_run(run_name=run_name):
                self._log_params(params=params)
                self._log_metrics(metrics)
                self._log_plot_images(artifacts_dir)

                logger.info(f"Successfully logged run '{run_name}' to MLflow")

        except Exception as e:
            logger.error(f"Failed to log run to MLflow: {str(e)}")
            raise

    def _log_params(
        self,
        params: Dict[str, Any],
    ) -> None:
        """Log params."""
        for key, value in flatten(params, reducer="dot").items():
            mlflow.log_param(key, value)

    def _log_metrics(self, metrics_results: Dict[str, Any]) -> None:
        """Flatten a nested metrics dict and log each metric."""
        for key, value in flatten(metrics_results, reducer="dot").items():
            if not isinstance(value, (str, int, float)):
                logger.warning(
                    f"Skipped logging metric '{key}' as it is not a string or a real number."
                )
                continue
            mlflow.log_metric(key, value)

    def _log_plot_images(self, artifacts_dir: pathlib.Path) -> None:
        """Log PNG plot images from the specified directory."""
        if artifacts_dir.exists():
            for png_file in artifacts_dir.glob("*.png"):
                try:
                    mlflow.log_artifact(str(png_file))
                    logger.info(f"Logged plot: {png_file.name}")
                except Exception as e:
                    logger.error(f"Failed to log plot {png_file.name}: {str(e)}")
