import mlflow
import pathlib
from typing import Any, Dict
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
        self,
        run_name: str,
        algorithm_params: Dict[str, Any],
        data_params: Dict[str, Any],
        metrics_results: Dict[str, Any],
        artifacts_dir: pathlib.Path,
    ) -> None:
        """
        Log a single experiment run to MLflow.

        Args:
            run_name: Name of the run
            algorithm_params: Algorithm configuration parameters
            data_params: Dataset parameters
            metrics_results: Results from the experiment
            artifacts_dir: Directory containing artifacts to log (plots, etc.)
        """
        try:
            with mlflow.start_run(run_name=run_name):
                # Log parameters
                self._log_parameters(
                    algorithm_params=algorithm_params,
                    data_params=data_params,
                )

                # Log metrics
                self._log_metrics(metrics_results)

                # Log plot images
                self._log_plot_images(artifacts_dir)

                logger.info(f"Successfully logged run '{run_name}' to MLflow")

        except Exception as e:
            logger.error(f"Failed to log run to MLflow: {str(e)}")
            raise

    def _log_parameters(
        self,
        algorithm_params: Dict[str, Any],
        data_params: Dict[str, Any],
    ) -> None:
        """Log parameters with appropriate prefixes."""
        # Log algorithm parameters
        for key, value in algorithm_params.items():
            mlflow.log_param(f"algorithm_{key}", value)

        # Log data parameters
        for key, value in data_params.items():
            mlflow.log_param(f"data_{key}", value)

    def _log_metrics(self, metrics_results: Dict[str, Any]) -> None:
        """Log metrics from different categories."""
        for category, metrics in metrics_results.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    # Skip confusion matrices
                    if metric_name != "confusion_matrix":
                        mlflow.log_metric(f"{category}_{metric_name}", value)

    def _log_plot_images(self, artifacts_dir: pathlib.Path) -> None:
        """Log PNG plot images from the specified directory."""
        if artifacts_dir.exists():
            for png_file in artifacts_dir.glob("*.png"):
                try:
                    mlflow.log_artifact(str(png_file))
                    logger.info(f"Logged plot: {png_file.name}")
                except Exception as e:
                    logger.error(f"Failed to log plot {png_file.name}: {str(e)}")
