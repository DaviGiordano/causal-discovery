import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import sys


def setup_logging(algorithm_tag: str, data_tag: str):
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = Path(f"logs/{algorithm_tag}_{data_tag}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_filename = f"{log_dir}/info.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename)],  # Remove StreamHandler
    )
    logger = logging.getLogger(__name__)

    logger.info(
        f"Starting experiment with algorithm={algorithm_tag}, dataset={data_tag}"
    )
    logger.info(f"Results will be saved in {log_dir}")

    return logger, log_dir


@contextmanager
def redirect_stdout_to_logger():
    class LoggerWriter:
        def __init__(self):
            # Use the same logger as setup in setup_logging()
            self.logger = logging.getLogger(__name__)

        def write(self, message):
            if message and not message.isspace():
                self.logger.info(message.strip())

        def flush(self):
            pass

    stdout = sys.stdout
    sys.stdout = LoggerWriter()
    try:
        yield
    finally:
        sys.stdout = stdout
