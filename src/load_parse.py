import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)
    return cfg


def load_variable_types(path: Path) -> Dict:
    """Loads and parses Tetrad-formatted metadata into a dtype dict.
    {variable_name: type, ...}
    """
    pass


def apply_metadata_conversions(df: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    """Apply type conversions based on metadata file."""
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.loads(f.read())

        # Extract continuous column names
        continuous_cols = [
            domain["name"]
            for domain in metadata.get("domains", [])
            if not domain.get("discrete", True)
        ]

        # Convert continuous columns to float64
        converted_cols = []
        for col in continuous_cols:
            if col in df.columns:
                df[col] = df[col].astype("float64")
                converted_cols.append(col)

        if converted_cols:
            logger.info(
                f"Converted {len(converted_cols)} continuous columns to float64: {', '.join(converted_cols)}"
            )
    except Exception as err:
        logger.warning(f"Failed to apply metadata conversions: {err}")

    return df


def load_data(data_path: Path, metadata_path: Optional[Path] = None) -> pd.DataFrame:
    """Loads dataset and applies optional metadata conversions."""
    try:
        df = pd.read_csv(data_path)
    except Exception as err:
        raise RuntimeError(f"Unable to read dataset '{data_path}': {err}") from err

    if df.empty:
        raise ValueError("Dataset contains no rows.")

    # Process metadata if provided
    if metadata_path and metadata_path.exists():
        df = apply_metadata_conversions(df, metadata_path)

    return df


def load_txt(path: Path) -> str:
    """Loads a text file and returns as string"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        return txt
    except Exception as err:
        raise Exception(f"Error reading {path}: {err}") from err


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--metadata",
        required=False,
        type=Path,
        help="Optional metadata JSON file with column type information",
    )
    return parser.parse_args()
