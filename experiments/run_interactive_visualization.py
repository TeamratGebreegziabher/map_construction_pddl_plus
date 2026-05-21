from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from interactive_visualizer import run_interactive_visualizations


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = PROJECT_ROOT / config_path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()

    print("Interactive Route Visualization")
    print("===============================")

    outputs = run_interactive_visualizations(config)

    print("\nGenerated interactive maps:")
    for path in outputs["interactive_maps"]:
        print(f" - {path}")

    print("\nInteractive visualization completed successfully.")


if __name__ == "__main__":
    main()