from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from dijkstra_baseline import run_dijkstra_baseline
from visualizer import run_visualizations


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = PROJECT_ROOT / config_path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()

    print("Baseline Analysis: Dijkstra Comparison and Visualization")
    print("========================================================")

    print("\nComputing exact Dijkstra baselines...")
    comparison_outputs = run_dijkstra_baseline(config)

    print("\nGenerating visualizations...")
    visualization_outputs = run_visualizations(config)

    print("\nBaseline analysis completed successfully..")
    print(f"Comparison CSV: {comparison_outputs['csv_path']}")
    print(f"Comparison JSON: {comparison_outputs['json_path']}")

    print("\nGenerated route figures:")
    for path in visualization_outputs["route_figures"]:
        print(f" - {path}")

    print("\nGenerated charts:")
    for path in visualization_outputs["charts"]:
        print(f" - {path}")


if __name__ == "__main__":
    main()