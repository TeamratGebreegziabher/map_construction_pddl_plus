from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from sumo_simulator import run_sumo_simulations


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = PROJECT_ROOT / config_path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()

    print("SUMO Simulation of PDDL+ Planner Routes")
    print("=======================================")

    outputs = run_sumo_simulations(config)

    print("\nGenerated SUMO simulations:")
    for result in outputs["results"]:
        status = "OK" if result["validation"]["success"] else "FAILED"

        print(
            f"\n{result['instance_name']} — {status}"
            f"\n  SUMO config: {result['sumo_config_file']}"
            f"\n  route file:  {result['route_file']}"
            f"\n  network:     {result['network_file']}"
            f"\n  open script: {result['open_gui_script']}"
        )

    print(f"\nSummary: {outputs['summary_path']}")

    print("\nTo open the small simulation:")
    print(r"  powershell -ExecutionPolicy Bypass -File outputs\sumo\small\open_small_sumo_gui.ps1")


if __name__ == "__main__":
    main()