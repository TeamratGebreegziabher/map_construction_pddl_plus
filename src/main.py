from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from map_extractor import run_map_extraction
from graph_processor import run_graph_processing
from pddl_generator import run_pddl_generation
from planner_runner import run_planner
from plan_parser import run_plan_parsing
from evaluator import run_plan_validation
from dijkstra_baseline import run_dijkstra_baseline
from visualizer import run_visualizations
from interactive_visualizer import run_interactive_visualizations


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def print_project_info(config: dict[str, Any]) -> None:
    print("Project :", config["project"]["name"])
    print("Version :", config["project"]["version"])
    print("Map     :", config["map"]["place_name"])
    print("Planner :", config["planner"]["name"])


def separator(label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}\n")


def main() -> None:
    config = load_config()
    print_project_info(config)

    # ------------------------------------------------------------------
    # Stage 1 — OSM extraction
    # ------------------------------------------------------------------
    separator("Stage 1 — OpenStreetMap extraction")
    run_map_extraction(config)

    # ------------------------------------------------------------------
    # Stage 2 — Graph processing
    # ------------------------------------------------------------------
    separator("Stage 2 — Graph simplification and instance generation")
    instances = run_graph_processing(config)
    print(f"Generated {len(instances)} PDDL+ instances.")
    for inst in instances:
        print(
            f"  {inst['instance_name']:8s}: "
            f"{inst['num_locations']} locations, "
            f"{inst['num_edges']} edges, "
            f"start={inst['start']}, goal={inst['goal']}"
        )

    # ------------------------------------------------------------------
    # Stage 3 — PDDL+ generation
    # ------------------------------------------------------------------
    separator("Stage 3 — PDDL+ generation")
    pddl_outputs = run_pddl_generation(config)
    print(f"Domain : {pddl_outputs['domain_path']}")
    for p in pddl_outputs["problem_paths"]:
        print(f"Problem: {p}")
    print(f"Summary: {pddl_outputs['summary_path']}")

    # ------------------------------------------------------------------
    # Stage 4 — ENHSP planner
    # ------------------------------------------------------------------
    separator("Stage 4 — ENHSP planner execution")
    planner_outputs = run_planner(config)
    print(f"{'Instance':10s}  {'Status':10s}  {'Found':5s}  "
          f"{'Length':6s}  {'Runtime':>10s}")
    print("-" * 50)
    for r in planner_outputs["results"]:
        print(
            f"  {r['instance_name']:8s}  "
            f"{r['status']:10s}  "
            f"{'Yes' if r['plan_found'] else 'No':5s}  "
            f"{r['plan_length']:6d}  "
            f"{r['runtime_seconds']:9.3f}s"
        )
    print(f"\nPlanner CSV : {planner_outputs['csv_path']}")
    print(f"Planner JSON: {planner_outputs['json_path']}")

    # ------------------------------------------------------------------
    # Stage 5 — Plan parsing
    # ------------------------------------------------------------------
    separator("Stage 5 — Plan parsing")
    parsing_outputs = run_plan_parsing(config)
    for item in parsing_outputs["parsed_outputs"]:
        print(
            f"  {item['instance_name']:8s}: "
            f"{item['num_move_actions']} move actions, "
            f"route length {item['route_length']}"
        )

    # ------------------------------------------------------------------
    # Stage 6 — Plan validation
    # ------------------------------------------------------------------
    separator("Stage 6 — Plan validation")
    validation_outputs = run_plan_validation(config)
    for r in validation_outputs["validation_results"]:
        status = "VALID  " if r["valid"] else "INVALID"
        print(
            f"  {r['instance_name']:8s}: {status}  "
            f"distance={r['total_distance_m']} m  "
            f"battery_used={r['battery_used']}  "
            f"final_battery={r['final_battery']}"
        )
    print(f"\nValidation CSV : {validation_outputs['csv_path']}")
    print(f"Validation JSON: {validation_outputs['json_path']}")

    # ------------------------------------------------------------------
    # Stage 7 — Dijkstra baseline comparison
    # ------------------------------------------------------------------
    separator("Stage 7 — Dijkstra baseline comparison")
    comparison_outputs = run_dijkstra_baseline(config)
    for c in comparison_outputs["comparisons"]:
        gap = c["distance_gap_percent"]
        same = c["same_route_as_dijkstra"]
        print(
            f"  {c['instance_name']:8s}: "
            f"planner={c['planner_distance_m']} m  "
            f"dijkstra={c['dijkstra_distance_m']} m  "
            f"gap={gap}%  "
            f"{'same route' if same else 'different route'}"
        )
    print(f"\nComparison CSV : {comparison_outputs['csv_path']}")
    print(f"Comparison JSON: {comparison_outputs['json_path']}")

    # ------------------------------------------------------------------
    # Stage 8 — Static matplotlib visualisations
    # ------------------------------------------------------------------
    separator("Stage 8 — Static route visualisations")
    viz_outputs = run_visualizations(config)
    for fig in viz_outputs["route_figures"]:
        print(f"  Route figure: {fig}")
    for chart in viz_outputs["charts"]:
        print(f"  Chart       : {chart}")

    # ------------------------------------------------------------------
    # Stage 9 — Interactive Folium maps
    # ------------------------------------------------------------------
    separator("Stage 9 — Interactive Folium maps")
    map_outputs = run_interactive_visualizations(config)
    for m in map_outputs["interactive_maps"]:
        print(f"  Interactive map: {m}")

    separator("All stages completed successfully")


if __name__ == "__main__":
    main()