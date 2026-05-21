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

def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load project configuration from YAML file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def print_project_info(config: dict[str, Any]) -> None:
    """Print basic project information."""
    print("Project:", config["project"]["name"])
    print("Version:", config["project"]["version"])
    print("Target map:", config["map"]["place_name"])
    print("Network type:", config["map"]["network_type"])
    print("Planner:", config["planner"]["name"])

def main() -> None:
    config = load_config()

    print_project_info(config)
    print("\nStarting OpenStreetMap extraction...\n")
    print("=================================\n")
    run_map_extraction(config)

    print("Graph simplification and instance generation...\n")
    print("=================================\n")
    instances = run_graph_processing(config)
    print(f"Generated {len(instances)} PDDL+ instances.")
    for instance in instances:
        print(
            f" - {instance['instance_name']}:"
            f"{instance['num_locations']} locations,"
            f"{instance['num_edges']} edges,"
            f"start={instance['start']},"
            f"goal={instance['goal']}"
            )
        print("Pipeline completed successfully.")
    print("\nPDDL+ generation...")
    print("=================================\n")
    pddl_outputs = run_pddl_generation(config)
    print(f"Generated PDDL+ files:")
    print(f" - Domain file: {pddl_outputs['domain_path']}")
    for problem_path in pddl_outputs["problem_paths"]:
        print(f" - Problem file: {problem_path}")
    print(f"-Summary: {pddl_outputs['summary_path']}")
    print("All tasks completed successfully.")

    print("Planner execution...\n")
    print("=================================\n")
    planner_outtputs = run_planner(config)
    print("\nPlanner Results")
    for result in planner_outtputs['results']:
        print(f" - Result = {result['instance_name']}"
              f" Status = {result['status']}"
              f" Plan found = {result['plan_found']}"
              f" plan_length = {result['plan_length']}"
              f" runtime = {result['runtime_seconds']} seconds"
             )
        print(f"Planner CSV: {planner_outtputs['csv_path']}")
        print(f"Planner JSON: {planner_outtputs['json_path']}")
        print("Planner completed successfully.")
    
        print("\nPlan parsing...")
    print("=================================")
    parsing_outputs = run_plan_parsing(config)

    print("\nParsed plan files:")
    for item in parsing_outputs["parsed_outputs"]:
        print(
            f" - {item['instance_name']}: "
            f"{item['num_move_actions']} moves, "
            f"route length={item['route_length']}"
        )

    print("\nPlan validation...")
    print("=================================")
    validation_outputs = run_plan_validation(config)

    print("\nValidation results:")
    for result in validation_outputs["validation_results"]:
        status = "VALID" if result["valid"] else "INVALID"
        print(
            f" - {result['instance_name']}: {status}, "
            f"distance={result['total_distance_m']} m, "
            f"battery_used={result['battery_used']}, "
            f"final_battery={result['final_battery']}"
        )

    print(f"Validation CSV: {validation_outputs['csv_path']}")
    print(f"Validation JSON: {validation_outputs['json_path']}")


if __name__ == "__main__":
    main()