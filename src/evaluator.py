from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_instance(instance_name: str, config: dict[str, Any]) -> dict[str, Any]:
    processed_dir = Path(config["outputs"]["processed_data_dir"])
    instance_path = processed_dir / f"{instance_name}_instance.json"

    return load_json(instance_path)


def load_parsed_plan(instance_name: str, config: dict[str, Any]) -> dict[str, Any]:
    results_dir = Path(config["outputs"]["results_dir"])
    parsed_path = results_dir / f"{instance_name}_parsed_plan.json"

    return load_json(parsed_path)


def build_edge_lookup(instance: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    edge_lookup = {}

    for edge in instance["edges"]:
        key = (edge["from"], edge["to"])
        edge_lookup[key] = edge

    return edge_lookup


def evaluate_plan(
    instance: dict[str, Any],
    parsed_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate one parsed plan against the processed graph instance.
    """
    instance_name = instance["instance_name"]
    start = instance["start"]
    goal = instance["goal"]

    vehicle = instance["vehicle"]
    initial_battery = float(vehicle["initial_battery"])
    consumption_per_meter = float(vehicle["battery_consumption_per_meter"])

    move_actions = parsed_plan["move_actions"]
    route = parsed_plan["route"]

    edge_lookup = build_edge_lookup(instance)
    blocked_edge_set = build_blocked_edge_set(instance)

    errors = []
    warnings = []

    total_distance = 0.0
    total_travel_time = 0.0

    if not move_actions:
        errors.append("No start-move actions found in the plan.")

    if route:
        if route[0] != start:
            errors.append(
                f"Route starts at {route[0]}, but expected start is {start}."
            )

        if route[-1] != goal:
            errors.append(
                f"Route ends at {route[-1]}, but expected goal is {goal}."
            )
    else:
        errors.append("Route is empty.")
    repeated_locations = find_repeated_locations(route)

    if repeated_locations:
        errors.append(
            "Route contains repeated locations/cycles, which is not acceptable "
            f"for this route-planning task. Repeated locations: {repeated_locations[:10]}"
        )
    current_location = start

    for index, move in enumerate(move_actions):
        from_location = move.get("from")
        to_location = move.get("to")

        if from_location != current_location:
            errors.append(
                f"Move {index}: expected to move from {current_location}, "
                f"but action moves from {from_location}."
            )

        edge_key = (from_location, to_location)
        if edge_key in blocked_edge_set:
            errors.append(
                f"Move {index}: edge {from_location} -> {to_location} is blocked "
                "but was used by the plan."
            )
        if edge_key not in edge_lookup:
            errors.append(
                f"Move {index}: edge {from_location} -> {to_location} "
                "does not exist in the processed graph."
            )
        else:
            edge = edge_lookup[edge_key]
            total_distance += float(edge["distance_m"])
            total_travel_time += float(edge["travel_time_s"])

        current_location = to_location

    battery_used = total_distance * consumption_per_meter
    final_battery = initial_battery - battery_used

    if final_battery < -1e-6:
        errors.append(
            f"Battery becomes negative: final_battery={final_battery:.3f}."
        )

    estimated_shortest_distance = float(
        instance.get("estimated_shortest_distance_m", 0.0)
    )

    distance_gap = total_distance - estimated_shortest_distance

    if estimated_shortest_distance > 0:
        distance_gap_percent = (distance_gap / estimated_shortest_distance) * 100.0
    else:
        distance_gap_percent = 0.0

    if distance_gap_percent > 5:
        warnings.append(
            f"Plan distance is {distance_gap_percent:.2f}% longer "
            "than the Dijkstra shortest-distance estimate."
        )

    valid = len(errors) == 0

    return {
        "instance_name": instance_name,
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "start": start,
        "goal": goal,
        "route_start": route[0] if route else None,
        "route_end": route[-1] if route else None,
        "num_locations_in_route": len(route),
        "num_move_actions": len(move_actions),
        "total_distance_m": round(total_distance, 3),
        "estimated_shortest_distance_m": round(estimated_shortest_distance, 3),
        "distance_gap_m": round(distance_gap, 3),
        "distance_gap_percent": round(distance_gap_percent, 3),
        "total_travel_time_s": round(total_travel_time, 3),
        "initial_battery": round(initial_battery, 3),
        "battery_used": round(battery_used, 3),
        "final_battery": round(final_battery, 3),
        "route": route,
    }

def build_blocked_edge_set(instance: dict[str, Any]) -> set[tuple[str, str]]:
    blocked_edges = set()

    for edge in instance.get("blocked_edges", []):
        blocked_edges.add((edge["from"], edge["to"]))

    return blocked_edges

def save_validation_json(
    validation_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    results_dir = ensure_directory(config["outputs"]["results_dir"])
    output_path = results_dir / "validation_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(validation_results, file, indent=2)

    return output_path


def save_validation_csv(
    validation_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    results_dir = ensure_directory(config["outputs"]["results_dir"])
    output_path = results_dir / "validation_results.csv"

    fieldnames = [
        "instance_name",
        "valid",
        "num_move_actions",
        "num_locations_in_route",
        "total_distance_m",
        "estimated_shortest_distance_m",
        "distance_gap_m",
        "distance_gap_percent",
        "total_travel_time_s",
        "initial_battery",
        "battery_used",
        "final_battery",
        "route_start",
        "route_end",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in validation_results:
            writer.writerow({field: result.get(field) for field in fieldnames})

    return output_path


def run_plan_validation(config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate small, medium, and large plans.
    """
    validation_results = []

    for instance_name in ["small", "medium", "large"]:
        print(f"Validating plan for instance: {instance_name}")

        instance = load_instance(instance_name, config)
        parsed_plan = load_parsed_plan(instance_name, config)

        result = evaluate_plan(instance, parsed_plan)
        validation_results.append(result)

        status = "VALID" if result["valid"] else "INVALID"

        print(
            f"{instance_name}: {status}, "
            f"moves={result['num_move_actions']}, "
            f"distance={result['total_distance_m']} m, "
            f"battery_used={result['battery_used']}, "
            f"final_battery={result['final_battery']}"
        )

        if result["errors"]:
            print("Errors:")
            for error in result["errors"]:
                print(f"  - {error}")

        if result["warnings"]:
            print("Warnings:")
            for warning in result["warnings"]:
                print(f"  - {warning}")

    json_path = save_validation_json(validation_results, config)
    csv_path = save_validation_csv(validation_results, config)

    print(f"\nSaved validation JSON to: {json_path}")
    print(f"Saved validation CSV to: {csv_path}")

    return {
        "validation_results": validation_results,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }
def find_repeated_locations(route: list[str]) -> list[str]:
    seen = set()
    repeated = []

    for location in route:
        if location in seen:
            repeated.append(location)
        else:
            seen.add(location)

    return repeated