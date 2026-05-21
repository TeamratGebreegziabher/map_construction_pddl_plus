from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import networkx as nx


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(path: str | Path) -> Any:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_instance(instance_name: str, config: dict[str, Any]) -> dict[str, Any]:
    processed_dir = Path(config["outputs"]["processed_data_dir"])
    return load_json(processed_dir / f"{instance_name}_instance.json")


def load_parsed_plan(instance_name: str, config: dict[str, Any]) -> dict[str, Any]:
    results_dir = Path(config["outputs"]["results_dir"])
    return load_json(results_dir / f"{instance_name}_parsed_plan.json")


def load_validation_results(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results_dir = Path(config["outputs"]["results_dir"])
    results = load_json(results_dir / "validation_results.json")
    return {item["instance_name"]: item for item in results}


def load_planner_results(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results_dir = Path(config["outputs"]["results_dir"])
    results = load_json(results_dir / "planner_results.json")
    return {item["instance_name"]: item for item in results}


def build_weighted_graph(instance: dict[str, Any]) -> nx.DiGraph:
    """
    Build a directed graph from the processed instance.

    If multiple edges exist between two locations, keep the shortest one.
    """
    graph = nx.DiGraph()

    for location in instance["locations"]:
        graph.add_node(
            location["id"],
            lat=float(location["lat"]),
            lon=float(location["lon"]),
            osm_id=location["osm_id"],
        )

    for edge in instance["edges"]:
        source = edge["from"]
        target = edge["to"]
        distance = float(edge["distance_m"])
        travel_time = float(edge["travel_time_s"])

        if graph.has_edge(source, target):
            if distance < graph[source][target]["distance_m"]:
                graph[source][target].update(
                    distance_m=distance,
                    travel_time_s=travel_time,
                    name=edge.get("name", ""),
                    highway=edge.get("highway", ""),
                )
        else:
            graph.add_edge(
                source,
                target,
                distance_m=distance,
                travel_time_s=travel_time,
                name=edge.get("name", ""),
                highway=edge.get("highway", ""),
            )

    return graph


def compute_route_metrics(
    graph: nx.DiGraph,
    route: list[str],
    battery_consumption_per_meter: float,
) -> dict[str, float]:
    total_distance = 0.0
    total_time = 0.0

    for source, target in zip(route[:-1], route[1:]):
        if not graph.has_edge(source, target):
            raise ValueError(f"Route uses missing edge: {source} -> {target}")

        edge = graph[source][target]
        total_distance += float(edge["distance_m"])
        total_time += float(edge["travel_time_s"])

    battery_used = total_distance * battery_consumption_per_meter

    return {
        "distance_m": round(total_distance, 3),
        "travel_time_s": round(total_time, 3),
        "battery_used": round(battery_used, 3),
    }


def compute_dijkstra_baseline(instance: dict[str, Any]) -> dict[str, Any]:
    """
    Compute exact Dijkstra shortest-distance route for one instance.
    """
    graph = build_weighted_graph(instance)

    start = instance["start"]
    goal = instance["goal"]

    vehicle = instance["vehicle"]
    consumption = float(vehicle["battery_consumption_per_meter"])

    dijkstra_route = nx.shortest_path(
        graph,
        source=start,
        target=goal,
        weight="distance_m",
        method="dijkstra",
    )

    metrics = compute_route_metrics(
        graph=graph,
        route=dijkstra_route,
        battery_consumption_per_meter=consumption,
    )

    return {
        "instance_name": instance["instance_name"],
        "start": start,
        "goal": goal,
        "dijkstra_route": dijkstra_route,
        "dijkstra_num_moves": max(len(dijkstra_route) - 1, 0),
        "dijkstra_distance_m": metrics["distance_m"],
        "dijkstra_travel_time_s": metrics["travel_time_s"],
        "dijkstra_battery_used": metrics["battery_used"],
    }


def compare_planner_with_dijkstra(
    instance: dict[str, Any],
    parsed_plan: dict[str, Any],
    validation: dict[str, Any],
    planner_result: dict[str, Any],
    dijkstra: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare ENHSP planner route with Dijkstra shortest route.
    """
    planner_route = parsed_plan["route"]
    dijkstra_route = dijkstra["dijkstra_route"]

    planner_distance = float(validation["total_distance_m"])
    dijkstra_distance = float(dijkstra["dijkstra_distance_m"])

    distance_gap = planner_distance - dijkstra_distance

    if dijkstra_distance > 0:
        distance_gap_percent = (distance_gap / dijkstra_distance) * 100.0
    else:
        distance_gap_percent = 0.0

    planner_battery = float(validation["battery_used"])
    dijkstra_battery = float(dijkstra["dijkstra_battery_used"])

    battery_gap = planner_battery - dijkstra_battery

    if dijkstra_battery > 0:
        battery_gap_percent = (battery_gap / dijkstra_battery) * 100.0
    else:
        battery_gap_percent = 0.0

    same_route = planner_route == dijkstra_route

    return {
        "instance_name": instance["instance_name"],
        "num_locations": instance["num_locations"],
        "num_edges": instance["num_edges"],
        "start": instance["start"],
        "goal": instance["goal"],

        "planner_status": planner_result.get("status"),
        "planner_runtime_seconds": planner_result.get("runtime_seconds"),
        "planner_plan_found": planner_result.get("plan_found"),
        "planner_valid": validation["valid"],
        "planner_num_moves": validation["num_move_actions"],
        "planner_distance_m": round(planner_distance, 3),
        "planner_travel_time_s": validation["total_travel_time_s"],
        "planner_battery_used": round(planner_battery, 3),
        "planner_final_battery": validation["final_battery"],
        "planner_route": planner_route,

        "dijkstra_num_moves": dijkstra["dijkstra_num_moves"],
        "dijkstra_distance_m": round(dijkstra_distance, 3),
        "dijkstra_travel_time_s": dijkstra["dijkstra_travel_time_s"],
        "dijkstra_battery_used": round(dijkstra_battery, 3),
        "dijkstra_route": dijkstra_route,

        "distance_gap_m": round(distance_gap, 3),
        "distance_gap_percent": round(distance_gap_percent, 3),
        "battery_gap": round(battery_gap, 3),
        "battery_gap_percent": round(battery_gap_percent, 3),
        "same_route_as_dijkstra": same_route,
    }


def save_comparison_json(
    comparisons: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    results_dir = ensure_directory(config["outputs"]["results_dir"])
    output_path = results_dir / "comparison_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(comparisons, file, indent=2)

    return output_path


def save_comparison_csv(
    comparisons: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    results_dir = ensure_directory(config["outputs"]["results_dir"])
    output_path = results_dir / "comparison_results.csv"

    fieldnames = [
        "instance_name",
        "num_locations",
        "num_edges",
        "planner_status",
        "planner_runtime_seconds",
        "planner_valid",
        "planner_num_moves",
        "dijkstra_num_moves",
        "planner_distance_m",
        "dijkstra_distance_m",
        "distance_gap_m",
        "distance_gap_percent",
        "planner_battery_used",
        "dijkstra_battery_used",
        "battery_gap",
        "battery_gap_percent",
        "planner_final_battery",
        "same_route_as_dijkstra",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for comparison in comparisons:
            writer.writerow({field: comparison.get(field) for field in fieldnames})

    return output_path


def run_dijkstra_baseline(config: dict[str, Any]) -> dict[str, Any]:
    """
    Compute Dijkstra baselines and compare with planner plans.
    """
    validation_results = load_validation_results(config)
    planner_results = load_planner_results(config)

    comparisons = []

    for instance_name in ["small", "medium", "large"]:
        print(f"Computing Dijkstra baseline for: {instance_name}")

        instance = load_instance(instance_name, config)
        parsed_plan = load_parsed_plan(instance_name, config)

        dijkstra = compute_dijkstra_baseline(instance)

        comparison = compare_planner_with_dijkstra(
            instance=instance,
            parsed_plan=parsed_plan,
            validation=validation_results[instance_name],
            planner_result=planner_results[instance_name],
            dijkstra=dijkstra,
        )

        comparisons.append(comparison)

        print(
            f"{instance_name}: "
            f"planner={comparison['planner_distance_m']} m, "
            f"dijkstra={comparison['dijkstra_distance_m']} m, "
            f"gap={comparison['distance_gap_percent']}%"
        )

    json_path = save_comparison_json(comparisons, config)
    csv_path = save_comparison_csv(comparisons, config)

    print(f"\nSaved comparison JSON to: {json_path}")
    print(f"Saved comparison CSV to: {csv_path}")

    return {
        "comparisons": comparisons,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }