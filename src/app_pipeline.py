from __future__ import annotations

import math
import random
import subprocess
import xml.etree.ElementTree as ET
import json
import shutil
import time
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox
from shapely.geometry import shape

from evaluator import evaluate_plan
from graph_processor import (
    create_instance_subgraph,
    create_pddl_location_mapping,
    extract_edges,
    extract_locations,
    keep_largest_strongly_connected_component,
    normalize_edge_lengths,
)
from interactive_visualizer import create_interactive_route_map
from pddl_generator import save_domain, save_problem, save_multi_vehicle_problem
from plan_parser import parse_plan_file, save_parsed_plan
from planner_runner import run_single_problem


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return output_path


def update_edge_travel_times(
    instance: dict[str, Any],
    speed_m_per_s: float,
) -> dict[str, Any]:
    updated = dict(instance)
    updated["edges"] = [
        {**edge, "travel_time_s": round(float(edge["distance_m"]) / speed_m_per_s, 3)}
        for edge in instance["edges"]
    ]
    return updated


# ---------------------------------------------------------------------------
# Graph / routing utilities
# ---------------------------------------------------------------------------

def build_weighted_graph(
    instance: dict[str, Any],
    metric: str = "distance",
    ignore_blocked: bool = True,
) -> nx.DiGraph:
    graph = nx.DiGraph()

    blocked_set = {
        (e["from"], e["to"])
        for e in instance.get("blocked_edges", [])
    }

    for loc in instance["locations"]:
        graph.add_node(loc["id"])

    for edge in instance["edges"]:
        src = edge["from"]
        tgt = edge["to"]

        if ignore_blocked and (src, tgt) in blocked_set:
            continue

        distance = float(edge["distance_m"])
        travel_time = float(edge["travel_time_s"])

        if graph.has_edge(src, tgt):
            if distance < graph[src][tgt]["distance_m"]:
                graph[src][tgt].update(
                    distance_m=distance,
                    travel_time_s=travel_time,
                )
        else:
            graph.add_edge(src, tgt, distance_m=distance, travel_time_s=travel_time)

    return graph


def compute_route_metrics(
    graph: nx.DiGraph,
    route: list[str],
    battery_consumption_per_meter: float,
) -> dict[str, float]:
    total_distance = 0.0
    total_time = 0.0
    for src, tgt in zip(route[:-1], route[1:]):
        edge = graph[src][tgt]
        total_distance += float(edge["distance_m"])
        total_time += float(edge["travel_time_s"])
    return {
        "distance_m": round(total_distance, 3),
        "travel_time_s": round(total_time, 3),
        "battery_used": round(total_distance * battery_consumption_per_meter, 3),
    }


def compute_dijkstra_for_instance(
    instance: dict[str, Any],
    metric: str = "distance",
    start: str | None = None,
    goal: str | None = None,
) -> dict[str, Any]:
    """
    Compute Dijkstra shortest path.
    start/goal default to instance["start"] / instance["goal"].
    Accepts explicit start/goal for multi-vehicle per-vehicle comparison.
    """
    graph = build_weighted_graph(instance, metric=metric, ignore_blocked=True)
    start = start or instance["start"]
    goal = goal or instance["goal"]

    if not nx.has_path(graph, start, goal):
        return {
            "route_found": False,
            "dijkstra_route": [],
            "dijkstra_num_moves": 0,
            "dijkstra_distance_m": None,
            "dijkstra_travel_time_s": None,
            "dijkstra_battery_used": None,
        }

    weight = "travel_time_s" if metric == "time" else "distance_m"
    route = nx.shortest_path(
        graph, source=start, target=goal,
        weight=weight, method="dijkstra",
    )
    metrics = compute_route_metrics(
        graph=graph,
        route=route,
        battery_consumption_per_meter=float(
            instance["vehicle"]["battery_consumption_per_meter"]
        ),
    )
    return {
        "route_found": True,
        "dijkstra_route": route,
        "dijkstra_num_moves": max(len(route) - 1, 0),
        "dijkstra_distance_m": metrics["distance_m"],
        "dijkstra_travel_time_s": metrics["travel_time_s"],
        "dijkstra_battery_used": metrics["battery_used"],
    }


def create_comparison(
    instance: dict[str, Any],
    validation: dict[str, Any],
    planner_result: dict[str, Any],
    dijkstra: dict[str, Any],
) -> dict[str, Any]:
    planner_distance = float(validation["total_distance_m"])
    planner_battery = float(validation["battery_used"])

    if dijkstra["route_found"] and dijkstra["dijkstra_distance_m"] is not None:
        dijkstra_distance = float(dijkstra["dijkstra_distance_m"])
        distance_gap = planner_distance - dijkstra_distance
        distance_gap_percent = (
            distance_gap / dijkstra_distance * 100.0
            if dijkstra_distance > 0 else 0.0
        )
    else:
        dijkstra_distance = None
        distance_gap = None
        distance_gap_percent = None

    if dijkstra["dijkstra_battery_used"] is not None:
        dijkstra_battery = float(dijkstra["dijkstra_battery_used"])
        battery_gap = planner_battery - dijkstra_battery
        battery_gap_percent = (
            battery_gap / dijkstra_battery * 100.0
            if dijkstra_battery > 0 else 0.0
        )
    else:
        dijkstra_battery = None
        battery_gap = None
        battery_gap_percent = None

    return {
        "instance_name": instance["instance_name"],
        "num_locations": instance["num_locations"],
        "num_edges": instance["num_edges"],
        "start": instance["start"],
        "goal": instance["goal"],

        "planner_status": planner_result["status"],
        "planner_runtime_seconds": planner_result["runtime_seconds"],
        "planner_plan_found": planner_result["plan_found"],
        "planner_valid": validation["valid"],
        "planner_num_moves": validation["num_move_actions"],
        "planner_distance_m": validation["total_distance_m"],
        "planner_travel_time_s": validation["total_travel_time_s"],
        "planner_battery_used": validation["battery_used"],
        "planner_final_battery": validation["final_battery"],
        "planner_route": validation["route"],

        "dijkstra_route_found": dijkstra["route_found"],
        "dijkstra_route": dijkstra["dijkstra_route"],
        "dijkstra_num_moves": dijkstra["dijkstra_num_moves"],
        "dijkstra_distance_m": dijkstra_distance,
        "dijkstra_travel_time_s": dijkstra["dijkstra_travel_time_s"],
        "dijkstra_battery_used": dijkstra_battery,

        "distance_gap_m": round(distance_gap, 3) if distance_gap is not None else None,
        "distance_gap_percent": round(distance_gap_percent, 3) if distance_gap_percent is not None else None,
        "battery_gap": round(battery_gap, 3) if battery_gap is not None else None,
        "battery_gap_percent": round(battery_gap_percent, 3) if battery_gap_percent is not None else None,
        "same_route_as_dijkstra": (
            validation["route"] == dijkstra["dijkstra_route"]
            if dijkstra["route_found"] else False
        ),
    }


# ---------------------------------------------------------------------------
# Map extraction
# ---------------------------------------------------------------------------

def _build_base_instance(
    subgraph: Any,
    mapping: dict,
    locations: list,
    edges: list,
    place_name: str,
    network_type: str,
    vehicle_speed_m_per_s: float,
    config: dict[str, Any],
    polygon_geojson: dict | None = None,
) -> dict[str, Any]:
    """Shared builder for prepare_custom_map and prepare_custom_map_from_polygon."""
    n_signals = sum(1 for loc in locations if loc.get("has_traffic_signal", False))
    n_chargers = sum(1 for loc in locations if loc.get("has_charging_station", False))

    instance: dict[str, Any] = {
        "instance_name": "custom",
        "place_name": place_name,
        "network_type": network_type,
        "num_locations": len(locations),
        "num_edges": len(edges),
        "num_traffic_signals": n_signals,
        "num_charging_stations": n_chargers,
        "vehicle": {
            "name": config["vehicle"]["name"],
            "initial_battery": float(config["vehicle"]["initial_battery"]),
            "speed_m_per_s": float(vehicle_speed_m_per_s),
            "battery_consumption_per_meter": float(
                config["vehicle"]["battery_consumption_per_meter"]
            ),
        },
        "start": locations[0]["id"],
        "goal": locations[-1]["id"],
        "estimated_shortest_distance_m": 0.0,
        "locations": locations,
        "edges": edges,
        "blocked_edges": [],
        "congested_edges": [],
        "node_mapping": {
            str(original_id): pddl_id
            for original_id, pddl_id in mapping.items()
        },
    }

    if polygon_geojson is not None:
        instance["selection_mode"] = "drawn_polygon"
        instance["selected_polygon_geojson"] = polygon_geojson

    processed_dir = ensure_directory(config["outputs"]["processed_data_dir"])
    save_json(instance, processed_dir / "custom_base_instance.json")
    return instance


def prepare_custom_map(
    place_name: str,
    network_type: str,
    max_nodes: int,
    vehicle_speed_m_per_s: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    ox.settings.use_cache = True
    ox.settings.log_console = False

    graph = ox.graph_from_place(
        place_name, network_type=network_type,
        simplify=True, retain_all=False, truncate_by_edge=True,
    )
    graph = normalize_edge_lengths(graph)
    graph = keep_largest_strongly_connected_component(graph)
    subgraph = create_instance_subgraph(graph, max_nodes=max_nodes)
    mapping = create_pddl_location_mapping(subgraph)
    locations = extract_locations(subgraph, mapping)
    edges = extract_edges(subgraph, mapping, vehicle_speed_m_per_s=vehicle_speed_m_per_s)

    if len(locations) < 2:
        raise ValueError("The selected map is too small. Increase max_nodes or choose another area.")

    return _build_base_instance(
        subgraph, mapping, locations, edges,
        place_name, network_type, vehicle_speed_m_per_s, config,
    )


def prepare_custom_map_from_polygon(
    polygon_geojson: dict[str, Any],
    place_name: str,
    network_type: str,
    max_nodes: int,
    vehicle_speed_m_per_s: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    ox.settings.use_cache = True
    ox.settings.log_console = False

    polygon = shape(polygon_geojson)
    if polygon.is_empty:
        raise ValueError("Selected polygon is empty.")
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    graph = ox.graph_from_polygon(
        polygon, network_type=network_type,
        simplify=True, retain_all=False, truncate_by_edge=True,
    )
    graph = normalize_edge_lengths(graph)
    graph = keep_largest_strongly_connected_component(graph)
    subgraph = create_instance_subgraph(graph, max_nodes=max_nodes)
    mapping = create_pddl_location_mapping(subgraph)
    locations = extract_locations(subgraph, mapping)
    edges = extract_edges(subgraph, mapping, vehicle_speed_m_per_s=vehicle_speed_m_per_s)

    if len(locations) < 2:
        raise ValueError(
            "The selected area produced too few locations. "
            "Draw a larger area or increase max_nodes."
        )

    return _build_base_instance(
        subgraph, mapping, locations, edges,
        place_name, network_type, vehicle_speed_m_per_s, config,
        polygon_geojson=polygon_geojson,
    )


# ---------------------------------------------------------------------------
# Single-vehicle planning
# ---------------------------------------------------------------------------

def apply_user_choices_to_instance(
    base_instance: dict[str, Any],
    start: str,
    goal: str,
    initial_battery: float,
    speed_m_per_s: float,
    battery_consumption_per_meter: float,
    metric: str,
    blocked_edges: list[dict[str, Any]],
    config: dict[str, Any],
    congested_edges: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    instance = dict(base_instance)
    instance = update_edge_travel_times(instance, speed_m_per_s=speed_m_per_s)
    instance["instance_name"] = "custom"
    instance["start"] = start
    instance["goal"] = goal
    instance["vehicle"] = {
        "name": config["vehicle"]["name"],
        "initial_battery": float(initial_battery),
        "speed_m_per_s": float(speed_m_per_s),
        "battery_consumption_per_meter": float(battery_consumption_per_meter),
    }
    instance["blocked_edges"] = blocked_edges
    instance["congested_edges"] = congested_edges or []

    custom_config = dict(config)
    custom_config["planning"] = dict(config["planning"])
    custom_config["planning"]["metric"] = metric

    processed_dir = ensure_directory(config["outputs"]["processed_data_dir"])
    save_json(instance, processed_dir / "custom_instance.json")
    return instance, custom_config


def run_custom_planning_job(
    instance: dict[str, Any],
    config: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    """Single-vehicle planning job."""
    start_time = time.perf_counter()

    domain_path = save_domain(config)
    problem_path = save_problem(instance, config)

    planner_result = run_single_problem(
        config=config, domain_file=domain_path, problem_file=problem_path,
    )

    parsed_plan = parse_plan_file(planner_result["plan_file"])
    parsed_plan_path = save_parsed_plan(
        parsed_plan=parsed_plan,
        instance_name=instance["instance_name"],
        config=config,
    )

    validation = evaluate_plan(instance, parsed_plan)
    dijkstra = compute_dijkstra_for_instance(instance, metric=metric)

    comparison = create_comparison(
        instance=instance,
        validation=validation,
        planner_result=planner_result,
        dijkstra=dijkstra,
    )

    results_dir = ensure_directory(config["outputs"]["results_dir"])
    validation_path = save_json(validation, results_dir / "custom_validation_result.json")
    comparison_path = save_json(comparison, results_dir / "custom_comparison_result.json")

    interactive_map_path = None
    if validation["valid"]:
        interactive_map_path = create_interactive_route_map(
            instance=instance, comparison=comparison, config=config,
        )

    return {
        "success": validation["valid"],
        "is_multi_vehicle": False,
        "instance": instance,
        "domain_path": str(domain_path),
        "problem_path": str(problem_path),
        "planner_result": planner_result,
        "parsed_plan_path": str(parsed_plan_path),
        "validation": validation,
        "validation_path": str(validation_path),
        "dijkstra": dijkstra,
        "comparison": comparison,
        "comparison_path": str(comparison_path),
        "interactive_map_path": str(interactive_map_path) if interactive_map_path else None,
        "total_runtime_seconds": round(time.perf_counter() - start_time, 3),
    }


# ---------------------------------------------------------------------------
# Multi-vehicle planning
# ---------------------------------------------------------------------------

def run_multi_vehicle_planning_job(
    base_instance: dict[str, Any],
    vehicles: list[dict[str, Any]],
    metric: str,
    blocked_edges: list[dict[str, Any]],
    congested_edges: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Multi-vehicle planning job.

    vehicles: list of dicts with keys:
      id, start, goal, battery, speed_m_per_s,
      battery_consumption_per_meter, max_battery, charge_rate

    Shared starts and shared goals are supported.
    Each vehicle has an independent visited set in the domain.
    All constraints (blocked, congested) apply to all vehicles equally.
    """
    start_time = time.perf_counter()

    # Build instance for multi-vehicle
    instance = dict(base_instance)
    speed = float(vehicles[0].get("speed_m_per_s", 10.0))
    instance = update_edge_travel_times(instance, speed_m_per_s=speed)
    instance["instance_name"] = "custom_multivehicle"
    instance["blocked_edges"] = blocked_edges
    instance["congested_edges"] = congested_edges
    instance["vehicles"] = vehicles
    # Backward-compat fields from first vehicle
    instance["start"] = vehicles[0]["start"]
    instance["goal"] = vehicles[0]["goal"]
    instance["vehicle"] = {
        "name": vehicles[0]["id"],
        "initial_battery": float(vehicles[0].get("battery", 100.0)),
        "speed_m_per_s": speed,
        "battery_consumption_per_meter": float(
            vehicles[0].get("battery_consumption_per_meter", 0.01)
        ),
    }

    custom_config = dict(config)
    custom_config["planning"] = dict(config["planning"])
    custom_config["planning"]["metric"] = metric

    processed_dir = ensure_directory(config["outputs"]["processed_data_dir"])
    save_json(instance, processed_dir / "custom_multivehicle_instance.json")

    # Generate PDDL+ files
    domain_path = save_domain(custom_config)
    problem_path = save_multi_vehicle_problem(
        vehicles=vehicles,
        instance=instance,
        config=custom_config,
        metric=metric,
    )

    # Run ENHSP
    planner_result = run_single_problem(
        config=custom_config,
        domain_file=domain_path,
        problem_file=problem_path,
    )

    # Parse plan — per-vehicle routes
    parsed_plan = parse_plan_file(planner_result["plan_file"])
    parsed_plan_path = save_parsed_plan(
        parsed_plan=parsed_plan,
        instance_name=instance["instance_name"],
        config=custom_config,
    )

    per_vehicle_routes = parsed_plan.get("per_vehicle_routes", {})
    graph = build_weighted_graph(instance, metric=metric, ignore_blocked=True)

    # Validate and compare per vehicle
    per_vehicle_results: list[dict[str, Any]] = []

    for v in vehicles:
        vid = v["id"]
        v_start = v["start"]
        v_goal = v["goal"]
        v_route = per_vehicle_routes.get(vid, [])
        consumption = float(v.get("battery_consumption_per_meter", 0.01))

        if len(v_route) >= 2:
            try:
                planner_metrics = compute_route_metrics(
                    graph=graph, route=v_route,
                    battery_consumption_per_meter=consumption,
                )
                route_valid = True
            except Exception:
                planner_metrics = {"distance_m": 0.0, "travel_time_s": 0.0, "battery_used": 0.0}
                route_valid = False
        else:
            planner_metrics = {"distance_m": 0.0, "travel_time_s": 0.0, "battery_used": 0.0}
            route_valid = False

        dijkstra = compute_dijkstra_for_instance(
            instance=instance, metric=metric,
            start=v_start, goal=v_goal,
        )

        gap = None
        if (
            route_valid
            and dijkstra["route_found"]
            and dijkstra["dijkstra_distance_m"]
            and dijkstra["dijkstra_distance_m"] > 0
        ):
            gap = round(
                (planner_metrics["distance_m"] - dijkstra["dijkstra_distance_m"])
                / dijkstra["dijkstra_distance_m"] * 100.0,
                3,
            )

        per_vehicle_results.append({
            "vehicle_id": vid,
            "start": v_start,
            "goal": v_goal,
            "route": v_route,
            "route_valid": route_valid,
            "planner_distance_m": planner_metrics["distance_m"],
            "planner_travel_time_s": planner_metrics["travel_time_s"],
            "planner_battery_used": planner_metrics["battery_used"],
            "final_battery": round(
                float(v.get("battery", 100.0)) - planner_metrics["battery_used"], 3
            ),
            "dijkstra_route": dijkstra["dijkstra_route"],
            "dijkstra_distance_m": dijkstra["dijkstra_distance_m"],
            "distance_gap_percent": gap,
        })

    all_valid = all(r["route_valid"] for r in per_vehicle_results)
    total_distance = round(
        sum(r["planner_distance_m"] for r in per_vehicle_results), 3
    )

    # Interactive map — first vehicle route for backward compat
    interactive_map_path = None
    if all_valid and per_vehicle_results:
        first = per_vehicle_results[0]
        synthetic_comparison = {
            "planner_route": first["route"],
            "dijkstra_route": first["dijkstra_route"],
            "same_route_as_dijkstra": first["route"] == first["dijkstra_route"],
            "planner_distance_m": first["planner_distance_m"],
            "dijkstra_distance_m": first["dijkstra_distance_m"],
            "distance_gap_percent": first["distance_gap_percent"],
        }
        try:
            interactive_map_path = create_interactive_route_map(
                instance=instance,
                comparison=synthetic_comparison,
                config=custom_config,
            )
        except Exception:
            pass

    results_dir = ensure_directory(custom_config["outputs"]["results_dir"])
    mv_result = {
        "plan_found": planner_result["plan_found"],
        "all_routes_valid": all_valid,
        "total_distance_m": total_distance,
        "num_vehicles": len(vehicles),
        "vehicle_ids": [v["id"] for v in vehicles],
        "per_vehicle_results": per_vehicle_results,
    }
    save_json(mv_result, results_dir / "custom_multivehicle_result.json")

    return {
        "success": all_valid,
        "is_multi_vehicle": True,
        "num_vehicles": len(vehicles),
        "instance": instance,
        "vehicles": vehicles,
        "domain_path": str(domain_path),
        "problem_path": str(problem_path),
        "planner_result": planner_result,
        "parsed_plan": parsed_plan,
        "parsed_plan_path": str(parsed_plan_path),
        "per_vehicle_results": per_vehicle_results,
        "all_routes_valid": all_valid,
        "total_distance_m": total_distance,
        "interactive_map_path": str(interactive_map_path) if interactive_map_path else None,
        "total_runtime_seconds": round(time.perf_counter() - start_time, 3),
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def list_location_options(instance: dict[str, Any]) -> list[str]:
    return [
        f"{loc['id']} | lat={float(loc['lat']):.5f}, lon={float(loc['lon']):.5f}"
        for loc in instance["locations"]
    ]


def extract_location_id(option_label: str) -> str:
    return option_label.split("|")[0].strip()


def list_edge_options(instance: dict[str, Any]) -> list[str]:
    return [
        (
            f"{edge['from']} -> {edge['to']} | "
            f"{float(edge['distance_m']):.1f} m | "
            f"{edge.get('name', '')} | {edge.get('highway', '')}"
        )
        for edge in instance["edges"]
    ]


def extract_edge_pair(option_label: str) -> tuple[str, str]:
    pair = option_label.split("|")[0].strip()
    source, target = pair.split("->")
    return source.strip(), target.strip()


def build_blocked_edge_objects(
    selected_edge_labels: list[str],
    instance: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_pairs = {extract_edge_pair(label) for label in selected_edge_labels}
    return [
        {
            "from": edge["from"],
            "to": edge["to"],
            "distance_m": edge["distance_m"],
            "travel_time_s": edge["travel_time_s"],
            "name": edge.get("name", ""),
            "highway": edge.get("highway", ""),
            "reason": "Selected by user in interactive app",
        }
        for edge in instance["edges"]
        if (edge["from"], edge["to"]) in selected_pairs
    ]


def build_congested_edge_objects(
    selected_edge_labels: list[str],
    instance: dict[str, Any],
    congestion_factor: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Build congested edge objects.

    congestion_factor: speed multiplier (0.1 = very slow, 0.9 = mild).
    In the PDDL+ problem, the road-distance for these edges will be
    inflated by (real_distance / congestion_factor) so the planner
    treats them as longer and may reroute.
    """
    selected_pairs = {extract_edge_pair(label) for label in selected_edge_labels}
    return [
        {
            "from": edge["from"],
            "to": edge["to"],
            "distance_m": edge["distance_m"],
            "travel_time_s": edge["travel_time_s"],
            "name": edge.get("name", ""),
            "highway": edge.get("highway", ""),
            "congestion_factor": congestion_factor,
            "reason": "Selected by user in interactive app",
        }
        for edge in instance["edges"]
        if (edge["from"], edge["to"]) in selected_pairs
    ]


def find_nearest_location_id(
    instance: dict[str, Any],
    clicked_lat: float,
    clicked_lon: float,
) -> str:
    best_id = None
    best_dist = float("inf")
    for loc in instance["locations"]:
        d = (float(loc["lat"]) - clicked_lat) ** 2 + (float(loc["lon"]) - clicked_lon) ** 2
        if d < best_dist:
            best_dist = d
            best_id = loc["id"]
    if best_id is None:
        raise ValueError("No nearest location found.")
    return best_id


def clear_custom_outputs(config: dict[str, Any]) -> None:
    candidates = [
        Path(config["planning"]["problem_dir"]) / "problem_custom.pddl",
        Path(config["planning"]["problem_dir"]) / "problem_custom_multivehicle.pddl",
        Path(config["outputs"]["plans_dir"]) / "custom_plan.txt",
        Path(config["outputs"]["plans_dir"]) / "custom_multivehicle_plan.txt",
        Path(config["outputs"]["logs_dir"]) / "custom_planner.log",
        Path(config["outputs"]["logs_dir"]) / "custom_multivehicle_planner.log",
        Path(config["outputs"]["results_dir"]) / "custom_parsed_plan.json",
        Path(config["outputs"]["results_dir"]) / "custom_multivehicle_parsed_plan.json",
        Path(config["outputs"]["results_dir"]) / "custom_validation_result.json",
        Path(config["outputs"]["results_dir"]) / "custom_comparison_result.json",
        Path(config["outputs"]["results_dir"]) / "custom_multivehicle_result.json",
        Path(config["outputs"].get("maps_dir", "outputs/maps")) / "custom_interactive_route_map.html",
        Path(config["outputs"].get("maps_dir", "outputs/maps")) / "custom_multivehicle_interactive_route_map.html",
    ]
    for path in candidates:
        if path.exists():
            path.unlink()
    for sub in ("custom", "custom_multivehicle"):
        d = Path(config["outputs"].get("sumo_dir", "outputs/sumo")) / sub
        if d.exists():
            shutil.rmtree(d)


# ---------------------------------------------------------------------------
# SUMO utilities
# ---------------------------------------------------------------------------

def write_xml(root: ET.Element, output_path: str | Path) -> Path:
    path = Path(output_path)
    ensure_directory(path.parent)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def sumo_edge_id(source: str, target: str) -> str:
    return f"edge__{source}__{target}"


def get_sumo_origin(instance: dict[str, Any]) -> tuple[float, float]:
    lats = [float(loc["lat"]) for loc in instance["locations"]]
    lons = [float(loc["lon"]) for loc in instance["locations"]]
    return min(lats), min(lons)


def latlon_to_local_xy(
    lat: float, lon: float,
    origin_lat: float, origin_lon: float,
) -> tuple[float, float]:
    x = (lon - origin_lon) * 111_320.0 * math.cos(math.radians(origin_lat))
    y = (lat - origin_lat) * 110_540.0
    return x, y


def route_to_sumo_edges(route: list[str]) -> list[str]:
    return [sumo_edge_id(s, t) for s, t in zip(route[:-1], route[1:])]


def validate_sumo_route_edges(instance: dict[str, Any], route: list[str]) -> None:
    available = {(e["from"], e["to"]) for e in instance["edges"]}
    missing = [f"{s}->{t}" for s, t in zip(route[:-1], route[1:]) if (s, t) not in available]
    if missing:
        raise ValueError("Route contains edges not in SUMO network: " + ", ".join(missing))


def generate_background_routes_for_sumo(
    instance: dict[str, Any],
    number_of_vehicles: int,
    seed: int = 42,
) -> list[list[str]]:
    random.seed(seed)
    graph = build_weighted_graph(instance, metric="distance", ignore_blocked=True)
    nodes = list(graph.nodes)
    routes: list[list[str]] = []
    attempts = 0
    max_attempts = max(number_of_vehicles * 30, 100)

    while len(routes) < number_of_vehicles and attempts < max_attempts:
        attempts += 1
        src = random.choice(nodes)
        tgt = random.choice(nodes)
        if src == tgt or not nx.has_path(graph, src, tgt):
            continue
        route = nx.shortest_path(graph, src, tgt, weight="distance_m", method="dijkstra")
        if len(route) >= 3:
            routes.append(route)
    return routes


# Fixed colour palette for planned vehicles — indexed by vehicle position (0-based)
# Supports up to 5 vehicles; wraps for more.
_VEHICLE_SUMO_COLOURS = [
    "255,0,0",    # car1 — red
    "0,200,0",    # car2 — green
    "0,100,255",  # car3 — blue
    "255,165,0",  # car4 — orange
    "160,32,240", # car5 — purple
]


def generate_custom_sumo_simulation(
    instance: dict[str, Any],
    comparison: dict[str, Any],
    config: dict[str, Any],
    background_vehicle_count: int = 20,
) -> dict[str, Any]:
    """
    Export planning result to SUMO.
    Handles both single-vehicle and multi-vehicle.

    For multi-vehicle: reads per_vehicle_results from comparison dict.
    Each planned vehicle gets its own colour from _VEHICLE_SUMO_COLOURS.
    """
    instance_name = instance.get("instance_name", "custom")
    output_dir = ensure_directory(
        Path(config["outputs"].get("sumo_dir", "outputs/sumo")) / instance_name
    )

    node_file = output_dir / f"{instance_name}.nod.xml"
    edge_file = output_dir / f"{instance_name}.edg.xml"
    net_file  = output_dir / f"{instance_name}.net.xml"
    route_file = output_dir / f"{instance_name}.rou.xml"
    config_file = output_dir / f"{instance_name}.sumocfg"

    # 1. Nodes
    origin_lat, origin_lon = get_sumo_origin(instance)
    nodes_root = ET.Element("nodes")
    for loc in instance["locations"]:
        lat, lon = float(loc["lat"]), float(loc["lon"])
        x, y = latlon_to_local_xy(lat, lon, origin_lat, origin_lon)
        node_type = "traffic_light" if loc.get("has_traffic_signal") else "priority"
        ET.SubElement(nodes_root, "node", {
            "id": loc["id"], "x": f"{x:.3f}", "y": f"{y:.3f}", "type": node_type,
        })
    write_xml(nodes_root, node_file)

    # 2. Edges
    edges_root = ET.Element("edges")
    speed = float(config.get("sumo", {}).get("default_speed_m_per_s", 10.0))
    num_lanes = int(config.get("sumo", {}).get("num_lanes", 1))
    blocked_set = {(e["from"], e["to"]) for e in instance.get("blocked_edges", [])}

    for edge in instance["edges"]:
        if (edge["from"], edge["to"]) in blocked_set:
            continue
        ET.SubElement(edges_root, "edge", {
            "id": sumo_edge_id(edge["from"], edge["to"]),
            "from": edge["from"], "to": edge["to"],
            "priority": "1", "numLanes": str(num_lanes),
            "speed": f"{speed:.3f}",
            "length": f"{float(edge['distance_m']):.3f}",
        })
    write_xml(edges_root, edge_file)

    # 3. netconvert
    netconvert_path = config.get("sumo", {}).get("netconvert_path", "netconvert")
    completed = subprocess.run(
        [netconvert_path, "--node-files", str(node_file),
         "--edge-files", str(edge_file), "--output-file", str(net_file),
         "--no-turnarounds", "true"],
        capture_output=True, text=True, check=False,
    )
    netconvert_log = output_dir / f"{instance_name}_netconvert.log"
    netconvert_log.write_text(
        (completed.stdout or "") + "\n" + (completed.stderr or ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"netconvert failed. Check: {netconvert_log}")

    # 4. Routes
    routes_root = ET.Element("routes")
    per_vehicle_results = comparison.get("per_vehicle_results")

    if per_vehicle_results:
        # Multi-vehicle — one vType + vehicle per planned route
        for idx, vr in enumerate(per_vehicle_results):
            colour = _VEHICLE_SUMO_COLOURS[idx % len(_VEHICLE_SUMO_COLOURS)]
            vid = vr["vehicle_id"]
            route = vr["route"]
            if len(route) < 2:
                continue
            validate_sumo_route_edges(instance, route)
            ET.SubElement(routes_root, "vType", {
                "id": f"type_{vid}", "vClass": "passenger",
                "accel": "2.6", "decel": "4.5", "sigma": "0.2",
                "length": "30.0", "width": "5.0", "minGap": "2.5",
                "maxSpeed": "10", "color": colour,
            })
            v_elem = ET.SubElement(routes_root, "vehicle", {
                "id": f"ENHSP_{vid}", "type": f"type_{vid}",
                "depart": "5", "departLane": "best", "departSpeed": "max",
                "color": colour,
            })
            ET.SubElement(v_elem, "route", {
                "edges": " ".join(route_to_sumo_edges(route))
            })
    else:
        # Single-vehicle
        planner_route = comparison["planner_route"]
        if len(planner_route) < 2:
            raise ValueError("Planner route is too short for SUMO simulation.")
        validate_sumo_route_edges(instance, planner_route)
        ET.SubElement(routes_root, "vType", {
            "id": "planner_car", "vClass": "passenger",
            "accel": "2.6", "decel": "4.5", "sigma": "0.2",
            "length": "30.0", "width": "5.0", "minGap": "2.5",
            "maxSpeed": str(instance["vehicle"]["speed_m_per_s"]),
            "color": "255,0,0",
        })
        v_elem = ET.SubElement(routes_root, "vehicle", {
            "id": "ENHSP_planner_vehicle", "type": "planner_car",
            "depart": "5", "departLane": "best", "departSpeed": "max",
            "color": "255,0,0",
        })
        ET.SubElement(v_elem, "route", {
            "edges": " ".join(route_to_sumo_edges(planner_route))
        })

    # Background traffic
    ET.SubElement(routes_root, "vType", {
        "id": "background_car", "vClass": "passenger",
        "accel": "2.6", "decel": "4.5", "sigma": "0.7",
        "length": "20.0", "width": "4.0", "minGap": "2.5",
        "maxSpeed": str(instance["vehicle"]["speed_m_per_s"]),
        "color": "0,0,255",
    })
    if background_vehicle_count > 0:
        for idx, route in enumerate(
            generate_background_routes_for_sumo(instance, background_vehicle_count)
        ):
            try:
                validate_sumo_route_edges(instance, route)
            except ValueError:
                continue
            v_elem = ET.SubElement(routes_root, "vehicle", {
                "id": f"background_{idx}", "type": "background_car",
                "depart": str(10 + idx * 3),
                "departLane": "best", "departSpeed": "max", "color": "0,0,255",
            })
            ET.SubElement(v_elem, "route", {
                "edges": " ".join(route_to_sumo_edges(route))
            })
    write_xml(routes_root, route_file)

    # 5. SUMO config
    sim_end = int(config.get("sumo", {}).get("simulation_end_time", 3000))
    cfg_root = ET.Element("configuration")
    inp = ET.SubElement(cfg_root, "input")
    ET.SubElement(inp, "net-file", {"value": net_file.name})
    ET.SubElement(inp, "route-files", {"value": route_file.name})
    t = ET.SubElement(cfg_root, "time")
    ET.SubElement(t, "begin", {"value": "0"})
    ET.SubElement(t, "end", {"value": str(sim_end)})
    proc = ET.SubElement(cfg_root, "processing")
    ET.SubElement(proc, "ignore-route-errors", {"value": "true"})
    write_xml(cfg_root, config_file)

    # 6. SUMO validation
    sumo_path = config.get("sumo", {}).get("sumo_path", "sumo")
    sumo_val = subprocess.run(
        [sumo_path, "-c", config_file.name,
         "--no-step-log", "true", "--duration-log.disable", "true"],
        cwd=output_dir, capture_output=True, text=True, check=False,
    )
    sumo_log = output_dir / f"{instance_name}_sumo_validation.log"
    sumo_log.write_text(
        (sumo_val.stdout or "") + "\n" + (sumo_val.stderr or ""),
        encoding="utf-8",
    )

    # 7. PowerShell open script
    sumo_gui_path = config.get("sumo", {}).get("sumo_gui_path", "sumo-gui")
    open_script = output_dir / f"open_{instance_name}_sumo_gui.ps1"
    open_script.write_text(
        f'cd "{output_dir.resolve()}"\n'
        f'& "{sumo_gui_path}" -c "{config_file.name}"\n',
        encoding="utf-8",
    )

    result = {
        "success": sumo_val.returncode == 0,
        "output_dir": str(output_dir),
        "node_file": str(node_file),
        "edge_file": str(edge_file),
        "network_file": str(net_file),
        "route_file": str(route_file),
        "config_file": str(config_file),
        "netconvert_log": str(netconvert_log),
        "sumo_validation_log": str(sumo_log),
        "open_gui_script": str(open_script),
        "background_vehicle_count": background_vehicle_count,
    }
    save_json(result, output_dir / f"{instance_name}_sumo_result.json")
    return result