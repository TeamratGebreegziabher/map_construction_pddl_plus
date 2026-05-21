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
from pddl_generator import save_domain, save_problem
from plan_parser import parse_plan_file, save_parsed_plan
from planner_runner import run_single_problem


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


def update_edge_travel_times(instance: dict[str, Any], speed_m_per_s: float) -> dict[str, Any]:
    updated = dict(instance)
    updated_edges = []

    for edge in instance["edges"]:
        new_edge = dict(edge)
        distance = float(new_edge["distance_m"])
        new_edge["travel_time_s"] = round(distance / speed_m_per_s, 3)
        updated_edges.append(new_edge)

    updated["edges"] = updated_edges
    return updated


def build_weighted_graph(
    instance: dict[str, Any],
    metric: str = "distance",
    ignore_blocked: bool = True,
) -> nx.DiGraph:
    graph = nx.DiGraph()

    blocked_edges = {
        (edge["from"], edge["to"])
        for edge in instance.get("blocked_edges", [])
    }

    for location in instance["locations"]:
        graph.add_node(location["id"])

    for edge in instance["edges"]:
        source = edge["from"]
        target = edge["to"]

        if ignore_blocked and (source, target) in blocked_edges:
            continue

        distance = float(edge["distance_m"])
        travel_time = float(edge["travel_time_s"])

        if graph.has_edge(source, target):
            if distance < graph[source][target]["distance_m"]:
                graph[source][target].update(
                    distance_m=distance,
                    travel_time_s=travel_time,
                )
        else:
            graph.add_edge(
                source,
                target,
                distance_m=distance,
                travel_time_s=travel_time,
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
        edge = graph[source][target]
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
) -> dict[str, Any]:
    graph = build_weighted_graph(instance, metric=metric, ignore_blocked=True)

    start = instance["start"]
    goal = instance["goal"]

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
        graph,
        source=start,
        target=goal,
        weight=weight,
        method="dijkstra",
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
            if dijkstra_distance > 0
            else 0.0
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
            if dijkstra_battery > 0
            else 0.0
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
        "distance_gap_percent": (
            round(distance_gap_percent, 3)
            if distance_gap_percent is not None
            else None
        ),
        "battery_gap": round(battery_gap, 3) if battery_gap is not None else None,
        "battery_gap_percent": (
            round(battery_gap_percent, 3)
            if battery_gap_percent is not None
            else None
        ),
        "same_route_as_dijkstra": (
            validation["route"] == dijkstra["dijkstra_route"]
            if dijkstra["route_found"]
            else False
        ),
    }


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
        place_name,
        network_type=network_type,
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
    )

    graph = normalize_edge_lengths(graph)
    graph = keep_largest_strongly_connected_component(graph)
    subgraph = create_instance_subgraph(graph, max_nodes=max_nodes)

    mapping = create_pddl_location_mapping(subgraph)

    locations = extract_locations(subgraph, mapping)
    edges = extract_edges(
        subgraph,
        mapping,
        vehicle_speed_m_per_s=vehicle_speed_m_per_s,
    )

    if len(locations) < 2:
        raise ValueError("The selected map is too small. Increase max_nodes or choose another area.")

    instance = {
        "instance_name": "custom",
        "place_name": place_name,
        "network_type": network_type,
        "num_locations": len(locations),
        "num_edges": len(edges),
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
        "node_mapping": {
            str(original_id): pddl_id
            for original_id, pddl_id in mapping.items()
        },
    }

    processed_dir = ensure_directory(config["outputs"]["processed_data_dir"])
    save_json(instance, processed_dir / "custom_base_instance.json")

    return instance


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
    start_time = time.perf_counter()

    domain_path = save_domain(config)
    problem_path = save_problem(instance, config)

    planner_result = run_single_problem(
        config=config,
        domain_file=domain_path,
        problem_file=problem_path,
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

    validation_path = save_json(
        validation,
        results_dir / "custom_validation_result.json",
    )

    comparison_path = save_json(
        comparison,
        results_dir / "custom_comparison_result.json",
    )

    interactive_map_path = None

    if validation["valid"]:
        interactive_map_path = create_interactive_route_map(
            instance=instance,
            comparison=comparison,
            config=config,
        )

    total_runtime = time.perf_counter() - start_time

    return {
        "success": validation["valid"],
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
        "total_runtime_seconds": round(total_runtime, 3),
    }


def list_location_options(instance: dict[str, Any]) -> list[str]:
    options = []

    for location in instance["locations"]:
        label = (
            f"{location['id']} | "
            f"lat={float(location['lat']):.5f}, "
            f"lon={float(location['lon']):.5f}"
        )
        options.append(label)

    return options


def extract_location_id(option_label: str) -> str:
    return option_label.split("|")[0].strip()


def list_edge_options(instance: dict[str, Any]) -> list[str]:
    options = []

    for edge in instance["edges"]:
        road_name = edge.get("name", "")
        highway = edge.get("highway", "")

        label = (
            f"{edge['from']} -> {edge['to']} | "
            f"{float(edge['distance_m']):.1f} m | "
            f"{road_name} | {highway}"
        )
        options.append(label)

    return options


def extract_edge_pair(option_label: str) -> tuple[str, str]:
    pair = option_label.split("|")[0].strip()
    source, target = pair.split("->")
    return source.strip(), target.strip()


def build_blocked_edge_objects(
    selected_edge_labels: list[str],
    instance: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_pairs = {extract_edge_pair(label) for label in selected_edge_labels}

    blocked = []

    for edge in instance["edges"]:
        pair = (edge["from"], edge["to"])

        if pair in selected_pairs:
            blocked.append(
                {
                    "from": edge["from"],
                    "to": edge["to"],
                    "distance_m": edge["distance_m"],
                    "travel_time_s": edge["travel_time_s"],
                    "name": edge.get("name", ""),
                    "highway": edge.get("highway", ""),
                    "reason": "Selected by user in interactive app",
                }
            )

    return blocked


def clear_custom_outputs(config: dict[str, Any]) -> None:
    candidates = [
        Path(config["planning"]["problem_dir"]) / "problem_custom.pddl",
        Path(config["outputs"]["plans_dir"]) / "custom_plan.txt",
        Path(config["outputs"]["logs_dir"]) / "custom_planner.log",
        Path(config["outputs"]["results_dir"]) / "custom_parsed_plan.json",
        Path(config["outputs"]["results_dir"]) / "custom_validation_result.json",
        Path(config["outputs"]["results_dir"]) / "custom_comparison_result.json",
        Path(config["outputs"].get("maps_dir", "outputs/maps")) / "custom_interactive_route_map.html",
    ]

    for path in candidates:
        if path.exists():
            path.unlink()

    custom_sumo_dir = Path(config["outputs"].get("sumo_dir", "outputs/sumo")) / "custom"

    if custom_sumo_dir.exists():
        shutil.rmtree(custom_sumo_dir)
def prepare_custom_map_from_polygon(
    polygon_geojson: dict[str, Any],
    place_name: str,
    network_type: str,
    max_nodes: int,
    vehicle_speed_m_per_s: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract and simplify an OSM road network from a user-drawn polygon/rectangle.
    """
    ox.settings.use_cache = True
    ox.settings.log_console = False

    polygon = shape(polygon_geojson)

    if polygon.is_empty:
        raise ValueError("Selected polygon is empty.")

    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    graph = ox.graph_from_polygon(
        polygon,
        network_type=network_type,
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
    )

    graph = normalize_edge_lengths(graph)
    graph = keep_largest_strongly_connected_component(graph)
    subgraph = create_instance_subgraph(graph, max_nodes=max_nodes)

    mapping = create_pddl_location_mapping(subgraph)

    locations = extract_locations(subgraph, mapping)
    edges = extract_edges(
        subgraph,
        mapping,
        vehicle_speed_m_per_s=vehicle_speed_m_per_s,
    )

    if len(locations) < 2:
        raise ValueError(
            "The selected area produced too few locations. "
            "Draw a larger area or increase max_nodes."
        )

    instance = {
        "instance_name": "custom",
        "place_name": place_name,
        "network_type": network_type,
        "selection_mode": "drawn_polygon",
        "num_locations": len(locations),
        "num_edges": len(edges),
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
        "selected_polygon_geojson": polygon_geojson,
    }

    processed_dir = ensure_directory(config["outputs"]["processed_data_dir"])
    save_json(instance, processed_dir / "custom_base_instance.json")

    return instance
def find_nearest_location_id(
    instance: dict[str, Any],
    clicked_lat: float,
    clicked_lon: float,
) -> str:
    """
    Find the nearest extracted planning location to a clicked map coordinate.
    Uses simple squared lat/lon distance, enough for local selection.
    """
    best_location_id = None
    best_distance = float("inf")

    for location in instance["locations"]:
        lat = float(location["lat"])
        lon = float(location["lon"])

        distance = (lat - clicked_lat) ** 2 + (lon - clicked_lon) ** 2

        if distance < best_distance:
            best_distance = distance
            best_location_id = location["id"]

    if best_location_id is None:
        raise ValueError("No nearest location found.")

    return best_location_id
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
    latitudes = [float(location["lat"]) for location in instance["locations"]]
    longitudes = [float(location["lon"]) for location in instance["locations"]]

    return min(latitudes), min(longitudes)


def latlon_to_local_xy(
    lat: float,
    lon: float,
    origin_lat: float,
    origin_lon: float,
) -> tuple[float, float]:
    meters_per_degree_lat = 110_540.0
    meters_per_degree_lon = 111_320.0 * math.cos(math.radians(origin_lat))

    x = (lon - origin_lon) * meters_per_degree_lon
    y = (lat - origin_lat) * meters_per_degree_lat

    return x, y


def route_to_sumo_edges(route: list[str]) -> list[str]:
    return [
        sumo_edge_id(source, target)
        for source, target in zip(route[:-1], route[1:])
    ]


def validate_sumo_route_edges(instance: dict[str, Any], route: list[str]) -> None:
    available_edges = {
        (edge["from"], edge["to"])
        for edge in instance["edges"]
    }

    missing_edges = []

    for source, target in zip(route[:-1], route[1:]):
        if (source, target) not in available_edges:
            missing_edges.append(f"{source}->{target}")

    if missing_edges:
        raise ValueError(
            "Planner route contains edges that are not in the SUMO network: "
            + ", ".join(missing_edges)
        )


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

        source = random.choice(nodes)
        target = random.choice(nodes)

        if source == target:
            continue

        if not nx.has_path(graph, source, target):
            continue

        route = nx.shortest_path(
            graph,
            source=source,
            target=target,
            weight="distance_m",
            method="dijkstra",
        )

        if len(route) >= 3:
            routes.append(route)

    return routes


def generate_custom_sumo_simulation(
    instance: dict[str, Any],
    comparison: dict[str, Any],
    config: dict[str, Any],
    background_vehicle_count: int = 20,
) -> dict[str, Any]:
    """
    Export the user-selected custom planning result to SUMO.
    This is applied after PDDL+ planning.
    """
    instance_name = "custom"

    output_root = ensure_directory(config["outputs"].get("sumo_dir", "outputs/sumo"))
    output_dir = ensure_directory(output_root / instance_name)

    node_file = output_dir / "custom.nod.xml"
    edge_file = output_dir / "custom.edg.xml"
    net_file = output_dir / "custom.net.xml"
    route_file = output_dir / "custom.rou.xml"
    config_file = output_dir / "custom.sumocfg"

    # 1. SUMO nodes
    origin_lat, origin_lon = get_sumo_origin(instance)

    nodes_root = ET.Element("nodes")

    for location in instance["locations"]:
        lat = float(location["lat"])
        lon = float(location["lon"])
        x, y = latlon_to_local_xy(lat, lon, origin_lat, origin_lon)

        ET.SubElement(
            nodes_root,
            "node",
            {
                "id": location["id"],
                "x": f"{x:.3f}",
                "y": f"{y:.3f}",
                "type": "priority",
            },
        )

    write_xml(nodes_root, node_file)

    # 2. SUMO edges
    edges_root = ET.Element("edges")

    speed = float(config.get("sumo", {}).get("default_speed_m_per_s", 10.0))
    num_lanes = int(config.get("sumo", {}).get("num_lanes", 1))

    blocked_edges = {
        (edge["from"], edge["to"])
        for edge in instance.get("blocked_edges", [])
    }

    for edge in instance["edges"]:
        source = edge["from"]
        target = edge["to"]

        if (source, target) in blocked_edges:
            continue

        ET.SubElement(
            edges_root,
            "edge",
            {
                "id": sumo_edge_id(source, target),
                "from": source,
                "to": target,
                "priority": "1",
                "numLanes": str(num_lanes),
                "speed": f"{speed:.3f}",
                "length": f"{float(edge['distance_m']):.3f}",
            },
        )

    write_xml(edges_root, edge_file)

    # 3. netconvert
    netconvert_path = config.get("sumo", {}).get("netconvert_path", "netconvert")

    netconvert_command = [
        netconvert_path,
        "--node-files",
        str(node_file),
        "--edge-files",
        str(edge_file),
        "--output-file",
        str(net_file),
        "--no-turnarounds",
        "true",
    ]

    completed = subprocess.run(
        netconvert_command,
        capture_output=True,
        text=True,
        check=False,
    )

    netconvert_log = output_dir / "custom_netconvert.log"
    netconvert_log.write_text(
        (completed.stdout or "") + "\n" + (completed.stderr or ""),
        encoding="utf-8",
    )

    if completed.returncode != 0:
        raise RuntimeError(f"netconvert failed. Check: {netconvert_log}")

    # 4. SUMO route file
    planner_route = comparison["planner_route"]

    if len(planner_route) < 2:
        raise ValueError("Planner route is too short for SUMO simulation.")

    validate_sumo_route_edges(instance, planner_route)

    routes_root = ET.Element("routes")

    ET.SubElement(
        routes_root,
        "vType",
        {
            "id": "planner_car",
            "vClass": "passenger",
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.2",
            "length": "30.0",
            "width": "5.0",
            "minGap": "2.5",
            "maxSpeed": str(instance["vehicle"]["speed_m_per_s"]),
            "color": "255,0,0",
        },
    )

    ET.SubElement(
        routes_root,
        "vType",
        {
            "id": "background_car",
            "vClass": "passenger",
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.7",
            "length": "20.0",
            "width": "4.0",
            "minGap": "2.5",
            "maxSpeed": str(instance["vehicle"]["speed_m_per_s"]),
            "color": "0,0,255",
        },
    )

    planner_vehicle = ET.SubElement(
        routes_root,
        "vehicle",
        {
            "id": "ENHSP_planner_vehicle",
            "type": "planner_car",
            "depart": "5",
            "departLane": "best",
            "departSpeed": "max",
            "color": "255,0,0",
        },
    )

    ET.SubElement(
        planner_vehicle,
        "route",
        {
            "edges": " ".join(route_to_sumo_edges(planner_route)),
        },
    )

    if background_vehicle_count > 0:
        background_routes = generate_background_routes_for_sumo(
            instance=instance,
            number_of_vehicles=background_vehicle_count,
        )

        for index, route in enumerate(background_routes):
            try:
                validate_sumo_route_edges(instance, route)
            except ValueError:
                continue

            vehicle = ET.SubElement(
                routes_root,
                "vehicle",
                {
                    "id": f"background_{index}",
                    "type": "background_car",
                    "depart": str(10 + index * 3),
                    "departLane": "best",
                    "departSpeed": "max",
                    "color": "0,0,255",
                },
            )

            ET.SubElement(
                vehicle,
                "route",
                {
                    "edges": " ".join(route_to_sumo_edges(route)),
                },
            )

    write_xml(routes_root, route_file)

    # 5. SUMO config
    simulation_end_time = int(config.get("sumo", {}).get("simulation_end_time", 3000))

    config_root = ET.Element("configuration")

    input_element = ET.SubElement(config_root, "input")

    ET.SubElement(
        input_element,
        "net-file",
        {
            "value": net_file.name,
        },
    )

    ET.SubElement(
        input_element,
        "route-files",
        {
            "value": route_file.name,
        },
    )

    time_element = ET.SubElement(config_root, "time")

    ET.SubElement(
        time_element,
        "begin",
        {
            "value": "0",
        },
    )

    ET.SubElement(
        time_element,
        "end",
        {
            "value": str(simulation_end_time),
        },
    )

    processing_element = ET.SubElement(config_root, "processing")

    ET.SubElement(
        processing_element,
        "ignore-route-errors",
        {
            "value": "true",
        },
    )

    write_xml(config_root, config_file)

    # 6. Validate SUMO simulation
    sumo_path = config.get("sumo", {}).get("sumo_path", "sumo")

    sumo_command = [
        sumo_path,
        "-c",
        config_file.name,
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]

    sumo_validation = subprocess.run(
        sumo_command,
        cwd=output_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    sumo_log = output_dir / "custom_sumo_validation.log"
    sumo_log.write_text(
        (sumo_validation.stdout or "") + "\n" + (sumo_validation.stderr or ""),
        encoding="utf-8",
    )

    # 7. PowerShell script to open SUMO-GUI
    sumo_gui_path = config.get("sumo", {}).get("sumo_gui_path", "sumo-gui")
    open_script = output_dir / "open_custom_sumo_gui.ps1"

    open_script.write_text(
        f'cd "{output_dir.resolve()}"\n'
        f'& "{sumo_gui_path}" -c "custom.sumocfg"\n',
        encoding="utf-8",
    )

    result = {
        "success": sumo_validation.returncode == 0,
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

    save_json(result, output_dir / "custom_sumo_result.json")

    return result
