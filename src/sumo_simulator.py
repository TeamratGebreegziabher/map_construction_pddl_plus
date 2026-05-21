from __future__ import annotations

import json
import math
import random
import subprocess
import xml.etree.ElementTree as ET
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


def save_json(data: Any, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return output_path


def write_xml(root: ET.Element, output_path: str | Path) -> Path:
    path = Path(output_path)
    ensure_directory(path.parent)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(path, encoding="utf-8", xml_declaration=True)

    return path


def load_instance(instance_name: str, config: dict[str, Any]) -> dict[str, Any]:
    processed_dir = Path(config["outputs"]["processed_data_dir"])
    return load_json(processed_dir / f"{instance_name}_instance.json")


def load_comparisons(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results_dir = Path(config["outputs"]["results_dir"])
    comparisons = load_json(results_dir / "comparison_results.json")
    return {item["instance_name"]: item for item in comparisons}


def sumo_edge_id(source: str, target: str) -> str:
    return f"edge__{source}__{target}"


def get_origin(instance: dict[str, Any]) -> tuple[float, float]:
    latitudes = [float(location["lat"]) for location in instance["locations"]]
    longitudes = [float(location["lon"]) for location in instance["locations"]]
    return min(latitudes), min(longitudes)


def latlon_to_xy(
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


def build_graph(instance: dict[str, Any]) -> nx.DiGraph:
    graph = nx.DiGraph()

    for location in instance["locations"]:
        graph.add_node(location["id"])

    for edge in instance["edges"]:
        source = edge["from"]
        target = edge["to"]
        distance = float(edge["distance_m"])

        if graph.has_edge(source, target):
            if distance < graph[source][target]["distance_m"]:
                graph[source][target]["distance_m"] = distance
        else:
            graph.add_edge(source, target, distance_m=distance)

    return graph


def route_to_sumo_edges(route: list[str]) -> list[str]:
    return [
        sumo_edge_id(source, target)
        for source, target in zip(route[:-1], route[1:])
    ]


def generate_nodes_file(instance: dict[str, Any], output_dir: Path) -> Path:
    instance_name = instance["instance_name"]
    output_path = output_dir / f"{instance_name}.nod.xml"

    origin_lat, origin_lon = get_origin(instance)

    root = ET.Element("nodes")

    for location in instance["locations"]:
        loc_id = location["id"]
        lat = float(location["lat"])
        lon = float(location["lon"])

        x, y = latlon_to_xy(lat, lon, origin_lat, origin_lon)

        ET.SubElement(
            root,
            "node",
            {
                "id": loc_id,
                "x": f"{x:.3f}",
                "y": f"{y:.3f}",
                "type": "priority",
            },
        )

    write_xml(root, output_path)

    print(f"Saved SUMO nodes: {output_path}")

    return output_path


def generate_edges_file(
    instance: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
) -> Path:
    instance_name = instance["instance_name"]
    output_path = output_dir / f"{instance_name}.edg.xml"

    sumo_config = config.get("sumo", {})
    speed = float(sumo_config.get("default_speed_m_per_s", 10.0))
    num_lanes = int(sumo_config.get("num_lanes", 1))

    root = ET.Element("edges")

    for edge in instance["edges"]:
        source = edge["from"]
        target = edge["to"]

        ET.SubElement(
            root,
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

    write_xml(root, output_path)

    print(f"Saved SUMO edges: {output_path}")

    return output_path


def run_netconvert(
    instance_name: str,
    node_file: Path,
    edge_file: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> Path:
    netconvert_path = config.get("sumo", {}).get("netconvert_path", "netconvert")
    net_file = output_dir / f"{instance_name}.net.xml"

    command = [
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

    print("Running netconvert:")
    print(" ".join(command))

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    log_path = output_dir / f"{instance_name}_netconvert.log"
    log_path.write_text(
        (completed.stdout or "") + "\n" + (completed.stderr or ""),
        encoding="utf-8",
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"netconvert failed for {instance_name}. Check log: {log_path}"
        )

    print(f"Saved SUMO network: {net_file}")

    return net_file


def generate_background_routes(
    instance: dict[str, Any],
    number_of_vehicles: int,
    seed: int = 42,
) -> list[list[str]]:
    """
    Generate random shortest routes for background vehicles.
    """
    random.seed(seed)

    graph = build_graph(instance)
    nodes = list(graph.nodes)

    routes: list[list[str]] = []
    attempts = 0
    max_attempts = number_of_vehicles * 20

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

        if len(route) < 3:
            continue

        routes.append(route)

    return routes


def generate_route_file(
    instance: dict[str, Any],
    comparison: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
) -> Path:
    instance_name = instance["instance_name"]
    output_path = output_dir / f"{instance_name}.rou.xml"

    root = ET.Element("routes")

    ET.SubElement(
        root,
        "vType",
        {
            "id": "planner_car",
            "vClass": "passenger",
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.2",
            "length": "15.0",
            "maxSpeed": str(instance["vehicle"]["speed_m_per_s"]),
            "color": "255,0,0",
        },
    )

    ET.SubElement(
        root,
        "vType",
        {
            "id": "background_car",
            "vClass": "passenger",
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.7",
            "length": "10.0",
            "maxSpeed": str(instance["vehicle"]["speed_m_per_s"]),
            "color": "0,0,255",
        },
    )

    planner_route = comparison["planner_route"]
    planner_edges = route_to_sumo_edges(planner_route)

    planner_vehicle = ET.SubElement(
        root,
        "vehicle",
        {
            "id": "ENHSP_planner_vehicle",
            "type": "planner_car",
            "depart": "0",
            "departLane": "best",
            "departSpeed": "max",
        },
    )

    ET.SubElement(
        planner_vehicle,
        "route",
        {
            "edges": " ".join(planner_edges),
        },
    )

    sumo_config = config.get("sumo", {})
    add_background = bool(sumo_config.get("add_background_traffic", True))
    background_counts = sumo_config.get("background_vehicles", {})
    number_of_background_vehicles = int(background_counts.get(instance_name, 0))

    if add_background and number_of_background_vehicles > 0:
        background_routes = generate_background_routes(
            instance=instance,
            number_of_vehicles=number_of_background_vehicles,
            seed=42,
        )

        for index, route in enumerate(background_routes):
            vehicle = ET.SubElement(
                root,
                "vehicle",
                {
                    "id": f"background_{index}",
                    "type": "background_car",
                    "depart": str(index * 3),
                    "departLane": "best",
                    "departSpeed": "max",
                },
            )

            ET.SubElement(
                vehicle,
                "route",
                {
                    "edges": " ".join(route_to_sumo_edges(route)),
                },
            )

    write_xml(root, output_path)

    print(f"Saved SUMO routes: {output_path}")

    return output_path


def generate_sumo_config(
    instance_name: str,
    net_file: Path,
    route_file: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> Path:
    output_path = output_dir / f"{instance_name}.sumocfg"

    simulation_end_time = int(config.get("sumo", {}).get("simulation_end_time", 2000))

    root = ET.Element("configuration")

    input_element = ET.SubElement(root, "input")

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

    time_element = ET.SubElement(root, "time")

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

    processing_element = ET.SubElement(root, "processing")

    ET.SubElement(
        processing_element,
        "ignore-route-errors",
        {
            "value": "true",
        },
    )

    report_element = ET.SubElement(root, "report")

    ET.SubElement(
        report_element,
        "verbose",
        {
            "value": "true",
        },
    )

    write_xml(root, output_path)

    print(f"Saved SUMO config: {output_path}")

    return output_path


def run_sumo_validation(
    instance_name: str,
    sumo_config_file: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    sumo_path = config.get("sumo", {}).get("sumo_path", "sumo")

    command = [
        sumo_path,
        "-c",
        sumo_config_file.name,
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]

    print("Running SUMO validation:")
    print(" ".join(command))

    completed = subprocess.run(
        command,
        cwd=output_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    log_path = output_dir / f"{instance_name}_sumo_validation.log"
    log_path.write_text(
        (completed.stdout or "") + "\n" + (completed.stderr or ""),
        encoding="utf-8",
    )

    success = completed.returncode == 0

    if success:
        print(f"SUMO validation succeeded for {instance_name}.")
    else:
        print(f"SUMO validation failed for {instance_name}. Check {log_path}")

    return {
        "success": success,
        "return_code": completed.returncode,
        "log_file": str(log_path),
    }


def generate_open_gui_script(
    instance_name: str,
    output_dir: Path,
    config: dict[str, Any],
) -> Path:
    """
    Create a PowerShell script to open SUMO-GUI easily.
    """
    sumo_gui_path = config.get("sumo", {}).get("sumo_gui_path", "sumo-gui")
    config_file = f"{instance_name}.sumocfg"

    script_path = output_dir / f"open_{instance_name}_sumo_gui.ps1"

    script_text = f"""cd "{output_dir.resolve()}"
& "{sumo_gui_path}" -c "{config_file}"
"""

    script_path.write_text(script_text, encoding="utf-8")

    return script_path


def generate_one_sumo_simulation(
    instance_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    instance = load_instance(instance_name, config)
    comparisons = load_comparisons(config)
    comparison = comparisons[instance_name]

    sumo_root = ensure_directory(config["outputs"].get("sumo_dir", "outputs/sumo"))
    output_dir = ensure_directory(sumo_root / instance_name)

    node_file = generate_nodes_file(instance, output_dir)
    edge_file = generate_edges_file(instance, output_dir, config)
    net_file = run_netconvert(instance_name, node_file, edge_file, output_dir, config)
    route_file = generate_route_file(instance, comparison, output_dir, config)

    sumo_config_file = generate_sumo_config(
        instance_name=instance_name,
        net_file=net_file,
        route_file=route_file,
        output_dir=output_dir,
        config=config,
    )

    validation = run_sumo_validation(
        instance_name=instance_name,
        sumo_config_file=sumo_config_file,
        output_dir=output_dir,
        config=config,
    )

    open_gui_script = generate_open_gui_script(
        instance_name=instance_name,
        output_dir=output_dir,
        config=config,
    )

    return {
        "instance_name": instance_name,
        "output_dir": str(output_dir),
        "node_file": str(node_file),
        "edge_file": str(edge_file),
        "network_file": str(net_file),
        "route_file": str(route_file),
        "sumo_config_file": str(sumo_config_file),
        "open_gui_script": str(open_gui_script),
        "validation": validation,
    }


def run_sumo_simulations(config: dict[str, Any]) -> dict[str, Any]:
    results = []

    for instance_name in ["small", "medium", "large"]:
        print("\n" + "=" * 70)
        print(f"Generating SUMO simulation for: {instance_name}")
        print("=" * 70)

        result = generate_one_sumo_simulation(instance_name, config)
        results.append(result)

    results_dir = ensure_directory(config["outputs"]["results_dir"])
    summary_path = results_dir / "sumo_simulation_results.json"
    save_json(results, summary_path)

    print(f"\nSaved SUMO simulation summary: {summary_path}")

    return {
        "results": results,
        "summary_path": str(summary_path),
    }