from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

from map_extractor import ensure_directory, slugify_place_name


def load_raw_graph(config: dict[str, Any]) -> nx.MultiDiGraph:
    """
    Load the raw OSMnx graph saved during Stage Two.
    """
    place_name = config["map"]["place_name"]
    safe_name = slugify_place_name(place_name)

    raw_data_dir = Path(config["outputs"]["raw_data_dir"])
    graph_path = raw_data_dir / f"{safe_name}_raw.graphml"

    if not graph_path.exists():
        raise FileNotFoundError(
            f"Raw graph not found: {graph_path}\n"
            "Run Stage Two first to download and save the OpenStreetMap graph."
        )

    print(f"Loading raw graph from: {graph_path}")

    graph = ox.io.load_graphml(graph_path)

    print("Raw graph loaded.")
    print(f"Raw nodes: {graph.number_of_nodes()}")
    print(f"Raw edges: {graph.number_of_edges()}")

    return graph


def to_float(value: Any, default: float = 0.0) -> float:
    """
    Convert OSMnx/GraphML attribute values to float safely.
    """
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_edge_lengths(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Ensure all edge lengths are numeric floats.
    Edges without usable length receive a fallback value.
    """
    fallback_length = 1.0

    for _, _, _, data in graph.edges(keys=True, data=True):
        length = to_float(data.get("length"), fallback_length)

        if length <= 0:
            length = fallback_length

        data["length"] = length

    return graph


def keep_largest_strongly_connected_component(
    graph: nx.MultiDiGraph,
) -> nx.MultiDiGraph:
    """
    Keep the largest strongly connected component.

    Strong connectivity is useful because, in a directed road network,
    it ensures that selected nodes can reach each other by following
    directed edges.

    If the graph is too fragmented, we fall back to the largest weakly
    connected component.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("The graph has no nodes.")

    if nx.is_directed(graph):
        strong_components = list(nx.strongly_connected_components(graph))

        if strong_components:
            largest_strong = max(strong_components, key=len)

            if len(largest_strong) >= 5:
                subgraph = graph.subgraph(largest_strong).copy()
                print(
                    "Kept largest strongly connected component: "
                    f"{subgraph.number_of_nodes()} nodes, "
                    f"{subgraph.number_of_edges()} edges"
                )
                return subgraph

        weak_components = list(nx.weakly_connected_components(graph))
        largest_weak = max(weak_components, key=len)
        subgraph = graph.subgraph(largest_weak).copy()

        print(
            "Strong component too small. "
            "Kept largest weakly connected component instead: "
            f"{subgraph.number_of_nodes()} nodes, "
            f"{subgraph.number_of_edges()} edges"
        )

        return subgraph

    components = list(nx.connected_components(graph))
    largest = max(components, key=len)
    subgraph = graph.subgraph(largest).copy()

    print(
        "Kept largest connected component: "
        f"{subgraph.number_of_nodes()} nodes, "
        f"{subgraph.number_of_edges()} edges"
    )

    return subgraph


def choose_seed_node(graph: nx.MultiDiGraph) -> Any:
    """
    Choose a central-ish seed node.

    We use the node with the highest degree because it is likely to be
    near a denser part of the road network.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot choose seed node from an empty graph.")

    return max(graph.nodes, key=lambda node: graph.degree(node))


def bfs_node_sample(graph: nx.MultiDiGraph, max_nodes: int) -> list[Any]:
    """
    Select up to max_nodes nodes using BFS from a seed node.

    We run BFS on the undirected version to obtain a compact local subgraph.
    """
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive.")

    undirected_graph = graph.to_undirected()
    seed = choose_seed_node(graph)

    selected: list[Any] = []
    visited = set()
    queue: deque[Any] = deque([seed])

    while queue and len(selected) < max_nodes:
        node = queue.popleft()

        if node in visited:
            continue

        visited.add(node)
        selected.append(node)

        neighbors = list(undirected_graph.neighbors(node))
        neighbors.sort(key=lambda n: undirected_graph.degree(n), reverse=True)

        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append(neighbor)

    return selected


def create_instance_subgraph(
    graph: nx.MultiDiGraph,
    max_nodes: int,
) -> nx.MultiDiGraph:
    """
    Create a smaller subgraph with at most max_nodes nodes.
    """
    selected_nodes = bfs_node_sample(graph, max_nodes=max_nodes)
    subgraph = graph.subgraph(selected_nodes).copy()

    # Remove isolated nodes, if any appeared after induced subgraph creation.
    isolated_nodes = list(nx.isolates(subgraph.to_undirected()))
    if isolated_nodes:
        subgraph.remove_nodes_from(isolated_nodes)

    print(
        f"Created subgraph with target max_nodes={max_nodes}: "
        f"{subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges"
    )

    return subgraph


def build_simple_digraph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    """
    Convert MultiDiGraph to DiGraph by keeping the shortest edge
    between each ordered pair of nodes.

    This is useful for shortest-path calculations and for selecting
    start/goal pairs.
    """
    simple_graph = nx.DiGraph()

    for node, data in graph.nodes(data=True):
        simple_graph.add_node(node, **data)

    for u, v, data in graph.edges(data=True):
        length = to_float(data.get("length"), default=1.0)

        if u == v:
            continue

        if simple_graph.has_edge(u, v):
            existing_length = simple_graph[u][v]["length"]
            if length < existing_length:
                simple_graph[u][v]["length"] = length
        else:
            simple_graph.add_edge(u, v, length=length)

    return simple_graph


def choose_start_goal(graph: nx.MultiDiGraph) -> tuple[Any, Any, float]:
    """
    Choose a start and goal pair with a non-trivial shortest-path distance.

    We search for the reachable pair with the largest shortest-path distance
    inside the selected subgraph.
    """
    simple_graph = build_simple_digraph(graph)

    best_start = None
    best_goal = None
    best_distance = -1.0

    for source in simple_graph.nodes:
        lengths = nx.single_source_dijkstra_path_length(
            simple_graph,
            source,
            weight="length",
        )

        for target, distance in lengths.items():
            if source == target:
                continue

            if distance > best_distance:
                best_start = source
                best_goal = target
                best_distance = float(distance)

    if best_start is None or best_goal is None:
        raise ValueError(
            "Could not find a reachable start-goal pair in the selected graph."
        )

    print(
        f"Selected start-goal pair with approximate shortest distance "
        f"{best_distance:.2f} meters"
    )

    return best_start, best_goal, best_distance


def create_pddl_location_mapping(graph: nx.MultiDiGraph) -> dict[Any, str]:
    """
    Rename OSM node IDs to PDDL-safe location names.
    """
    mapping = {}

    for index, node in enumerate(graph.nodes):
        mapping[node] = f"loc_{index}"

    return mapping


def extract_locations(
    graph: nx.MultiDiGraph,
    mapping: dict[Any, str],
) -> list[dict[str, Any]]:
    """
    Extract location objects with coordinates.
    """
    locations = []

    for original_id, pddl_id in mapping.items():
        data = graph.nodes[original_id]

        lon = to_float(data.get("x"), default=0.0)
        lat = to_float(data.get("y"), default=0.0)

        locations.append(
            {
                "id": pddl_id,
                "osm_id": str(original_id),
                "lat": lat,
                "lon": lon,
            }
        )

    return locations


def extract_edges(
    graph: nx.MultiDiGraph,
    mapping: dict[Any, str],
    vehicle_speed_m_per_s: float,
) -> list[dict[str, Any]]:
    """
    Extract directed road segments.

    If there are multiple edges between the same two locations,
    we keep the shortest distance.
    """
    best_edges: dict[tuple[str, str], dict[str, Any]] = {}

    for u, v, data in graph.edges(data=True):
        if u not in mapping or v not in mapping:
            continue

        if u == v:
            continue

        from_location = mapping[u]
        to_location = mapping[v]
        distance_m = to_float(data.get("length"), default=1.0)

        if distance_m <= 0:
            continue

        travel_time_s = distance_m / vehicle_speed_m_per_s

        key = (from_location, to_location)

        edge_data = {
            "from": from_location,
            "to": to_location,
            "distance_m": round(distance_m, 3),
            "travel_time_s": round(travel_time_s, 3),
            "name": str(data.get("name", "")),
            "highway": str(data.get("highway", "")),
        }

        if key not in best_edges:
            best_edges[key] = edge_data
        else:
            if distance_m < best_edges[key]["distance_m"]:
                best_edges[key] = edge_data

    return list(best_edges.values())


def save_instance_json(
    instance: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """
    Save processed instance as JSON.
    """
    processed_data_dir = ensure_directory(config["outputs"]["processed_data_dir"])
    instance_name = instance["instance_name"]

    output_path = processed_data_dir / f"{instance_name}_instance.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(instance, file, indent=2)

    print(f"Saved processed instance to: {output_path}")

    return output_path


def save_processed_graphml(
    graph: nx.MultiDiGraph,
    instance_name: str,
    config: dict[str, Any],
) -> Path:
    """
    Save processed subgraph as GraphML.
    """
    processed_data_dir = ensure_directory(config["outputs"]["processed_data_dir"])
    output_path = processed_data_dir / f"{instance_name}_graph.graphml"

    ox.io.save_graphml(graph, filepath=output_path)

    print(f"Saved processed GraphML to: {output_path}")

    return output_path


def plot_processed_graph(
    graph: nx.MultiDiGraph,
    instance: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """
    Save visualization of processed instance.
    """
    figures_dir = ensure_directory(config["outputs"]["figures_dir"])
    instance_name = instance["instance_name"]

    output_path = figures_dir / f"{instance_name}_processed_graph.png"

    fig, ax = ox.plot_graph(
        graph,
        show=False,
        close=False,
        node_size=20,
        edge_linewidth=1.0,
        bgcolor="white",
        node_color="black",
        edge_color="gray",
    )

    ax.set_title(
        f"{instance_name.capitalize()} processed graph "
        f"({instance['num_locations']} nodes, {instance['num_edges']} edges)",
        fontsize=12,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved processed graph visualization to: {output_path}")

    return output_path


def build_instance(
    graph: nx.MultiDiGraph,
    instance_name: str,
    max_nodes: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Build one processed planning instance.
    """
    vehicle_speed = float(config["vehicle"]["speed_m_per_s"])

    subgraph = create_instance_subgraph(graph, max_nodes=max_nodes)

    if subgraph.number_of_nodes() < 2:
        raise ValueError(
            f"Instance {instance_name} has fewer than 2 nodes. "
            "Choose a larger map or increase max_nodes."
        )

    start_osm, goal_osm, shortest_distance = choose_start_goal(subgraph)

    mapping = create_pddl_location_mapping(subgraph)

    locations = extract_locations(subgraph, mapping)
    edges = extract_edges(
        subgraph,
        mapping,
        vehicle_speed_m_per_s=vehicle_speed,
    )

    instance = {
        "instance_name": instance_name,
        "place_name": config["map"]["place_name"],
        "network_type": config["map"]["network_type"],
        "num_locations": len(locations),
        "num_edges": len(edges),
        "vehicle": {
            "name": config["vehicle"]["name"],
            "initial_battery": float(config["vehicle"]["initial_battery"]),
            "speed_m_per_s": vehicle_speed,
            "battery_consumption_per_meter": float(
                config["vehicle"]["battery_consumption_per_meter"]
            ),
        },
        "start": mapping[start_osm],
        "goal": mapping[goal_osm],
        "estimated_shortest_distance_m": round(shortest_distance, 3),
        "locations": locations,
        "edges": edges,
        "node_mapping": {
            str(original_id): pddl_id for original_id, pddl_id in mapping.items()
        },
    }

    save_instance_json(instance, config)
    save_processed_graphml(subgraph, instance_name, config)
    plot_processed_graph(subgraph, instance, config)

    return instance


def run_graph_processing(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Complete Stage Three:
    - load raw graph
    - normalize edge lengths
    - keep largest useful component
    - generate small, medium, and large instances
    """
    raw_graph = load_raw_graph(config)
    raw_graph = normalize_edge_lengths(raw_graph)
    graph = keep_largest_strongly_connected_component(raw_graph)

    instance_specs = [
        ("small", int(config["map"]["max_nodes_small"])),
        ("medium", int(config["map"]["max_nodes_medium"])),
        ("large", int(config["map"]["max_nodes_large"])),
    ]

    instances = []

    for instance_name, max_nodes in instance_specs:
        print("\n" + "=" * 60)
        print(f"Building {instance_name} instance")
        print("=" * 60)

        instance = build_instance(
            graph=graph,
            instance_name=instance_name,
            max_nodes=max_nodes,
            config=config,
        )

        instances.append(instance)

    print("\nStage Three graph processing completed.")

    return instances