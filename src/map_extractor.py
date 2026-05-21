from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox


def slugify_place_name(place_name: str) -> str:
    """
    Convert a place name into a safe filename.

    Example:
        "Rende, Calabria, Italy" -> "rende_calabria_italy"
    """
    name = place_name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def extract_graph_from_osm(config: dict[str, Any]) -> nx.MultiDiGraph:
    """
    Download a road network from OpenStreetMap using OSMnx.

    The returned graph is a NetworkX MultiDiGraph where:
    - nodes represent road intersections or important road points
    - edges represent road segments
    - edge attributes can include length, name, road type, speed, etc.
    """
    map_config = config["map"]

    place_name = map_config["place_name"]
    network_type = map_config.get("network_type", "drive")
    simplify = map_config.get("simplify", True)
    retain_all = map_config.get("retain_all", False)
    truncate_by_edge = map_config.get("truncate_by_edge", True)

    print(f"Downloading road network for: {place_name}")
    print(f"Network type: {network_type}")
    print(f"Simplify: {simplify}")

    ox.settings.use_cache = True
    ox.settings.log_console = True

    graph = ox.graph_from_place(
        place_name,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
        truncate_by_edge=truncate_by_edge,
    )

    print("Download complete.")
    print_graph_summary(graph)

    return graph


def print_graph_summary(graph: nx.MultiDiGraph) -> None:
    """Print basic graph information."""
    number_of_nodes = graph.number_of_nodes()
    number_of_edges = graph.number_of_edges()

    print("\nGraph summary")
    print("-------------")
    print(f"Nodes: {number_of_nodes}")
    print(f"Edges: {number_of_edges}")

    if number_of_edges > 0:
        lengths = [
            data.get("length", 0.0)
            for _, _, data in graph.edges(data=True)
            if data.get("length") is not None
        ]

        if lengths:
            print(f"Total edge length: {sum(lengths):.2f} meters")
            print(f"Average edge length: {sum(lengths) / len(lengths):.2f} meters")


def save_graph(graph: nx.MultiDiGraph, config: dict[str, Any]) -> Path:
    """
    Save the raw OSMnx graph to GraphML.

    GraphML is useful because we can load it again later without downloading
    from OpenStreetMap every time.
    """
    place_name = config["map"]["place_name"]
    safe_name = slugify_place_name(place_name)

    raw_data_dir = ensure_directory(config["outputs"]["raw_data_dir"])
    output_path = raw_data_dir / f"{safe_name}_raw.graphml"

    ox.io.save_graphml(graph, filepath=output_path)

    print(f"Saved raw graph to: {output_path}")

    return output_path


def save_graph_statistics(graph: nx.MultiDiGraph, config: dict[str, Any]) -> Path:
    """Save basic graph statistics to a JSON file."""
    place_name = config["map"]["place_name"]
    safe_name = slugify_place_name(place_name)

    raw_data_dir = ensure_directory(config["outputs"]["raw_data_dir"])
    output_path = raw_data_dir / f"{safe_name}_stats.json"

    edge_lengths = [
        float(data.get("length", 0.0))
        for _, _, data in graph.edges(data=True)
        if data.get("length") is not None
    ]

    stats = {
        "place_name": place_name,
        "network_type": config["map"].get("network_type", "drive"),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "total_edge_length_m": round(sum(edge_lengths), 3),
        "average_edge_length_m": round(
            sum(edge_lengths) / len(edge_lengths), 3
        )
        if edge_lengths
        else 0.0,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    print(f"Saved graph statistics to: {output_path}")

    return output_path


def plot_graph(graph: nx.MultiDiGraph, config: dict[str, Any]) -> Path:
    """
    Save a visualization of the extracted road network.
    """
    place_name = config["map"]["place_name"]
    safe_name = slugify_place_name(place_name)

    figures_dir = ensure_directory(config["outputs"]["figures_dir"])
    output_path = figures_dir / f"{safe_name}_raw_graph.png"

    fig, ax = ox.plot_graph(
        graph,
        show=False,
        close=False,
        node_size=5,
        edge_linewidth=0.6,
        bgcolor="white",
        node_color="black",
        edge_color="gray",
    )

    ax.set_title(f"Raw road network: {place_name}", fontsize=12)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved graph visualization to: {output_path}")

    return output_path


def run_map_extraction(config: dict[str, Any]) -> dict[str, str]:
    """
    Complete Stage Two:
    - download graph
    - save GraphML
    - save statistics
    - save visualization
    """
    graph = extract_graph_from_osm(config)

    graph_path = save_graph(graph, config)
    stats_path = save_graph_statistics(graph, config)
    figure_path = plot_graph(graph, config)

    return {
        "graph_path": str(graph_path),
        "stats_path": str(stats_path),
        "figure_path": str(figure_path),
    }