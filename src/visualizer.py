from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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


def load_comparisons(config: dict[str, Any]) -> list[dict[str, Any]]:
    results_dir = Path(config["outputs"]["results_dir"])
    return load_json(results_dir / "comparison_results.json")


def get_location_coordinates(instance: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """
    Return mapping:
        loc_id -> (lon, lat)
    """
    coordinates = {}

    for location in instance["locations"]:
        coordinates[location["id"]] = (
            float(location["lon"]),
            float(location["lat"]),
        )

    return coordinates


def draw_base_graph(
    ax: Any,
    instance: dict[str, Any],
    coordinates: dict[str, tuple[float, float]],
) -> None:
    """
    Draw all road edges in the instance.
    """
    for edge in instance["edges"]:
        source = edge["from"]
        target = edge["to"]

        if source not in coordinates or target not in coordinates:
            continue

        x1, y1 = coordinates[source]
        x2, y2 = coordinates[target]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color="lightgray",
            linewidth=0.8,
            zorder=1,
        )

    xs = [coord[0] for coord in coordinates.values()]
    ys = [coord[1] for coord in coordinates.values()]

    ax.scatter(xs, ys, s=10, color="black", zorder=2)


def draw_route(
    ax: Any,
    route: list[str],
    coordinates: dict[str, tuple[float, float]],
    color: str,
    label: str,
    linewidth: float,
    linestyle: str = "-",
) -> None:
    """
    Draw one route over the base graph.
    """
    first_segment = True

    for source, target in zip(route[:-1], route[1:]):
        if source not in coordinates or target not in coordinates:
            continue

        x1, y1 = coordinates[source]
        x2, y2 = coordinates[target]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=3,
            label=label if first_segment else None,
        )

        first_segment = False


def plot_route_comparison(
    instance: dict[str, Any],
    comparison: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """
    Plot planner route and Dijkstra route on the same graph.
    """
    figures_dir = ensure_directory(config["outputs"]["figures_dir"])
    instance_name = instance["instance_name"]

    output_path = figures_dir / f"{instance_name}_route_comparison.png"

    coordinates = get_location_coordinates(instance)

    fig, ax = plt.subplots(figsize=(9, 7))

    draw_base_graph(ax, instance, coordinates)

    same_route = comparison.get("same_route_as_dijkstra", False)

    if same_route:
        draw_route(
            ax=ax,
            route=comparison["planner_route"],
            coordinates=coordinates,
            color="red",
            label="ENHSP = Dijkstra route",
            linewidth=3.0,
            linestyle="-",
        )
    else:
        draw_route(
            ax=ax,
            route=comparison["planner_route"],
            coordinates=coordinates,
            color="red",
            weight=7,
            opacity=0.75,
            label="ENHSP planner route",
            linewidth=2.5,
            linestyle="-",
        )

        draw_route(
            ax=ax,
            route=comparison["dijkstra_route"],
            coordinates=coordinates,
            color="blue",
            weight=4,
            opacity=0.95,
            dash_array="10,8",
            label="Dijkstra shortest route",
            linewidth=3.5,
            linestyle="--",
        )

    start = instance["start"]
    goal = instance["goal"]

    if start in coordinates:
        ax.scatter(
            coordinates[start][0],
            coordinates[start][1],
            s=80,
            color="green",
            marker="o",
            label="Start",
            zorder=4,
        )

    if goal in coordinates:
        ax.scatter(
            coordinates[goal][0],
            coordinates[goal][1],
            s=100,
            color="purple",
            marker="*",
            label="Goal",
            zorder=4,
        )

    ax.set_title(
        f"{instance_name.capitalize()} instance: planner vs Dijkstra route",
        fontsize=13,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Force normal coordinate labels, no scientific offset like +3.935e1
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    ax.legend()
    ax.grid(True, alpha=0.25)

    # This was missing in your function
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved route comparison figure: {output_path}")

    return output_path


def plot_runtime_chart(comparisons: list[dict[str, Any]], config: dict[str, Any]) -> Path:
    figures_dir = ensure_directory(config["outputs"]["figures_dir"])
    output_path = figures_dir / "runtime_chart.png"

    names = [item["instance_name"] for item in comparisons]
    runtimes = [float(item["planner_runtime_seconds"]) for item in comparisons]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, runtimes)
    ax.set_title("ENHSP Runtime by Instance Size")
    ax.set_xlabel("Instance")
    ax.set_ylabel("Runtime (seconds)")
    ax.grid(axis="y", alpha=0.3)

    for index, value in enumerate(runtimes):
        ax.text(index, value, f"{value:.2f}s", ha="center", va="bottom")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved runtime chart: {output_path}")

    return output_path


def plot_distance_comparison_chart(
    comparisons: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    figures_dir = ensure_directory(config["outputs"]["figures_dir"])
    output_path = figures_dir / "distance_comparison_chart.png"

    names = [item["instance_name"] for item in comparisons]
    planner_distances = [float(item["planner_distance_m"]) for item in comparisons]
    dijkstra_distances = [float(item["dijkstra_distance_m"]) for item in comparisons]

    x_positions = list(range(len(names)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        [x - width / 2 for x in x_positions],
        planner_distances,
        width,
        label="ENHSP planner",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        dijkstra_distances,
        width,
        label="Dijkstra",
    )

    ax.set_title("Route Distance: ENHSP Planner vs Dijkstra")
    ax.set_xlabel("Instance")
    ax.set_ylabel("Distance (meters)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved distance comparison chart: {output_path}")

    return output_path


def plot_battery_usage_chart(
    comparisons: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    figures_dir = ensure_directory(config["outputs"]["figures_dir"])
    output_path = figures_dir / "battery_usage_chart.png"

    names = [item["instance_name"] for item in comparisons]
    planner_battery = [float(item["planner_battery_used"]) for item in comparisons]
    dijkstra_battery = [float(item["dijkstra_battery_used"]) for item in comparisons]

    x_positions = list(range(len(names)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        [x - width / 2 for x in x_positions],
        planner_battery,
        width,
        label="ENHSP planner",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        dijkstra_battery,
        width,
        label="Dijkstra",
    )

    ax.set_title("Battery Consumption: ENHSP Planner vs Dijkstra")
    ax.set_xlabel("Instance")
    ax.set_ylabel("Battery units used")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved battery usage chart: {output_path}")

    return output_path


def plot_distance_gap_chart(
    comparisons: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    figures_dir = ensure_directory(config["outputs"]["figures_dir"])
    output_path = figures_dir / "distance_gap_chart.png"

    names = [item["instance_name"] for item in comparisons]
    gaps = [float(item["distance_gap_percent"]) for item in comparisons]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, gaps)
    ax.set_title("Distance Gap Between ENHSP Plan and Dijkstra Baseline")
    ax.set_xlabel("Instance")
    ax.set_ylabel("Gap (%)")
    ax.grid(axis="y", alpha=0.3)

    for index, value in enumerate(gaps):
        ax.text(index, value, f"{value:.2f}%", ha="center", va="bottom")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved distance gap chart: {output_path}")

    return output_path


def run_visualizations(config: dict[str, Any]) -> dict[str, Any]:
    comparisons = load_comparisons(config)

    route_figures = []

    for comparison in comparisons:
        instance_name = comparison["instance_name"]
        instance = load_instance(instance_name, config)

        route_figure = plot_route_comparison(
            instance=instance,
            comparison=comparison,
            config=config,
        )

        route_figures.append(str(route_figure))

    chart_paths = [
        str(plot_runtime_chart(comparisons, config)),
        str(plot_distance_comparison_chart(comparisons, config)),
        str(plot_battery_usage_chart(comparisons, config)),
        str(plot_distance_gap_chart(comparisons, config)),
    ]

    return {
        "route_figures": route_figures,
        "charts": chart_paths,
    }