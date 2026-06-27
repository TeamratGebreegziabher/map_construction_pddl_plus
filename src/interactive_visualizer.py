from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import folium


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


def get_coordinates(instance: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """
    Return:
        loc_id -> (lat, lon)
    """
    coordinates = {}

    for location in instance["locations"]:
        coordinates[location["id"]] = (
            float(location["lat"]),
            float(location["lon"]),
        )

    return coordinates


def get_map_center(coordinates: dict[str, tuple[float, float]]) -> tuple[float, float]:
    latitudes = [coord[0] for coord in coordinates.values()]
    longitudes = [coord[1] for coord in coordinates.values()]

    center_lat = sum(latitudes) / len(latitudes)
    center_lon = sum(longitudes) / len(longitudes)

    return center_lat, center_lon


def route_to_coordinates(
    route: list[str],
    coordinates: dict[str, tuple[float, float]],
) -> list[tuple[float, float]]:
    route_coordinates = []

    for location_id in route:
        if location_id in coordinates:
            route_coordinates.append(coordinates[location_id])

    return route_coordinates


def build_edge_lookup(instance: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    edge_lookup = {}

    for edge in instance["edges"]:
        edge_lookup[(edge["from"], edge["to"])] = edge

    return edge_lookup


def build_route_popup_text(
    route_name: str,
    route: list[str],
    instance: dict[str, Any],
) -> str:
    edge_lookup = build_edge_lookup(instance)

    total_distance = 0.0
    total_time = 0.0

    rows = []

    for index, (source, target) in enumerate(zip(route[:-1], route[1:]), start=1):
        edge = edge_lookup.get((source, target))

        if edge is None:
            rows.append(
                f"<tr><td>{index}</td><td>{source}</td><td>{target}</td>"
                f"<td colspan='3'>missing edge</td></tr>"
            )
            continue

        distance = float(edge["distance_m"])
        travel_time = float(edge["travel_time_s"])
        road_name = edge.get("name", "")

        total_distance += distance
        total_time += travel_time

        rows.append(
            f"<tr>"
            f"<td>{index}</td>"
            f"<td>{source}</td>"
            f"<td>{target}</td>"
            f"<td>{distance:.2f}</td>"
            f"<td>{travel_time:.2f}</td>"
            f"<td>{road_name}</td>"
            f"</tr>"
        )

    table_rows = "\n".join(rows)

    html = f"""
    <div style="width: 520px;">
      <h4>{route_name}</h4>
      <p>
        <b>Moves:</b> {max(len(route) - 1, 0)}<br>
        <b>Total distance:</b> {total_distance:.2f} m<br>
        <b>Total travel time:</b> {total_time:.2f} s
      </p>
      <table border="1" style="border-collapse: collapse; font-size: 11px;">
        <tr>
          <th>#</th>
          <th>From</th>
          <th>To</th>
          <th>Distance m</th>
          <th>Time s</th>
          <th>Road</th>
        </tr>
        {table_rows}
      </table>
    </div>
    """

    return html


def add_base_graph(
    fmap: folium.Map,
    instance: dict[str, Any],
    coordinates: dict[str, tuple[float, float]],
) -> None:
    """
    Draw all available directed road edges in light gray.
    """
    base_group = folium.FeatureGroup(name="Road graph", show=True)

    for edge in instance["edges"]:
        source = edge["from"]
        target = edge["to"]

        if source not in coordinates or target not in coordinates:
            continue

        folium.PolyLine(
            locations=[coordinates[source], coordinates[target]],
            color="gray",
            weight=2,
            opacity=0.35,
            tooltip=f"{source} -> {target}",
        ).add_to(base_group)

    base_group.add_to(fmap)


def add_route(
    fmap: folium.Map,
    route: list[str],
    coordinates: dict[str, tuple[float, float]],
    instance: dict[str, Any],
    route_name: str,
    color: str,
    dashed: bool = False,
    show: bool = True,
) -> None:
    route_coordinates = route_to_coordinates(route, coordinates)

    if len(route_coordinates) < 2:
        return

    route_group = folium.FeatureGroup(name=route_name, show=show)

    popup_html = build_route_popup_text(
        route_name=route_name,
        route=route,
        instance=instance,
    )

    folium.PolyLine(
        locations=route_coordinates,
        color=color,
        weight=6,
        opacity=0.9,
        dash_array="8, 8" if dashed else None,
        tooltip=route_name,
        popup=folium.Popup(popup_html, max_width=600),
    ).add_to(route_group)

    route_group.add_to(fmap)


def add_start_goal_markers(
    fmap: folium.Map,
    instance: dict[str, Any],
    coordinates: dict[str, tuple[float, float]],
) -> None:
    start = instance["start"]
    goal = instance["goal"]

    if start in coordinates:
        folium.Marker(
            location=coordinates[start],
            tooltip=f"Start: {start}",
            popup=f"<b>Start</b><br>{start}",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(fmap)

    if goal in coordinates:
        folium.Marker(
            location=coordinates[goal],
            tooltip=f"Goal: {goal}",
            popup=f"<b>Goal</b><br>{goal}",
            icon=folium.Icon(color="purple", icon="flag"),
        ).add_to(fmap)


def add_summary_panel(
    fmap: folium.Map,
    comparison: dict[str, Any],
) -> None:
    """
    Add a fixed-position HTML summary panel to the map.
    """
    html = f"""
    <div style="
        position: fixed;
        bottom: 35px;
        left: 35px;
        z-index: 9999;
        background-color: white;
        padding: 12px;
        border: 2px solid gray;
        border-radius: 8px;
        font-size: 13px;
        max-width: 360px;
    ">
      <h4 style="margin-top:0;">Route Comparison: {comparison['instance_name']}</h4>
      <b>Planner status:</b> {comparison['planner_status']}<br>
      <b>Planner valid:</b> {comparison['planner_valid']}<br>
      <b>Planner distance:</b> {comparison['planner_distance_m']} m<br>
      <b>Dijkstra distance:</b> {comparison['dijkstra_distance_m']} m<br>
      <b>Distance gap:</b> {comparison['distance_gap_percent']}%<br>
      <b>Planner battery used:</b> {comparison['planner_battery_used']}<br>
      <b>Final battery:</b> {comparison['planner_final_battery']}<br>
    </div>
    """

    fmap.get_root().html.add_child(folium.Element(html))


def create_interactive_route_map(
    instance: dict[str, Any],
    comparison: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    maps_dir = ensure_directory(config["outputs"].get("maps_dir", "outputs/maps"))
    instance_name = instance["instance_name"]

    output_path = maps_dir / f"{instance_name}_interactive_route_map.html"

    coordinates = get_coordinates(instance)
    center = get_map_center(coordinates)

    fmap = folium.Map(
        location=center,
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    add_base_graph(fmap, instance, coordinates)

    same_route = comparison.get("same_route_as_dijkstra", False)

    if same_route:
        add_route(
            fmap=fmap,
            route=comparison["planner_route"],
            coordinates=coordinates,
            instance=instance,
            route_name="ENHSP = Dijkstra route",
            color="red",
            dashed=False,
            show=True,
        )
    else:
        add_route(
            fmap=fmap,
            route=comparison["planner_route"],
            coordinates=coordinates,
            instance=instance,
            route_name="ENHSP planner route",
            color="red",
            dashed=False,
            show=True,
        )

        add_route(
            fmap=fmap,
            route=comparison["dijkstra_route"],
            coordinates=coordinates,
            instance=instance,
            route_name="Dijkstra shortest route",
            color="blue",
            dashed=True,
            show=True,
        )

    add_start_goal_markers(fmap, instance, coordinates)
    add_summary_panel(fmap, comparison)

    folium.LayerControl(collapsed=False).add_to(fmap)

    fmap.save(str(output_path))

    print(f"Saved interactive route map: {output_path}")

    return output_path


def run_interactive_visualizations(config: dict[str, Any]) -> dict[str, Any]:
    comparisons = load_comparisons(config)

    generated_maps = []

    for comparison in comparisons:
        instance_name = comparison["instance_name"]
        instance = load_instance(instance_name, config)

        map_path = create_interactive_route_map(
            instance=instance,
            comparison=comparison,
            config=config,
        )

        generated_maps.append(str(map_path))

    return {
        "interactive_maps": generated_maps,
    }

# ---------------------------------------------------------------------------
# Multi-vehicle interactive map
# ---------------------------------------------------------------------------

# Colours matching the UI palette — indexed by vehicle position
_MV_COLOURS = ["red", "green", "blue", "orange", "purple"]
_MV_GOAL_COLOURS = ["darkred", "darkgreen", "darkblue", "cadetblue", "darkpurple"]


def create_multi_vehicle_route_map(
    instance: dict[str, Any],
    per_vehicle_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    """
    Generate an interactive Folium map showing all vehicle routes.

    Each vehicle gets its own colour (matching the UI palette).
    Vehicles with no route are skipped.
    Start and goal markers are shown per vehicle.
    """
    maps_dir = ensure_directory(config["outputs"].get("maps_dir", "outputs/maps"))
    instance_name = instance["instance_name"]
    output_path = maps_dir / f"{instance_name}_interactive_route_map.html"

    coordinates = get_coordinates(instance)
    center = get_map_center(coordinates)

    fmap = folium.Map(
        location=center,
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    add_base_graph(fmap, instance, coordinates)

    for idx, vr in enumerate(per_vehicle_results):
        vid = vr["vehicle_id"]
        route = vr.get("route", [])
        colour = _MV_COLOURS[idx % len(_MV_COLOURS)]
        goal_colour = _MV_GOAL_COLOURS[idx % len(_MV_GOAL_COLOURS)]

        if len(route) >= 2:
            add_route(
                fmap=fmap,
                route=route,
                coordinates=coordinates,
                instance=instance,
                route_name=f"{vid} (ENHSP)",
                color=colour,
                dashed=False,
                show=True,
            )

            # Dijkstra comparison route
            dijkstra_route = vr.get("dijkstra_route", [])
            if dijkstra_route and dijkstra_route != route:
                add_route(
                    fmap=fmap,
                    route=dijkstra_route,
                    coordinates=coordinates,
                    instance=instance,
                    route_name=f"{vid} (Dijkstra)",
                    color=goal_colour,
                    dashed=True,
                    show=False,
                )

        # Start marker
        v_start = vr.get("start")
        if v_start and v_start in coordinates:
            lat, lon = coordinates[v_start]
            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                color=colour,
                fill=True,
                fill_opacity=0.9,
                tooltip=f"{vid} START: {v_start}",
                popup=f"<b>{vid} Start</b><br>{v_start}",
            ).add_to(fmap)

        # Goal marker
        v_goal = vr.get("goal")
        if v_goal and v_goal in coordinates:
            lat, lon = coordinates[v_goal]
            folium.Marker(
                location=[lat, lon],
                tooltip=f"{vid} GOAL: {v_goal}",
                popup=f"<b>{vid} Goal</b><br>{v_goal}",
                icon=folium.Icon(color=colour, icon="flag"),
            ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    fmap.save(str(output_path))
    print(f"Saved multi-vehicle interactive route map: {output_path}")

    return output_path