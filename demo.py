#!/usr/bin/env python3
"""
demo.py — Interactive command-line demo for the PDDL+ Urban EV Fleet Planner.

Runs the full pipeline:
  OSM download → PDDL+ generation → ENHSP → results printed to terminal

Two vehicles are planned simultaneously:
  Vehicle 1 — priority (bypasses red signals)
  Vehicle 2 — regular  (must wait at red signals)

All domain features are exercised:
  traffic signal, charging station, blocked edge, congested edge.

Usage:
  python demo.py
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import webbrowser

import yaml
import networkx as nx

from app_pipeline import (
    prepare_custom_map,
    run_multi_vehicle_planning_job,
    generate_custom_sumo_simulation,
    haversine_m,
    build_weighted_graph,
)
from sumo_traci_runner import run_sumo_traci_simulation

# ── helpers ───────────────────────────────────────────────────────────────────

def ask(prompt: str, default: str) -> str:
    """Print prompt with default and return user input, or default on Enter."""
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  Please enter a whole number.")


def ask_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number (e.g. 0.4).")


def pick_node(locations: list[dict], prompt: str, default_idx: int) -> str:
    """Ask the user to pick a node by index, return its id."""
    default_id = locations[default_idx]["id"]
    while True:
        raw = input(f"{prompt} [{default_idx} → {default_id}]: ").strip()
        if not raw:
            return default_id
        try:
            idx = int(raw)
            if 0 <= idx < len(locations):
                return locations[idx]["id"]
            print(f"  Index must be between 0 and {len(locations)-1}.")
        except ValueError:
            # accept a raw location id too
            ids = [l["id"] for l in locations]
            if raw in ids:
                return raw
            print("  Enter a number from the list above.")


def pick_edge(edges: list[dict], prompt: str, default_idx: int) -> dict:
    """Ask the user to pick an edge by index, return the edge dict."""
    default_edge = edges[default_idx]
    default_label = f"{default_edge['from']} → {default_edge['to']}"
    while True:
        raw = input(f"{prompt} [{default_idx} → {default_label}]: ").strip()
        if not raw:
            return default_edge
        try:
            idx = int(raw)
            if 0 <= idx < len(edges):
                return edges[idx]
            print(f"  Index must be between 0 and {len(edges)-1}.")
        except ValueError:
            print("  Enter a number from the list above.")


def farthest_reachable_node(from_id: str, locations: list[dict],
                            graph: nx.DiGraph, exclude: set[str]) -> int:
    """Return index of the reachable node farthest (Haversine) from from_id.

    Only considers nodes that have a directed path from from_id in the graph.
    Falls back to any reachable node if none are outside the exclude set.
    """
    reachable = nx.descendants(graph, from_id) if from_id in graph else set()
    src = next(l for l in locations if l["id"] == from_id)
    best_idx, best_d = -1, -1.0
    for i, loc in enumerate(locations):
        if loc["id"] in exclude or loc["id"] == from_id:
            continue
        if loc["id"] not in reachable:
            continue
        d = haversine_m(float(src["lat"]), float(src["lon"]),
                        float(loc["lat"]), float(loc["lon"]))
        if d > best_d:
            best_d, best_idx = d, i
    # fallback: any reachable node not excluded
    if best_idx == -1:
        for i, loc in enumerate(locations):
            if loc["id"] != from_id and loc["id"] not in exclude and loc["id"] in reachable:
                return i
    return best_idx if best_idx != -1 else 0


def visualise_folium(result: dict, base_instance: dict) -> None:
    """Render vehicle routes as animated moving markers on a Folium map."""
    import folium
    from folium.plugins import TimestampedGeoJson
    from datetime import datetime, timedelta

    loc_map  = {l["id"]: l for l in base_instance["locations"]}
    edge_dist = {(e["from"], e["to"]): float(e["distance_m"])
                 for e in base_instance["edges"]}

    lats = [float(l["lat"]) for l in base_instance["locations"]]
    lons = [float(l["lon"]) for l in base_instance["locations"]]
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]

    m = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    # Road network — thin gray lines
    for edge in base_instance["edges"]:
        src = loc_map[edge["from"]]
        dst = loc_map[edge["to"]]
        folium.PolyLine(
            [(float(src["lat"]), float(src["lon"])),
             (float(dst["lat"]), float(dst["lon"]))],
            color="#aaaaaa", weight=1.5, opacity=0.5,
        ).add_to(m)

    # Blocked edges — dark red
    for e in result["instance"].get("blocked_edges", []):
        if e["from"] in loc_map and e["to"] in loc_map:
            src, dst = loc_map[e["from"]], loc_map[e["to"]]
            folium.PolyLine(
                [(float(src["lat"]), float(src["lon"])),
                 (float(dst["lat"]), float(dst["lon"]))],
                color="#cc0000", weight=4, opacity=0.9,
                tooltip="Blocked",
            ).add_to(m)

    # Congested edges — orange
    for e in result["instance"].get("congested_edges", []):
        if e["from"] in loc_map and e["to"] in loc_map:
            src, dst = loc_map[e["from"]], loc_map[e["to"]]
            folium.PolyLine(
                [(float(src["lat"]), float(src["lon"])),
                 (float(dst["lat"]), float(dst["lon"]))],
                color="#ff8800", weight=4, opacity=0.9,
                tooltip=f"Congested (factor {e.get('congestion_factor', '?')})",
            ).add_to(m)

    # Signal nodes — blue circle markers
    for loc in base_instance["locations"]:
        if loc.get("has_traffic_signal"):
            folium.CircleMarker(
                location=(float(loc["lat"]), float(loc["lon"])),
                radius=6, color="blue", fill=True, fill_color="blue",
                fill_opacity=0.7, tooltip=f"{loc['id']} — signal",
            ).add_to(m)

    # Charging stations — green circle markers
    for loc in base_instance["locations"]:
        if loc.get("has_charging_station"):
            folium.CircleMarker(
                location=(float(loc["lat"]), float(loc["lon"])),
                radius=7, color="green", fill=True, fill_color="green",
                fill_opacity=0.8, tooltip=f"{loc['id']} — charging station",
            ).add_to(m)

    # Per-vehicle speed lookup
    vehicles_info = {v["id"]: v for v in result["instance"].get("vehicles", [])}
    default_speed = float(
        result["instance"].get("vehicle", {}).get("speed_m_per_s", 10.0)
    )

    palette     = ["#e60000", "#0055cc", "#009900", "#cc00cc", "#cc6600"]
    BASE_TIME   = datetime(2024, 1, 1, 8, 0, 0)
    features    = []
    STEP_S      = 1  # interpolation resolution in seconds

    for i, vr in enumerate(result.get("per_vehicle_results", [])):
        if vr.get("excluded"):
            continue
        route = vr.get("route", [])
        if len(route) < 2:
            continue

        color = palette[i % len(palette)]
        speed = float(vehicles_info.get(vr["vehicle_id"], {})
                      .get("speed_m_per_s", default_speed))

        coords = []   # [lon, lat] per time step
        times  = []   # ISO timestamp per time step
        elapsed = 0.0

        for j in range(len(route) - 1):
            f_id, t_id = route[j], route[j + 1]
            f_loc, t_loc = loc_map[f_id], loc_map[t_id]
            dist = edge_dist.get((f_id, t_id), 100.0)
            edge_time = dist / speed          # seconds to traverse this edge

            steps = max(2, int(edge_time / STEP_S))
            for s in range(steps):
                frac = s / steps
                lat = float(f_loc["lat"]) + frac * (float(t_loc["lat"]) - float(f_loc["lat"]))
                lon = float(f_loc["lon"]) + frac * (float(t_loc["lon"]) - float(f_loc["lon"]))
                coords.append([lon, lat])
                ts = BASE_TIME + timedelta(seconds=elapsed + frac * edge_time)
                times.append(ts.strftime("%Y-%m-%dT%H:%M:%S"))

            elapsed += edge_time

        # Final position — goal node
        goal = loc_map[route[-1]]
        coords.append([float(goal["lon"]), float(goal["lat"])])
        times.append((BASE_TIME + timedelta(seconds=elapsed)).strftime("%Y-%m-%dT%H:%M:%S"))

        label = vr["vehicle_id"] + (" ★priority" if i == 0 else "")
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "times": times,
                "style": {"color": color, "weight": 3, "opacity": 0.5},
                "icon": "circle",
                "iconstyle": {
                    "fillColor": color, "fillOpacity": 0.95,
                    "stroke": True, "color": "white",
                    "weight": 2, "radius": 9,
                },
                "popup": label,
            },
        })

        # Static start / goal markers
        start_loc = loc_map[route[0]]
        folium.Marker(
            (float(start_loc["lat"]), float(start_loc["lon"])),
            popup=f"{vr['vehicle_id']} START",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(m)
        folium.Marker(
            (float(goal["lat"]), float(goal["lon"])),
            popup=f"{vr['vehicle_id']} GOAL",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(m)

    if features:
        TimestampedGeoJson(
            {"type": "FeatureCollection", "features": features},
            period="PT1S",
            add_last_point=True,
            auto_play=True,
            loop=True,
            max_speed=10,
            loop_button=True,
            time_slider_drag_update=True,
        ).add_to(m)

    # Legend
    legend_rows = (
        "<span style='color:#cc0000'>&#9644;</span> Blocked edge<br>"
        "<span style='color:#ff8800'>&#9644;</span> Congested edge<br>"
        "<span style='color:blue'>&#11044;</span> Traffic signal<br>"
        "<span style='color:green'>&#11044;</span> Charging station<br>"
    )
    for i, vr in enumerate(result.get("per_vehicle_results", [])):
        if not vr.get("excluded"):
            color = palette[i % len(palette)]
            label = vr["vehicle_id"] + (" (priority)" if i == 0 else " (regular)")
            legend_rows += (
                f"<span style='color:{color}'>&#9644;</span> {label}<br>"
            )
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px 14px;border-radius:6px;border:1px solid #ccc;font-size:12px;
                line-height:1.8;">
      <b>Legend</b><br>{legend_rows}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    out = PROJECT_ROOT / "demo_output" / "animation.html"
    out.parent.mkdir(exist_ok=True)
    m.save(str(out))
    print(f"  Map saved: {out}")
    webbrowser.open(out.resolve().as_uri())


def reachable_start_idx(
    slot: int,
    locations: list[dict],
    graph: nx.DiGraph,
    exclude: set[str],
) -> int:
    """
    Return the index of the slot-th node (0-based) that has at least one
    outgoing edge in the graph and is not in exclude.
    Falls back to the first such node if slot exceeds the available count.
    """
    candidates = [
        i for i, loc in enumerate(locations)
        if loc["id"] not in exclude and graph.out_degree(loc["id"]) > 0
    ]
    if not candidates:
        # Last resort: any node not excluded
        candidates = [i for i, loc in enumerate(locations) if loc["id"] not in exclude]
    if not candidates:
        return 0
    return candidates[slot % len(candidates)]


def non_bridge_edge_idx(edges: list[dict], graph: nx.DiGraph, exclude_idx: int = -1) -> int:
    """Return index of an edge whose removal does not disconnect the graph."""
    undirected = graph.to_undirected()
    bridges = set(nx.bridges(undirected))
    for i, e in enumerate(edges):
        if i == exclude_idx:
            continue
        pair = (e["from"], e["to"])
        pair_rev = (e["to"], e["from"])
        if pair not in bridges and pair_rev not in bridges:
            return i
    # fallback: return any edge that isn't the excluded one
    return 0 if exclude_idx != 0 else 1


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        print(f"ERROR: config.yaml not found at {config_path}")
        sys.exit(1)
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print()
    print("=" * 60)
    print("  PDDL+ Urban EV Fleet Planner — Interactive Demo")
    print("=" * 60)
    print()

    # ── Step 1: Map ───────────────────────────────────────────────────────────
    print("─── Step 1: Map ─────────────────────────────────────────────")
    place  = ask("Place name", "Rende, Cosenza, Italy")
    max_nodes = ask_int("Max nodes to extract", 100)
    print()
    print(f"  Downloading OSM graph for '{place}' (max {max_nodes} nodes)...")

    try:
        base_instance = prepare_custom_map(
            place_name=place,
            network_type="drive",
            max_nodes=max_nodes,
            vehicle_speed_m_per_s=float(config["vehicle"].get("speed_m_per_s", 10.0)),
            config=config,
            min_chargers=1,
        )
    except Exception as exc:
        print(f"\nERROR downloading map: {exc}")
        sys.exit(1)

    locations = base_instance["locations"]
    edges     = base_instance["edges"]
    print(f"  Graph ready: {len(locations)} nodes, {len(edges)} edges.")
    print()

    # ── Show nodes ────────────────────────────────────────────────────────────
    print("  Nodes:")
    for i, loc in enumerate(locations):
        tags = []
        if loc.get("has_traffic_signal"):
            tags.append("traffic_signal")
        if loc.get("has_charging_station"):
            tags.append("charging_station")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        print(f"    [{i:2d}] {loc['id']}  ({float(loc['lat']):.4f}, {float(loc['lon']):.4f}){tag_str}")
    print()

    # ── Step 2: Signal nodes ──────────────────────────────────────────────────
    print("─── Step 2: Signal nodes ────────────────────────────────────")
    signal_nodes = [l["id"] for l in locations if l.get("has_traffic_signal")]
    if signal_nodes:
        print(f"  Detected from OSM: {', '.join(signal_nodes)}")
        if input("  Add another signal node? (y/N): ").strip().lower() == "y":
            extra = pick_node(locations, "  Node index to mark as signal",
                              len(locations) // 2)
            for loc in base_instance["locations"]:
                if loc["id"] == extra:
                    loc["has_traffic_signal"] = True
            print(f"  Added signal node: {extra}")
    else:
        print("  No signal nodes detected in this area.")
        if input("  Designate one manually? (y/N): ").strip().lower() == "y":
            chosen = pick_node(locations, "  Node index to mark as signal",
                               len(locations) // 2)
            for loc in base_instance["locations"]:
                if loc["id"] == chosen:
                    loc["has_traffic_signal"] = True
            print(f"  Signal node set: {chosen}")
        else:
            print("  Continuing without signal nodes.")
    print()

    # ── Step 3: Charging stations ─────────────────────────────────────────────
    print("─── Step 3: Charging stations ───────────────────────────────")
    charger_nodes = [l["id"] for l in locations if l.get("has_charging_station")]
    if charger_nodes:
        print(f"  Detected / placed: {', '.join(charger_nodes)}")
        if input("  Add another charging station? (y/N): ").strip().lower() == "y":
            extra = pick_node(locations, "  Node index to mark as charging station",
                              len(locations) // 3)
            for loc in base_instance["locations"]:
                if loc["id"] == extra:
                    loc["has_charging_station"] = True
            print(f"  Added charging station: {extra}")
    else:
        print("  No charging stations detected. Adding one manually.")
        chosen = pick_node(locations, "  Node index to mark as charging station",
                           len(locations) // 3)
        for loc in base_instance["locations"]:
            if loc["id"] == chosen:
                loc["has_charging_station"] = True
        print(f"  Charging station set: {chosen}")
    print()

    # ── Show edges ────────────────────────────────────────────────────────────
    print("  Edges:")
    for i, e in enumerate(edges):
        print(f"    [{i:2d}] {e['from']} → {e['to']}  ({float(e['distance_m']):.0f} m)")
    print()

    # build graph for bridge detection
    g = build_weighted_graph(base_instance, metric="distance", ignore_blocked=False)

    # ── Step 4: Blocked edge ──────────────────────────────────────────────────
    print("─── Step 4: Blocked edge ────────────────────────────────────")
    blocked_default = non_bridge_edge_idx(edges, g)
    blocked_edge = pick_edge(edges, "Blocked edge index", blocked_default)
    print(f"  Blocked: {blocked_edge['from']} → {blocked_edge['to']}")
    print()

    # ── Step 5: Congested edge ────────────────────────────────────────────────
    print("─── Step 5: Congested edge ──────────────────────────────────")
    blocked_idx = next(
        i for i, e in enumerate(edges)
        if e["from"] == blocked_edge["from"] and e["to"] == blocked_edge["to"]
    )
    congested_default = non_bridge_edge_idx(edges, g, exclude_idx=blocked_idx)
    congested_edge = pick_edge(edges, "Congested edge index", congested_default)
    congestion_factor = ask_float("Congestion factor (0.1=severe … 0.9=mild)", 0.4)
    print(f"  Congested: {congested_edge['from']} → {congested_edge['to']}  (factor {congestion_factor})")
    print()

    # ── Step 6: Vehicles ──────────────────────────────────────────────────────
    num_vehicles = ask_int("─── Step 6: How many vehicles? (1–5)", 2)
    num_vehicles = max(1, min(5, num_vehicles))
    print()

    battery  = float(config["vehicle"].get("initial_battery", 100.0))
    speed    = float(config["vehicle"].get("speed_m_per_s",   10.0))
    cons     = float(config["vehicle"].get("battery_consumption_per_meter", 0.01))
    ch_rate  = float(config["vehicle"].get("charge_rate", 5.0))

    vehicles      = []
    used_nodes: set[str] = set()

    for vi in range(num_vehicles):
        vid      = f"vehicle_{vi + 1}"
        priority = vi == 0          # only first vehicle is priority
        label    = "priority" if priority else "regular"
        step_num = 6 + vi

        print(f"─── Step {step_num}: {vid} ({label}) ────────────────────────────")

        start_default_idx = reachable_start_idx(vi, locations, g, used_nodes)
        goal_default_idx  = farthest_reachable_node(
            locations[start_default_idx]["id"], locations, g,
            exclude=used_nodes | {locations[start_default_idx]["id"]},
        )

        v_start = pick_node(locations, "  Start node index", start_default_idx)
        v_goal  = pick_node(locations, "  Goal  node index", goal_default_idx)
        print(f"  {vid}: {v_start} → {v_goal}  ({label})")
        print()

        used_nodes.update({v_start, v_goal})
        vehicles.append({
            "id":    vid,
            "start": v_start,
            "goal":  v_goal,
            "battery":     battery,
            "max_battery": battery,
            "speed_m_per_s": speed,
            "battery_consumption_per_meter": cons,
            "charge_rate": ch_rate,
            "priority": priority,
        })

    # ── Run pipeline ──────────────────────────────────────────────────────────
    pipeline_step = 6 + num_vehicles
    print(f"─── Step {pipeline_step}: Running pipeline ────────────────────────────")

    if len(locations) > 25 or num_vehicles > 2:
        strategy = config.get("planner", {}).get("search_strategy", "") or "WAStar"
        timeout  = config.get("planner", {}).get("timeout_seconds", 120)
        print(f"  NOTE: {len(locations)} nodes × {num_vehicles} vehicles — using"
              f" {strategy} search, timeout {timeout} s."
              f" Set 'search_strategy: gbfs' in config.yaml for fastest results.")

    blocked_objs = [{
        "from": blocked_edge["from"],
        "to":   blocked_edge["to"],
        "distance_m":    blocked_edge["distance_m"],
        "travel_time_s": blocked_edge.get("travel_time_s", 0),
        "name":    blocked_edge.get("name", ""),
        "highway": blocked_edge.get("highway", ""),
    }]

    congested_objs = [{
        "from": congested_edge["from"],
        "to":   congested_edge["to"],
        "distance_m":    congested_edge["distance_m"],
        "travel_time_s": congested_edge.get("travel_time_s", 0),
        "name":    congested_edge.get("name", ""),
        "highway": congested_edge.get("highway", ""),
        "congestion_factor": congestion_factor,
        "reason": "Selected in demo",
    }]

    sig_red   = int(config.get("signal", {}).get("red_duration",    30))
    sig_green = int(config.get("signal", {}).get("green_duration",  45))

    print("  Generating PDDL+ and invoking ENHSP...")
    try:
        result = run_multi_vehicle_planning_job(
            base_instance=base_instance,
            vehicles=vehicles,
            metric="time",
            blocked_edges=blocked_objs,
            congested_edges=congested_objs,
            config=config,
            signal_red_duration=sig_red,
            signal_green_duration=sig_green,
        )
    except Exception as exc:
        print(f"\nERROR during planning: {exc}")
        sys.exit(1)

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)

    if not result.get("success"):
        print("  No plan found.")
        for vr in result.get("per_vehicle_results", []):
            if vr.get("excluded"):
                print(f"  Vehicle {vr['vehicle_id']} excluded: "
                      f"{vr['diagnosis'].get('reason', '')}")
        return

    print(f"  Plan found in {result.get('total_runtime_seconds', 0):.1f} s")
    print(f"  Vehicles planned: {result.get('num_planned', 0)} / "
          f"{result.get('num_vehicles', 0)}")
    print()

    for vr in result.get("per_vehicle_results", []):
        vid = vr.get("vehicle_id", "?")
        if vr.get("excluded"):
            print(f"  [{vid}] EXCLUDED — {vr['diagnosis'].get('reason', '')}")
            continue
        route = vr.get("route", [])
        dist  = vr.get("total_distance_m", 0)
        batt  = vr.get("battery_remaining", None)
        print(f"  [{vid}]  Route: {' → '.join(route)}")
        print(f"           Distance: {dist:.0f} m"
              + (f"  |  Battery remaining: {batt:.1f}" if batt is not None else ""))

    print()
    domain_path  = result.get("domain_path",  "")
    problem_path = result.get("problem_path", "")
    plan_path    = result.get("planner_result", {}).get("plan_file", "")
    if domain_path:
        print(f"  Domain:  {domain_path}")
    if problem_path:
        print(f"  Problem: {problem_path}")
    if plan_path:
        print(f"  Plan:    {plan_path}")
        print()
        plan_text = Path(plan_path).read_text(encoding="utf-8") if Path(plan_path).exists() else ""
        if plan_text.strip():
            print("  ── Plan actions ──")
            for line in plan_text.strip().splitlines():
                print(f"  {line}")

    # ── Visualisation ─────────────────────────────────────────────────────────
    print()
    print("─── Visualisation ───────────────────────────────────────────")
    print("  [1] Folium   — animated moving vehicles (browser)")
    print("  [2] SUMO-GUI — full traffic simulation with background vehicles")
    print("  [3] Both     — Folium first, then SUMO-GUI")
    print("  [4] Skip")
    vis_choice = input("  Choose [1]: ").strip() or "1"

    if vis_choice in ("1", "3"):
        print("  Building Folium animation...")
        visualise_folium(result, base_instance)

    if vis_choice in ("2", "3"):
        print("  Starting traci-controlled SUMO-GUI simulation...")
        print("  Close the SUMO-GUI window when done.")
        try:
            run_sumo_traci_simulation(
                result=result,
                instance=result["instance"],
                config=config,
                gui=True,
            )
        except FileNotFoundError:
            print("  sumo-gui not found on PATH. Check your SUMO installation.")
        except Exception as exc:
            print(f"  SUMO error: {exc}")

    if vis_choice == "4":
        print("  Skipping visualisation.")

    print()
    print("  Demo complete.")
    print()


if __name__ == "__main__":
    main()
