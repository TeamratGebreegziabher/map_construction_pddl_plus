from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import subprocess
import folium
import osmnx as ox
import streamlit as st
import streamlit.components.v1 as components
import yaml
from folium.plugins import Draw
from streamlit_folium import st_folium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from app_pipeline import (
    apply_user_choices_to_instance,
    build_blocked_edge_objects,
    build_congested_edge_objects,
    clear_custom_outputs,
    extract_location_id,
    find_nearest_location_id,
    list_edge_options,
    list_location_options,
    prepare_custom_map,
    prepare_custom_map_from_polygon,
    run_custom_planning_job,
    run_multi_vehicle_planning_job,
    generate_custom_sumo_simulation,
)

# ---------------------------------------------------------------------------
# Fixed colour palette — indexed by vehicle position (0-based)
# Defined once here, used consistently across map, UI labels, and SUMO.
# ---------------------------------------------------------------------------
_PALETTE = [
    {"folium_start": "red",       "folium_goal": "darkred",    "icon": "🔴", "sumo": "255,0,0"},
    {"folium_start": "green",     "folium_goal": "darkgreen",  "icon": "🟢", "sumo": "0,200,0"},
    {"folium_start": "blue",      "folium_goal": "darkblue",   "icon": "🔵", "sumo": "0,100,255"},
    {"folium_start": "orange",    "folium_goal": "cadetblue",  "icon": "🟠", "sumo": "255,165,0"},
    {"folium_start": "purple",    "folium_goal": "darkpurple", "icon": "🟣", "sumo": "160,32,240"},
]

def vehicle_icon(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]["icon"]

def vehicle_folium_start(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]["folium_start"]

def vehicle_folium_goal(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]["folium_goal"]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        st.error(f"config.yaml not found: {config_path}")
        st.stop()
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def render_html_file(path: str | Path, height: int = 650) -> None:
    p = Path(path)
    if not p.exists():
        st.warning(f"Map file not found: {p}")
        return
    components.html(p.read_text(encoding="utf-8", errors="replace"),
                    height=height, scrolling=True)


@st.cache_data(show_spinner=False)
def geocode_place_center(place_name: str) -> tuple[float, float]:
    try:
        lat, lon = ox.geocode(place_name)
        return float(lat), float(lon)
    except Exception:
        return 39.3500, 16.2250


def get_last_drawn_geometry(draw_data: dict | None) -> dict | None:
    if not draw_data:
        return None
    drawings = draw_data.get("all_drawings")
    if not drawings:
        return None
    last = drawings[-1]
    if not last:
        return None
    geom = last.get("geometry")
    if not geom or geom.get("type") not in ["Polygon", "MultiPolygon"]:
        return None
    return geom


def build_area_selection_map(center_lat: float, center_lon: float) -> folium.Map:
    fmap = folium.Map(
        location=[center_lat, center_lon], zoom_start=14,
        tiles="OpenStreetMap", control_scale=True,
    )
    Draw(
        export=False,
        draw_options={
            "polyline": False, "circle": False, "circlemarker": False,
            "marker": False, "polygon": True, "rectangle": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)
    return fmap


def build_node_selection_map(
    instance: dict[str, Any],
    n_vehicles: int,
) -> folium.Map:
    """
    Node selection map.

    Node colours:
      🟡 Yellow (radius 10) — last clicked node (pending assignment)
      Per vehicle: filled circle = start, outlined = goal
        car1: red start, darkred goal
        car2: green start, darkgreen goal
        car3: blue start, darkblue goal
        car4: orange start, cadetblue goal
        car5: purple start, darkpurple goal
      🟠 Orange — traffic signal
      Light green — charging station
      Blue (small) — unassigned node
    """
    lats = [float(loc["lat"]) for loc in instance["locations"]]
    lons = [float(loc["lon"]) for loc in instance["locations"]]
    fmap = folium.Map(
        location=[sum(lats)/len(lats), sum(lons)/len(lons)],
        zoom_start=15, tiles="OpenStreetMap", control_scale=True,
    )

    coords = {
        loc["id"]: (float(loc["lat"]), float(loc["lon"]))
        for loc in instance["locations"]
    }

    # Draw road edges
    for edge in instance["edges"]:
        s, t = edge["from"], edge["to"]
        if s not in coords or t not in coords:
            continue
        folium.PolyLine(
            locations=[coords[s], coords[t]],
            color="gray", weight=2, opacity=0.4,
            tooltip=f"{s} → {t}",
        ).add_to(fmap)

    # Collect assigned locations
    last_clicked = st.session_state.get("last_clicked_node")
    assigned_starts: dict[str, int] = {}  # loc_id -> vehicle index
    assigned_goals: dict[str, int] = {}

    if n_vehicles == 1:
        s = st.session_state.get("selected_start")
        g = st.session_state.get("selected_goal")
        if s:
            assigned_starts[s] = 0
        if g:
            assigned_goals[g] = 0
    else:
        for i in range(1, n_vehicles + 1):
            s = st.session_state.get(f"mv_start_{i}")
            g = st.session_state.get(f"mv_goal_{i}")
            if s:
                assigned_starts[s] = i - 1
            if g:
                assigned_goals[g] = i - 1

    for loc in instance["locations"]:
        lid = loc["id"]
        lat, lon = float(loc["lat"]), float(loc["lon"])
        tip = f"{lid} ({lat:.5f}, {lon:.5f})"

        if loc.get("has_traffic_signal"):
            tip += " 🚦"
        if loc.get("has_charging_station"):
            tip += " ⚡"

        # Last clicked — yellow, prominent
        if lid == last_clicked:
            folium.CircleMarker(
                location=[lat, lon], radius=10,
                color="gold", fill=True, fill_color="yellow",
                fill_opacity=0.95, weight=3,
                tooltip=tip + " ← clicked",
            ).add_to(fmap)

        elif lid in assigned_starts:
            idx = assigned_starts[lid]
            folium.CircleMarker(
                location=[lat, lon], radius=9,
                color=vehicle_folium_start(idx),
                fill=True, fill_opacity=0.9,
                tooltip=tip + f" [car{idx+1} START]",
            ).add_to(fmap)

        elif lid in assigned_goals:
            idx = assigned_goals[lid]
            # Outlined (fill_opacity low) to distinguish goal from start
            folium.CircleMarker(
                location=[lat, lon], radius=9,
                color=vehicle_folium_goal(idx),
                fill=True, fill_opacity=0.3, weight=3,
                tooltip=tip + f" [car{idx+1} GOAL]",
            ).add_to(fmap)

        elif loc.get("has_traffic_signal"):
            folium.CircleMarker(
                location=[lat, lon], radius=5,
                color="orange", fill=True, fill_opacity=0.8,
                tooltip=tip,
            ).add_to(fmap)

        elif loc.get("has_charging_station"):
            folium.CircleMarker(
                location=[lat, lon], radius=5,
                color="lightgreen", fill=True, fill_opacity=0.8,
                tooltip=tip,
            ).add_to(fmap)

        else:
            folium.CircleMarker(
                location=[lat, lon], radius=4,
                color="blue", fill=True, fill_opacity=0.6,
                tooltip=tip,
            ).add_to(fmap)

    return fmap


# ---------------------------------------------------------------------------
# Result display helpers
# ---------------------------------------------------------------------------

def show_planner_output(result: dict[str, Any]) -> None:
    if result.get("is_multi_vehicle"):
        _show_multi_output(result)
    else:
        _show_single_output(result)


def _show_single_output(result: dict[str, Any]) -> None:
    validation = result["validation"]
    planner_result = result["planner_result"]
    comparison = result["comparison"]

    if validation["valid"]:
        st.success("Planner produced a valid plan.")
    else:
        st.error("Planner did not produce a valid plan.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", planner_result.get("status"))
    c2.metric("Runtime", f"{planner_result.get('runtime_seconds')} s")
    c3.metric("Plan length", planner_result.get("plan_length"))
    c4.metric("Valid", "Yes" if validation["valid"] else "No")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distance", f"{validation['total_distance_m']} m")
    c2.metric("Battery used", validation["battery_used"])
    c3.metric("Final battery", validation["final_battery"])
    gap = comparison.get("distance_gap_percent")
    c4.metric("Dijkstra gap", f"{gap}%" if gap is not None else "N/A")

    st.markdown("**Route:**")
    if validation.get("route"):
        st.write("  →  ".join(validation["route"]))
    else:
        st.warning("No route extracted from the planner output.")

    st.markdown("**ENHSP plan:**")
    plan_text = read_text(planner_result["plan_file"])
    if plan_text.strip():
        st.code(plan_text, language="text")
    else:
        st.warning("Plan file is empty.")

    with st.expander("Planner log"):
        st.code(read_text(planner_result["log_file"])[:12000], language="text")


def _show_multi_output(result: dict[str, Any]) -> None:
    pvr = result.get("per_vehicle_results", [])
    all_valid = result.get("all_routes_valid", False)
    n = len(pvr)

    if all_valid:
        st.success(f"Valid routes found for all {n} vehicles.")
    else:
        st.warning("Not all vehicles reached their goals. Check logs.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", result["planner_result"].get("status"))
    c2.metric("Runtime", f"{result['planner_result'].get('runtime_seconds')} s")
    c3.metric("Total distance", f"{result.get('total_distance_m', 0)} m")

    for idx, vr in enumerate(pvr):
        icon = vehicle_icon(idx)
        vid = vr["vehicle_id"]
        status = "valid" if vr["route_valid"] else "no route"
        with st.expander(f"{icon} {vid}: {vr['start']} → {vr['goal']} ({status})", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Distance", f"{vr['planner_distance_m']} m")
            c2.metric("Battery used", vr["planner_battery_used"])
            c3.metric("Final battery", vr["final_battery"])
            gap = vr.get("distance_gap_percent")
            c4.metric("Dijkstra gap", f"{gap}%" if gap is not None else "N/A")
            if vr["route"]:
                st.write("Route: " + "  →  ".join(vr["route"]))
            else:
                st.warning("No route extracted.")

    st.markdown("**ENHSP plan:**")
    plan_text = read_text(result["planner_result"]["plan_file"])
    if plan_text.strip():
        st.code(plan_text, language="text")
    else:
        st.warning("Plan file is empty.")

    with st.expander("Planner log"):
        st.code(read_text(result["planner_result"]["log_file"])[:12000], language="text")


def open_sumo_gui_from_app(sumo_gui_path: str, sumo_config_file: str | Path) -> None:
    p = Path(sumo_config_file)
    subprocess.Popen([sumo_gui_path, "-c", p.name], cwd=p.parent)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Automated Planning — PDDL+", layout="wide")
    config = load_config()

    st.title("Automated Planning Final Project")
    st.markdown("### Map Construction and PDDL+ Planning App")
    st.caption(
        "Draw map area → extract graph → set constraints → "
        "select start/goal → generate PDDL+ → run ENHSP → visualize."
    )

    # Session state defaults
    for key, val in {
        "base_instance": None,
        "planning_result": None,
        "drawn_geometry": None,
        "selected_start": None,
        "selected_goal": None,
        "last_clicked_node": None,
        "custom_sumo_result": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ----------------------------------------------------------------
    # Sidebar
    # ----------------------------------------------------------------
    st.sidebar.header("Map Extraction")
    place_name = st.sidebar.text_input(
        "Search OpenStreetMap location",
        value=config["map"].get("place_name", "Rende, Calabria, Italy"),
    )
    network_type = st.sidebar.selectbox(
        "Routing mode", ["drive", "walk", "bike"], index=0
    )
    max_nodes = st.sidebar.slider(
        "Max planning graph nodes", min_value=10, max_value=150, value=30, step=10
    )
    default_speed = st.sidebar.number_input(
        "Default speed (m/s)",
        min_value=1.0, max_value=80.0,
        value=float(config["vehicle"].get("speed_m_per_s", 10.0)),
        step=1.0,
    )

    if st.sidebar.button("Extract whole searched place"):
        clear_custom_outputs(config)
        for k in ["planning_result", "selected_start", "selected_goal",
                  "last_clicked_node", "custom_sumo_result"]:
            st.session_state[k] = None
        with st.spinner("Extracting from OpenStreetMap..."):
            try:
                inst = prepare_custom_map(
                    place_name=place_name, network_type=network_type,
                    max_nodes=max_nodes, vehicle_speed_m_per_s=default_speed,
                    config=config,
                )
                st.session_state.base_instance = inst
                st.sidebar.success(
                    f"{inst['num_locations']} locations, {inst['num_edges']} edges, "
                    f"{inst.get('num_traffic_signals', 0)} 🚦, "
                    f"{inst.get('num_charging_stations', 0)} ⚡"
                )
            except Exception as exc:
                st.sidebar.error(f"Extraction failed: {exc}")

    # ----------------------------------------------------------------
    # Tabs
    # ----------------------------------------------------------------
    tab_area, tab_config, tab_result, tab_map, tab_pddl, tab_plan, tab_sumo = st.tabs([
        "Area Selection", "Planning Configuration",
        "Planner Result", "Interactive Map",
        "Generated PDDL+", "Plan / Logs", "SUMO Simulation",
    ])

    # ----------------------------------------------------------------
    # Tab 1 — Area selection
    # ----------------------------------------------------------------
    with tab_area:
        st.header("Step 1: Search and Draw Map Area")
        st.write(
            "Use the sidebar to search a location, or draw a polygon/rectangle "
            "on the map below to define the area."
        )

        center_lat, center_lon = geocode_place_center(place_name)
        draw_data = st_folium(
            build_area_selection_map(center_lat, center_lon),
            height=600, use_container_width=True, key="area_map",
        )

        geom = get_last_drawn_geometry(draw_data)
        if geom:
            st.session_state.drawn_geometry = geom
            st.success("Area drawn. Click Extract below.")

        if st.session_state.drawn_geometry:
            with st.expander("GeoJSON of selected area"):
                st.json(st.session_state.drawn_geometry)

        if st.button("Extract graph from drawn area", type="primary"):
            if not st.session_state.drawn_geometry:
                st.error("Draw a rectangle or polygon on the map first.")
            else:
                clear_custom_outputs(config)
                for k in ["planning_result", "selected_start", "selected_goal",
                          "last_clicked_node", "custom_sumo_result"]:
                    st.session_state[k] = None
                with st.spinner("Extracting OSM graph..."):
                    try:
                        inst = prepare_custom_map_from_polygon(
                            polygon_geojson=st.session_state.drawn_geometry,
                            place_name=place_name, network_type=network_type,
                            max_nodes=max_nodes, vehicle_speed_m_per_s=default_speed,
                            config=config,
                        )
                        st.session_state.base_instance = inst
                        st.success(
                            f"Extracted: {inst['num_locations']} locations, "
                            f"{inst['num_edges']} edges, "
                            f"{inst.get('num_traffic_signals', 0)} traffic signals, "
                            f"{inst.get('num_charging_stations', 0)} charging stations."
                        )
                    except Exception as exc:
                        st.error(f"Extraction failed: {exc}")

    base_instance = st.session_state.base_instance

    # ----------------------------------------------------------------
    # Tab 2 — Planning configuration
    # ----------------------------------------------------------------
    with tab_config:
        st.header("Step 2: Planning Configuration")

        if base_instance is None:
            st.info("Extract a map area first (Area Selection tab or sidebar).")
        else:
            # Map stats
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Locations", base_instance["num_locations"])
            c2.metric("Road edges", base_instance["num_edges"])
            c3.metric("Routing mode", base_instance["network_type"])
            c4.metric("🚦 Signals", base_instance.get("num_traffic_signals", 0))
            c5.metric("⚡ Chargers", base_instance.get("num_charging_stations", 0))

            st.divider()

            # --------------------------------------------------------
            # Step 3 — Constraints
            # All constraints are independent and stackable.
            # --------------------------------------------------------
            st.subheader("Step 3: Set Constraints")

            # Number of vehicles
            n_vehicles = st.number_input(
                "Number of vehicles",
                min_value=1, max_value=5, value=1, step=1,
                help=(
                    "1 = single-vehicle planning. "
                    "2–5 = multi-vehicle. "
                    "Shared starts and shared goals are allowed."
                ),
            )
            is_multi = n_vehicles > 1

            if is_multi:
                # Show colour legend
                legend_parts = [
                    f"{vehicle_icon(i)} car{i+1}"
                    for i in range(n_vehicles)
                ]
                st.caption(
                    "Vehicle colours: " + "   ".join(legend_parts) + "\n\n"
                    "On the map: **filled** circle = START  |  "
                    "**outlined** circle = GOAL  |  "
                    "🟡 yellow = last clicked node"
                )

            st.markdown("**Vehicle parameters**")
            c1, c2, c3 = st.columns(3)
            with c1:
                default_battery = st.number_input(
                    "Initial battery (per vehicle)",
                    min_value=1.0, max_value=500.0,
                    value=float(config["vehicle"].get("initial_battery", 100.0)),
                    step=5.0,
                )
            with c2:
                speed_m_per_s = st.number_input(
                    "Vehicle speed (m/s)",
                    min_value=1.0, max_value=40.0,
                    value=default_speed, step=1.0,
                )
            with c3:
                consumption = st.number_input(
                    "Battery consumption per meter",
                    min_value=0.001, max_value=1.0,
                    value=float(config["vehicle"].get("battery_consumption_per_meter", 0.01)),
                    step=0.001, format="%.3f",
                )

            metric = st.radio(
                "Planning metric", options=["distance", "time"], horizontal=True,
            )

            # Low battery
            use_low_battery = st.checkbox(
                "Enable low battery scenario",
                help="Override initial battery with a constrained lower value.",
            )
            if use_low_battery:
                initial_battery = st.slider(
                    "Low battery value",
                    min_value=5.0, max_value=float(default_battery),
                    value=min(float(default_battery), 40.0), step=5.0,
                )
            else:
                initial_battery = float(default_battery)

            # Blocked roads
            st.markdown("**Blocked road segments** *(optional — models planned closures)*")
            selected_blocked = st.multiselect(
                "Select segments to block",
                options=list_edge_options(base_instance),
                help="Vehicles cannot use these segments. Models planned closures known before departure.",
            )

            # Congested roads
            st.markdown("**Congested road segments** *(optional — models reduced speed)*")
            selected_congested = st.multiselect(
                "Select segments to congest",
                options=list_edge_options(base_instance),
                help=(
                    "Effective road distance is inflated by 1/factor. "
                    "The planner may reroute if a detour is cheaper. "
                    "Models known congestion at planning time."
                ),
            )
            if selected_congested:
                congestion_factor = st.slider(
                    "Congestion factor (speed multiplier)",
                    min_value=0.1, max_value=0.9, value=0.3, step=0.1,
                    help="0.1 = 10% of normal speed (severe). 0.9 = 90% (mild).",
                )
            else:
                congestion_factor = 0.3

            blocked_edges = build_blocked_edge_objects(selected_blocked, base_instance)
            congested_edges = build_congested_edge_objects(
                selected_congested, base_instance, congestion_factor
            )

            st.divider()

            # --------------------------------------------------------
            # Step 4 — Initial and goal state selection
            # --------------------------------------------------------
            st.subheader("Step 4: Select Initial and Goal States")

            # Node selection map
            node_click_data = st_folium(
                build_node_selection_map(base_instance, n_vehicles),
                height=520, use_container_width=True, key="node_map",
            )

            # Update last clicked node
            clicked = node_click_data.get("last_clicked") if node_click_data else None
            if clicked:
                nearest = find_nearest_location_id(
                    base_instance, clicked["lat"], clicked["lng"]
                )
                st.session_state.last_clicked_node = nearest

            last_clicked = st.session_state.last_clicked_node
            if last_clicked:
                st.info(f"Last clicked node: **`{last_clicked}`**")

            location_options = list_location_options(base_instance)
            location_ids = [o.split("|")[0].strip() for o in location_options]
            n_locs = len(location_ids)

            # ---- Single vehicle ----
            if not is_multi:
                if last_clicked:
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Set clicked node as START"):
                            st.session_state.selected_start = last_clicked
                            st.rerun()
                    with c2:
                        if st.button("Set clicked node as GOAL"):
                            st.session_state.selected_goal = last_clicked
                            st.rerun()

                c1, c2 = st.columns(2)
                with c1:
                    si = 0
                    if st.session_state.selected_start in location_ids:
                        si = location_ids.index(st.session_state.selected_start)
                    start_label = st.selectbox(
                        "Initial state: vehicle at",
                        options=location_options, index=si,
                    )
                with c2:
                    gi = n_locs - 1
                    if st.session_state.selected_goal in location_ids:
                        gi = location_ids.index(st.session_state.selected_goal)
                    goal_label = st.selectbox(
                        "Goal state: vehicle at",
                        options=location_options, index=gi,
                    )

                start_location = extract_location_id(start_label)
                goal_location = extract_location_id(goal_label)
                st.session_state.selected_start = start_location
                st.session_state.selected_goal = goal_location

                if start_location == goal_location:
                    st.warning("Start and goal are the same location.")

            # ---- Multi-vehicle ----
            vehicles: list[dict[str, Any]] = []

            if is_multi:
                st.markdown(
                    f"Configure {n_vehicles} vehicles below. "
                    "**Shared starts and shared goals are allowed.**"
                )

                for i in range(1, n_vehicles + 1):
                    vid = f"car{i}"
                    idx = i - 1
                    icon = vehicle_icon(idx)

                    st.markdown(f"---\n**{icon} {vid}**")

                    # Map-click shortcut buttons
                    if last_clicked:
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button(
                                f"Set `{last_clicked}` as {icon} {vid} START",
                                key=f"btn_start_{i}",
                            ):
                                st.session_state[f"mv_start_{i}"] = last_clicked
                                st.rerun()
                        with c2:
                            if st.button(
                                f"Set `{last_clicked}` as {icon} {vid} GOAL",
                                key=f"btn_goal_{i}",
                            ):
                                st.session_state[f"mv_goal_{i}"] = last_clicked
                                st.rerun()

                    # Default spread: starts all at loc_0 (shared depot),
                    # goals spread across graph
                    default_start_idx = 0
                    default_goal_idx = min(
                        int((i * n_locs) / (n_vehicles + 1)), n_locs - 1
                    )
                    if default_start_idx == default_goal_idx:
                        default_goal_idx = min(default_goal_idx + 1, n_locs - 1)

                    sv_s = st.session_state.get(f"mv_start_{i}")
                    sv_g = st.session_state.get(f"mv_goal_{i}")
                    si = location_ids.index(sv_s) if sv_s in location_ids else default_start_idx
                    gi = location_ids.index(sv_g) if sv_g in location_ids else default_goal_idx

                    c1, c2 = st.columns(2)
                    with c1:
                        v_start_label = st.selectbox(
                            f"{icon} {vid} — Start",
                            options=location_options, index=si,
                            key=f"sel_start_{i}",
                        )
                    with c2:
                        v_goal_label = st.selectbox(
                            f"{icon} {vid} — Goal",
                            options=location_options, index=gi,
                            key=f"sel_goal_{i}",
                        )

                    v_start = extract_location_id(v_start_label)
                    v_goal = extract_location_id(v_goal_label)

                    # Per-vehicle battery override
                    with st.expander(f"Battery override for {icon} {vid}"):
                        v_battery = st.number_input(
                            f"{vid} initial battery",
                            min_value=1.0, max_value=500.0,
                            value=float(initial_battery),
                            step=5.0, key=f"mv_battery_{i}",
                        )

                    vehicles.append({
                        "id": vid,
                        "start": v_start,
                        "goal": v_goal,
                        "battery": float(
                            st.session_state.get(f"mv_battery_{i}", initial_battery)
                        ),
                        "speed_m_per_s": float(speed_m_per_s),
                        "battery_consumption_per_meter": float(consumption),
                        "max_battery": float(default_battery),
                        "charge_rate": 5.0,
                    })

            # --------------------------------------------------------
            # Step 5 — Run planner
            # --------------------------------------------------------
            st.divider()
            st.subheader("Step 5: Run Planner")

            btn_label = (
                f"Generate PDDL+ and run ENHSP ({n_vehicles} vehicle{'s' if n_vehicles > 1 else ''})"
            )

            if st.button(btn_label, type="primary"):
                st.session_state.custom_sumo_result = None

                if is_multi:
                    with st.spinner(f"Running ENHSP for {n_vehicles} vehicles..."):
                        try:
                            result = run_multi_vehicle_planning_job(
                                base_instance=base_instance,
                                vehicles=vehicles,
                                metric=metric,
                                blocked_edges=blocked_edges,
                                congested_edges=congested_edges,
                                config=config,
                            )
                            st.session_state.planning_result = result
                            show_planner_output(result)
                        except Exception as exc:
                            st.error(f"Multi-vehicle planning failed: {exc}")
                else:
                    if start_location == goal_location:
                        st.error("Start and goal must be different.")
                    else:
                        with st.spinner("Generating PDDL+, running ENHSP..."):
                            try:
                                instance, custom_config = apply_user_choices_to_instance(
                                    base_instance=base_instance,
                                    start=start_location,
                                    goal=goal_location,
                                    initial_battery=initial_battery,
                                    speed_m_per_s=speed_m_per_s,
                                    battery_consumption_per_meter=consumption,
                                    metric=metric,
                                    blocked_edges=blocked_edges,
                                    congested_edges=congested_edges,
                                    config=config,
                                )
                                result = run_custom_planning_job(
                                    instance=instance,
                                    config=custom_config,
                                    metric=metric,
                                )
                                st.session_state.planning_result = result
                                show_planner_output(result)
                            except Exception as exc:
                                st.error(f"Planning failed: {exc}")

    result = st.session_state.planning_result

    # ----------------------------------------------------------------
    # Tab 3 — Planner result
    # ----------------------------------------------------------------
    with tab_result:
        st.header("Planner Result")
        if not result:
            st.info("Run the planner first.")
        elif result.get("is_multi_vehicle"):
            pvr = result.get("per_vehicle_results", [])
            all_valid = result.get("all_routes_valid", False)

            c1, c2, c3 = st.columns(3)
            c1.metric("All routes valid", "Yes" if all_valid else "No")
            c2.metric("Total distance", f"{result.get('total_distance_m', 0)} m")
            c3.metric("Vehicles", result.get("num_vehicles", len(pvr)))

            for idx, vr in enumerate(pvr):
                icon = vehicle_icon(idx)
                st.markdown(f"**{icon} {vr['vehicle_id']}: {vr['start']} → {vr['goal']}**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Distance", f"{vr['planner_distance_m']} m")
                c2.metric("Battery used", vr["planner_battery_used"])
                c3.metric("Final battery", vr["final_battery"])
                gap = vr.get("distance_gap_percent")
                c4.metric("Dijkstra gap", f"{gap}%" if gap is not None else "N/A")
                if vr["route"]:
                    st.write("Route: " + "  →  ".join(vr["route"]))
                else:
                    st.warning("No route extracted.")
                st.divider()
        else:
            validation = result["validation"]
            comparison = result["comparison"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Valid", "Yes" if validation["valid"] else "No")
            c2.metric("Runtime", f"{result['planner_result']['runtime_seconds']} s")
            c3.metric("Distance", f"{validation['total_distance_m']} m")
            c4.metric("Final battery", validation["final_battery"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Dijkstra distance", comparison["dijkstra_distance_m"])
            gap = comparison.get("distance_gap_percent")
            c2.metric("Distance gap", f"{gap}%" if gap is not None else "N/A")
            c3.metric("Moves", validation["num_move_actions"])

            if validation.get("errors"):
                st.error("Validation errors")
                st.write(validation["errors"])
            if validation.get("warnings"):
                st.warning("Validation warnings")
                st.write(validation["warnings"])

            st.subheader("Route")
            st.write("  →  ".join(validation["route"]))
            st.subheader("Comparison JSON")
            st.json(comparison)

    # ----------------------------------------------------------------
    # Tab 4 — Interactive map
    # ----------------------------------------------------------------
    with tab_map:
        st.header("Interactive Route Map")
        if not result or not result.get("interactive_map_path"):
            st.info("Run the planner first.")
        else:
            if result.get("is_multi_vehicle"):
                n_v = result.get("num_vehicles", 1)
                legend = "  ".join(
                    f"{vehicle_icon(i)} car{i+1}" for i in range(n_v)
                )
                st.caption(f"Showing car1 route (red=ENHSP, blue=Dijkstra). Vehicles: {legend}")
            render_html_file(result["interactive_map_path"])

    # ----------------------------------------------------------------
    # Tab 5 — Generated PDDL+
    # ----------------------------------------------------------------
    with tab_pddl:
        st.header("Generated PDDL+")
        if not result:
            st.info("Run the planner first.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Domain")
                st.code(read_text(result["domain_path"])[:15000], language="lisp")
            with c2:
                st.subheader("Problem")
                st.code(read_text(result["problem_path"])[:15000], language="lisp")

    # ----------------------------------------------------------------
    # Tab 6 — Plan and logs
    # ----------------------------------------------------------------
    with tab_plan:
        st.header("Plan and Logs")
        if not result:
            st.info("Run the planner first.")
        else:
            st.subheader("ENHSP plan")
            st.code(read_text(result["planner_result"]["plan_file"]), language="text")
            st.subheader("Planner log")
            st.code(read_text(result["planner_result"]["log_file"])[:12000], language="text")

    # ----------------------------------------------------------------
    # Tab 7 — SUMO simulation
    # ----------------------------------------------------------------
    with tab_sumo:
        st.header("SUMO Simulation")
        result = st.session_state.planning_result

        if not result:
            st.info("Run the planner first.")
        else:
            is_multi = result.get("is_multi_vehicle", False)
            plan_valid = (
                result.get("all_routes_valid", False)
                if is_multi
                else result["validation"]["valid"]
            )

            if not plan_valid:
                st.warning("Plan is not valid. SUMO simulation cannot be generated.")
            else:
                if is_multi:
                    n_v = result.get("num_vehicles", 1)
                    legend = "  ".join(
                        f"{vehicle_icon(i)} car{i+1}" for i in range(n_v)
                    )
                    st.write(f"Simulating {n_v} vehicles: {legend}")
                else:
                    st.write("🔴 Red = ENHSP planner vehicle. 🔵 Blue = background traffic.")

                bg_count = st.number_input(
                    "Background vehicles", min_value=0, max_value=300, value=100, step=10
                )

                if st.button("Generate SUMO simulation", type="primary"):
                    with st.spinner("Generating SUMO files..."):
                        try:
                            sumo_cmp = (
                                {
                                    "per_vehicle_results": result["per_vehicle_results"],
                                    "planner_route": (
                                        result["per_vehicle_results"][0]["route"]
                                        if result["per_vehicle_results"] else []
                                    ),
                                }
                                if is_multi
                                else result["comparison"]
                            )
                            sumo_result = generate_custom_sumo_simulation(
                                instance=result["instance"],
                                comparison=sumo_cmp,
                                config=config,
                                background_vehicle_count=int(bg_count),
                            )
                            st.session_state.custom_sumo_result = sumo_result
                            if sumo_result["success"]:
                                st.success("SUMO simulation generated successfully.")
                            else:
                                st.warning("SUMO files generated but validation failed.")
                        except Exception as exc:
                            st.error(f"SUMO generation failed: {exc}")

                sumo_result = st.session_state.custom_sumo_result
                if sumo_result:
                    st.success(f"Files: `{sumo_result.get('output_dir', '')}`")
                    st.warning("Close SUMO-GUI before generating a new simulation.")

                    if st.button("Open SUMO Simulation", type="primary"):
                        try:
                            open_sumo_gui_from_app(
                                config["sumo"]["sumo_gui_path"],
                                sumo_result["config_file"],
                            )
                            st.success("SUMO-GUI opened.")
                        except FileNotFoundError:
                            st.error("SUMO-GUI not found. Check sumo_gui_path in config.yaml.")
                        except Exception as exc:
                            st.error(f"Could not open SUMO-GUI: {exc}")

                    with st.expander("SUMO file details"):
                        st.write(f"Network: `{sumo_result['network_file']}`")
                        st.write(f"Routes: `{sumo_result['route_file']}`")
                        st.write(f"Config: `{sumo_result['config_file']}`")
                        st.write(f"SUMO log: `{sumo_result['sumo_validation_log']}`")
                        st.write(f"netconvert log: `{sumo_result['netconvert_log']}`")


if __name__ == "__main__":
    main()