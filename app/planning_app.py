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
    clear_custom_outputs,
    extract_location_id,
    find_nearest_location_id,
    list_edge_options,
    list_location_options,
    prepare_custom_map,
    prepare_custom_map_from_polygon,
    run_custom_planning_job,
    generate_custom_sumo_simulation,

)


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"

    if not config_path.exists():
        st.error(f"config.yaml not found: {config_path}")
        st.stop()

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_text(path: str | Path) -> str:
    file_path = Path(path)

    if not file_path.exists():
        return ""

    return file_path.read_text(encoding="utf-8", errors="replace")


def render_html_file(path: str | Path, height: int = 650) -> None:
    file_path = Path(path)

    if not file_path.exists():
        st.warning(f"Map file not found: {file_path}")
        return

    html = file_path.read_text(encoding="utf-8", errors="replace")
    components.html(html, height=height, scrolling=True)


@st.cache_data(show_spinner=False)
def geocode_place_center(place_name: str) -> tuple[float, float]:
    """
    Convert a place name into a map center coordinate.
    """
    try:
        lat, lon = ox.geocode(place_name)
        return float(lat), float(lon)
    except Exception:
        # Safe fallback around Rende / University of Calabria.
        return 39.3500, 16.2250


def get_last_drawn_geometry(draw_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Extract the last drawn polygon/rectangle geometry from streamlit-folium output.
    """
    if not draw_data:
        return None

    all_drawings = draw_data.get("all_drawings")

    if not all_drawings:
        return None

    last_feature = all_drawings[-1]

    if not last_feature:
        return None

    geometry = last_feature.get("geometry")

    if not geometry:
        return None

    if geometry.get("type") not in ["Polygon", "MultiPolygon"]:
        return None

    return geometry


def build_area_selection_map(
    center_lat: float,
    center_lon: float,
    zoom_start: int = 14,
) -> folium.Map:
    """
    Map where the user draws the extraction polygon/rectangle.
    """
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": True,
            "rectangle": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)

    return fmap


def build_node_selection_map(
    instance: dict[str, Any],
    start_location: str | None = None,
    goal_location: str | None = None,
) -> folium.Map:
    """
    Map where extracted nodes are displayed and the user can click near a node.
    """
    latitudes = [float(location["lat"]) for location in instance["locations"]]
    longitudes = [float(location["lon"]) for location in instance["locations"]]

    center_lat = sum(latitudes) / len(latitudes)
    center_lon = sum(longitudes) / len(longitudes)

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    coordinates = {
        location["id"]: (float(location["lat"]), float(location["lon"]))
        for location in instance["locations"]
    }

    # Draw OSM-derived planning graph edges.
    for edge in instance["edges"]:
        source = edge["from"]
        target = edge["to"]

        if source not in coordinates or target not in coordinates:
            continue

        folium.PolyLine(
            locations=[coordinates[source], coordinates[target]],
            color="gray",
            weight=2,
            opacity=0.45,
            tooltip=f"{source} -> {target}",
        ).add_to(fmap)

    # Draw selectable planning locations.
    for location in instance["locations"]:
        location_id = location["id"]
        lat = float(location["lat"])
        lon = float(location["lon"])

        color = "blue"
        radius = 4

        if location_id == start_location:
            color = "green"
            radius = 9
        elif location_id == goal_location:
            color = "purple"
            radius = 9

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.90,
            tooltip=f"{location_id} ({lat:.5f}, {lon:.5f})",
        ).add_to(fmap)

    return fmap


def show_immediate_planner_output(result: dict[str, Any]) -> None:
    """
    Show the result immediately under the Run Planner button.
    """
    validation = result["validation"]
    planner_result = result["planner_result"]
    comparison = result["comparison"]

    if validation["valid"]:
        st.success("Planner produced a valid plan.")
    else:
        st.error("Planner did not produce a valid plan.")

    st.subheader("Immediate planner output")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Planner status", planner_result.get("status"))

    with col2:
        st.metric("Runtime", f"{planner_result.get('runtime_seconds')} s")

    with col3:
        st.metric("Plan length", planner_result.get("plan_length"))

    with col4:
        st.metric("Valid", "Yes" if validation["valid"] else "No")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Distance", f"{validation['total_distance_m']} m")

    with col2:
        st.metric("Battery used", validation["battery_used"])

    with col3:
        st.metric("Final battery", validation["final_battery"])

    with col4:
        gap = comparison.get("distance_gap_percent")
        st.metric("Dijkstra gap", f"{gap}%" if gap is not None else "N/A")

    st.markdown("### Route")
    if validation.get("route"):
        st.write("  ->  ".join(validation["route"]))
    else:
        st.warning("No route was extracted from the planner output.")

    st.markdown("### ENHSP plan")
    plan_text = read_text(planner_result["plan_file"])

    if plan_text.strip():
        st.code(plan_text, language="text")
    else:
        st.warning("Plan file is empty or was not generated.")

    with st.expander("Planner log"):
        st.code(read_text(planner_result["log_file"])[:12000], language="text")

def open_sumo_gui_from_app(
    sumo_gui_path: str,
    sumo_config_file: str | Path,
) -> None:
    """
    Launch SUMO-GUI for the generated custom simulation.
    Works when Streamlit is running locally on the same machine as SUMO.
    """
    config_path = Path(sumo_config_file)
    working_dir = config_path.parent

    subprocess.Popen(
        [
            sumo_gui_path,
            "-c",
            config_path.name,
        ],
        cwd=working_dir,
    )

# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Automated Planning Final Project",
        # page_icon="🗺️",
        layout="wide",
    )

    config = load_config()
    st.title("Automated Planning Final Project")
    st.markdown("### Map Construction and PDDL+ Planning App")
    st.caption(
        "Search/draw map area => extract graph => select initial/goal state => "
        "set constraints/scenarios => generate PDDL+ => run ENHSP => validate and visualize."
    )

    # Session state
    if "base_instance" not in st.session_state:
        st.session_state.base_instance = None

    if "planning_result" not in st.session_state:
        st.session_state.planning_result = None

    if "drawn_geometry" not in st.session_state:
        st.session_state.drawn_geometry = None

    if "selected_start" not in st.session_state:
        st.session_state.selected_start = None

    if "selected_goal" not in st.session_state:
        st.session_state.selected_goal = None

    if "last_clicked_node" not in st.session_state:
        st.session_state.last_clicked_node = None
    if "custom_sumo_result" not in st.session_state:
        st.session_state.custom_sumo_result = None
    # Sidebar controls
    st.sidebar.header("Visual Map / Area Selection")

    place_name = st.sidebar.text_input(
        "Search OpenStreetMap location",
        value=config["map"].get("place_name", "Rende, Calabria, Italy"),
    )

    network_type = st.sidebar.selectbox(
        "Routing mode",
        options=["drive", "walk", "bike"],
        index=0,
    )

    max_nodes = st.sidebar.slider(
        "Maximum planning graph nodes",
        min_value=10,
        max_value=150,
        value=30,
        step=10,
    )

    default_speed = st.sidebar.number_input(
        "Default vehicle speed (m/s)",
        min_value=1.0,
        max_value=80.0,
        value=float(config["vehicle"].get("speed_m_per_s", 10.0)),
        step=1.0,
    )

    # st.sidebar.caption(
    #     "Preferred workflow: draw an area in the Area Selection tab, then extract the graph."
    # )

    # Optional fallback if user wants to extract by place name only.
    if st.sidebar.button("Extract whole searched place"):
        clear_custom_outputs(config)
        st.session_state.planning_result = None
        st.session_state.selected_start = None
        st.session_state.selected_goal = None
        st.session_state.last_clicked_node = None

        with st.spinner("Extracting place from OpenStreetMap..."):
            try:
                st.session_state.base_instance = prepare_custom_map(
                    place_name=place_name,
                    network_type=network_type,
                    max_nodes=max_nodes,
                    vehicle_speed_m_per_s=default_speed,
                    config=config,
                )

                instance = st.session_state.base_instance

                st.success(
                    f"Map prepared: {instance['num_locations']} locations, "
                    f"{instance['num_edges']} road edges."
                )

            except Exception as exc:
                st.error(f"Map extraction failed: {exc}")

    tab_area, tab_config, tab_result, tab_map, tab_pddl, tab_plan, tab_sumo = st.tabs(
        [
            "Area Selection",
            "Planning Configuration",
            "Planner Result",
            "Interactive Map",
            "Generated PDDL+",
            "Plan / Logs",
            "SUMO Simulation",
        ]
    )

    # -----------------------------------------------------------------
    # Tab 1: visual area selection
    # -----------------------------------------------------------------
    with tab_area:
        st.header("Step 1: Search and Draw Map Area")

        st.write(
            "Search a place to center the map, then draw a rectangle or polygon "
            "around the area you want to convert into a planning problem."
        )

        center_lat, center_lon = geocode_place_center(place_name)
        area_map = build_area_selection_map(center_lat, center_lon)

        draw_data = st_folium(
            area_map,
            height=600,
            use_container_width=True,
            key="area_selection_map",
        )

        drawn_geometry = get_last_drawn_geometry(draw_data)

        if drawn_geometry:
            st.session_state.drawn_geometry = drawn_geometry
            st.success("Area selected. You can now extract the planning graph.")

        if st.session_state.drawn_geometry:
            with st.expander("Selected GeoJSON geometry"):
                st.json(st.session_state.drawn_geometry)

        extract_area_clicked = st.button(
            "Extract graph from selected area",
            type="primary",
        )

        if extract_area_clicked:
            if not st.session_state.drawn_geometry:
                st.error("Please draw a rectangle or polygon on the map first.")
            else:
                clear_custom_outputs(config)
                st.session_state.planning_result = None
                st.session_state.selected_start = None
                st.session_state.selected_goal = None
                st.session_state.last_clicked_node = None
                st.session_state.custom_sumo_result = None


                with st.spinner("Extracting OSM graph from selected polygon..."):
                    try:
                        base_instance = prepare_custom_map_from_polygon(
                            polygon_geojson=st.session_state.drawn_geometry,
                            place_name=place_name,
                            network_type=network_type,
                            max_nodes=max_nodes,
                            vehicle_speed_m_per_s=default_speed,
                            config=config,
                        )

                        st.session_state.base_instance = base_instance

                        st.success(
                            f"Extracted graph: {base_instance['num_locations']} locations, "
                            f"{base_instance['num_edges']} road edges."
                        )

                    except Exception as exc:
                        st.error(f"Graph extraction failed: {exc}")

    # Get latest extracted graph after area/place extraction.
    base_instance = st.session_state.base_instance

    # -----------------------------------------------------------------
    # Tab 2: start/goal, constraints, scenarios, planner
    # -----------------------------------------------------------------
    with tab_config:
        st.header("Step 2: Select Initial and Goal States")

        if base_instance is None:
            st.info("Extract a map area first from the Area Selection tab.")
        else:
            st.subheader("Extracted map")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Locations", base_instance["num_locations"])

            with col2:
                st.metric("Road edges", base_instance["num_edges"])

            with col3:
                st.metric("Routing mode", base_instance["network_type"])
            st.markdown("#### Option A: Select on the map:")
            st.write(
                "Click near a graph node on the map, then assign the clicked node "
                "as the start or goal. Dropdown selection is also available as fallback."
            )

            node_map = build_node_selection_map(
                instance=base_instance,
                start_location=st.session_state.selected_start,
                goal_location=st.session_state.selected_goal,
            )

            node_click_data = st_folium(
                node_map,
                height=550,
                use_container_width=True,
                key="node_selection_map",
            )

            clicked = node_click_data.get("last_clicked") if node_click_data else None

            if clicked:
                clicked_lat = clicked["lat"]
                clicked_lon = clicked["lng"]

                nearest_location = find_nearest_location_id(
                    instance=base_instance,
                    clicked_lat=clicked_lat,
                    clicked_lon=clicked_lon,
                )

                st.session_state.last_clicked_node = nearest_location

            if st.session_state.last_clicked_node:
                st.info(
                    f"Nearest extracted node to last click: "
                    f"`{st.session_state.last_clicked_node}`"
                )

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Use clicked node as START"):
                        st.session_state.selected_start = st.session_state.last_clicked_node
                        st.rerun()

                with col2:
                    if st.button("Use clicked node as GOAL"):
                        st.session_state.selected_goal = st.session_state.last_clicked_node
                        st.rerun()

            st.markdown("#### Option B: Select from dropdown")

            location_options = list_location_options(base_instance)
            location_ids = [option.split("|")[0].strip() for option in location_options]

            default_start_index = 0
            default_goal_index = len(location_options) - 1

            if st.session_state.selected_start in location_ids:
                default_start_index = location_ids.index(st.session_state.selected_start)

            if st.session_state.selected_goal in location_ids:
                default_goal_index = location_ids.index(st.session_state.selected_goal)

            col1, col2 = st.columns(2)

            with col1:
                start_label = st.selectbox(
                    "Initial state: vehicle at",
                    options=location_options,
                    index=default_start_index,
                )

            with col2:
                goal_label = st.selectbox(
                    "Goal state: vehicle at",
                    options=location_options,
                    index=default_goal_index,
                )

            start_location = extract_location_id(start_label)
            goal_location = extract_location_id(goal_label)

            st.session_state.selected_start = start_location
            st.session_state.selected_goal = goal_location

            if start_location == goal_location:
                st.warning("Start and goal are the same. Select different locations.")

            st.subheader("Step 3: Set Constraints")

            col1, col2, col3 = st.columns(3)

            with col1:
                initial_battery = st.number_input(
                    "Initial battery",
                    min_value=1.0,
                    max_value=500.0,
                    value=float(config["vehicle"].get("initial_battery", 100.0)),
                    step=5.0,
                )

            with col2:
                speed_m_per_s = st.number_input(
                    "Vehicle speed (m/s)",
                    min_value=1.0,
                    max_value=40.0,
                    value=default_speed,
                    step=1.0,
                )

            with col3:
                consumption = st.number_input(
                    "Battery consumption per meter",
                    min_value=0.001,
                    max_value=1.0,
                    value=float(config["vehicle"].get("battery_consumption_per_meter", 0.01)),
                    step=0.001,
                    format="%.3f",
                )

            metric = st.radio(
                "Planning metric",
                options=["distance", "time"],
                horizontal=True,
            )

            st.subheader("Step 4: Choose Scenario")

            scenario = st.selectbox(
                "Scenario type",
                options=[
                    "Normal planning",
                    "Low battery",
                    "Blocked road",
                    "Blocked road + low battery",
                ],
            )

            if scenario in ["Low battery", "Blocked road + low battery"]:
                initial_battery = st.slider(
                    "Low-battery value",
                    min_value=5.0,
                    max_value=100.0,
                    value=min(initial_battery, 40.0),
                    step=5.0,
                )

            selected_blocked_edges = []

            if scenario in ["Blocked road", "Blocked road + low battery"]:
                selected_blocked_edges = st.multiselect(
                    "Select road segment(s) to block",
                    options=list_edge_options(base_instance),
                    help=(
                        "These are real OSM-derived road segments. "
                        "The blocked status is a planning scenario constraint."
                    ),
                )

            blocked_edges = build_blocked_edge_objects(
                selected_edge_labels=selected_blocked_edges,
                instance=base_instance,
            )

            st.subheader("Step 5: Run Planner")

            if st.button("Generate PDDL+ and run ENHSP planner", type="primary"):
                if start_location == goal_location:
                    st.error("Start and goal must be different.")
                else:
                    st.session_state.custom_sumo_result = None
                    with st.spinner("Generating PDDL+, running ENHSP, validating plan..."):
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
                                config=config,
                            )

                            result = run_custom_planning_job(
                                instance=instance,
                                config=custom_config,
                                metric=metric,
                            )

                            st.session_state.planning_result = result
                            show_immediate_planner_output(result)

                        except Exception as exc:
                            st.error(f"Planning failed: {exc}")

    result = st.session_state.planning_result

    # -----------------------------------------------------------------
    # Tab 3: planner result
    # -----------------------------------------------------------------
    with tab_result:
        st.header("Planner Result")

        if not result:
            st.info("Run the planner first.")
        else:
            validation = result["validation"]
            comparison = result["comparison"]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Plan valid", "Yes" if validation["valid"] else "No")
            with col2:
                st.metric(
                    "Planner runtime",
                    f"{result['planner_result']['runtime_seconds']} s",
                )
            with col3:
                st.metric("Distance", f"{validation['total_distance_m']} m")
            with col4:
                st.metric("Final battery", validation["final_battery"])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Dijkstra distance", comparison["dijkstra_distance_m"])
            with col2:
                gap = comparison.get("distance_gap_percent")
                st.metric("Distance gap", f"{gap}%" if gap is not None else "N/A")
            with col3:
                st.metric("Moves", validation["num_move_actions"])

            if validation["errors"]:
                st.error("Validation errors")
                st.write(validation["errors"])

            if validation["warnings"]:
                st.warning("Validation warnings")
                st.write(validation["warnings"])

            st.subheader("Route")
            st.write("  ->  ".join(validation["route"]))

            st.subheader("Comparison JSON")
            st.json(comparison)

    # -----------------------------------------------------------------
    # Tab 4: interactive Folium result map
    # -----------------------------------------------------------------
    with tab_map:
        st.header("Interactive Route Map")

        if not result or not result.get("interactive_map_path"):
            st.info("Run the planner first to generate the interactive map.")
        else:
            render_html_file(result["interactive_map_path"])

    # -----------------------------------------------------------------
    # Tab 5: generated PDDL+
    # -----------------------------------------------------------------
    with tab_pddl:
        st.header("Generated PDDL+")

        if not result:
            st.info("Run the planner first.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Domain")
                st.code(read_text(result["domain_path"])[:15000], language="lisp")

            with col2:
                st.subheader("Problem")
                st.code(read_text(result["problem_path"])[:15000], language="lisp")

    # -----------------------------------------------------------------
    # Tab 6: ENHSP plan and logs
    # -----------------------------------------------------------------
    with tab_plan:
        st.header("Plan and Logs")

        if not result:
            st.info("Run the planner first.")
        else:
            st.subheader("ENHSP plan")
            st.code(read_text(result["planner_result"]["plan_file"]), language="text")

            st.subheader("Planner log")
            st.code(
                read_text(result["planner_result"]["log_file"])[:12000],
                language="text",
            )

    with tab_sumo:
        st.header("Custom SUMO Simulation")

        result = st.session_state.planning_result

        if not result:
            st.info("Run the PDDL+ planner first before generating SUMO simulation.")
        elif not result["validation"]["valid"]:
            st.warning("The current plan is not valid, so SUMO simulation cannot be generated.")
        else:
            st.write(
                "Generate a SUMO simulation for the current user-selected route. "
                "The red vehicle represents the ENHSP planner route."
            )

            background_vehicle_count = st.number_input(
                "Background vehicles",
                min_value=0,
                max_value=300,
                value=100,
                step=10,
            )

            if st.button("Generate custom SUMO simulation", type="primary"):
                with st.spinner("Generating SUMO network, route, and simulation config..."):
                    try:
                        sumo_result = generate_custom_sumo_simulation(
                            instance=result["instance"],
                            comparison=result["comparison"],
                            config=config,
                            background_vehicle_count=int(background_vehicle_count),
                        )

                        st.session_state.custom_sumo_result = sumo_result

                        if sumo_result["success"]:
                            st.success("SUMO simulation generated and validated successfully.")
                        else:
                            st.warning(
                                "SUMO files were generated, but validation reported a problem. "
                                "Check the SUMO validation log."
                            )

                    except Exception as exc:
                        st.error(f"SUMO generation failed: {exc}")

            sumo_result = st.session_state.custom_sumo_result

            if sumo_result:
                st.success("SUMO files are saved under `outputs\\sumo\\custom`.")

                st.warning(
                    "Close SUMO-GUI before generating a new simulation, otherwise Windows may lock the SUMO files."
                )

                if st.button("Open SUMO Simulation", type="primary"):
                    try:
                        open_sumo_gui_from_app(
                            sumo_gui_path=config["sumo"]["sumo_gui_path"],
                            sumo_config_file=sumo_result["config_file"],
                        )

                        st.success("SUMO-GUI opened successfully.")

                    except FileNotFoundError:
                        st.error("Could not find SUMO-GUI. Check `sumo_gui_path` in `config.yaml`.")

                    except Exception as exc:
                        st.error(f"Could not open SUMO-GUI: {exc}")

                with st.expander("Show generated SUMO file details"):
                    st.write(f"Network: `{sumo_result['network_file']}`")
                    st.write(f"Route: `{sumo_result['route_file']}`")
                    st.write(f"Config: `{sumo_result['config_file']}`")
                    st.write(f"SUMO validation log: `{sumo_result['sumo_validation_log']}`")
                    st.write(f"netconvert log: `{sumo_result['netconvert_log']}`")
if __name__ == "__main__":
    main()
