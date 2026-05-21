from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


def load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"

    if not config_path.exists():
        st.error(f"config.yaml not found: {config_path}")
        st.stop()

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(path: Path) -> Any:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_text(path: Path) -> str:
    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8", errors="replace")


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def get_instance_paths(instance_name: str) -> dict[str, Path]:
    return {
        "instance_json": PROJECT_ROOT / "data" / "processed" / f"{instance_name}_instance.json",
        "problem_pddl": PROJECT_ROOT / "pddl" / "problems" / f"problem_{instance_name}.pddl",
        "plan_file": PROJECT_ROOT / "outputs" / "plans" / f"{instance_name}_plan.txt",
        "parsed_plan": PROJECT_ROOT / "outputs" / "results" / f"{instance_name}_parsed_plan.json",
        "route_figure": PROJECT_ROOT / "outputs" / "figures" / f"{instance_name}_route_comparison.png",
        "interactive_map": PROJECT_ROOT / "outputs" / "maps" / f"{instance_name}_interactive_route_map.html",
        "sumo_config": PROJECT_ROOT / "outputs" / "sumo" / instance_name / f"{instance_name}.sumocfg",
        "sumo_route": PROJECT_ROOT / "outputs" / "sumo" / instance_name / f"{instance_name}.rou.xml",
        "sumo_network": PROJECT_ROOT / "outputs" / "sumo" / instance_name / f"{instance_name}.net.xml",
        "sumo_open_script": PROJECT_ROOT / "outputs" / "sumo" / instance_name / f"open_{instance_name}_sumo_gui.ps1",
    }


def get_comparison(instance_name: str) -> dict[str, Any] | None:
    comparison_path = PROJECT_ROOT / "outputs" / "results" / "comparison_results.json"
    comparisons = load_json(comparison_path)

    if not comparisons:
        return None

    for item in comparisons:
        if item.get("instance_name") == instance_name:
            return item

    return None


def get_validation(instance_name: str) -> dict[str, Any] | None:
    validation_path = PROJECT_ROOT / "outputs" / "results" / "validation_results.json"
    validations = load_json(validation_path)

    if not validations:
        return None

    for item in validations:
        if item.get("instance_name") == instance_name:
            return item

    return None


def display_status_card(label: str, value: Any, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def render_interactive_map(map_path: Path) -> None:
    if not map_path.exists():
        st.warning(f"Interactive map not found: {map_path}")
        return

    html = map_path.read_text(encoding="utf-8", errors="replace")
    components.html(html, height=650, scrolling=True)


def show_file_download(path: Path, label: str, mime: str = "text/plain") -> None:
    if not path.exists():
        st.caption(f"{label}: not generated")
        return

    st.download_button(
        label=f"Download {label}",
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
    )


def main() -> None:
    st.set_page_config(
        page_title="OSM -> PDDL+ Planning Prototype",
        # page_icon="🗺️",
        layout="wide",
    )

    config = load_config()

    st.title("🗺️ OpenStreetMap-Based PDDL+ Route Planning Prototype")
    st.caption(
        "OSMnx -> NetworkX -> PDDL+ -> ENHSP -> Validation -> Dijkstra -> Folium -> SUMO"
    )

    st.sidebar.header("Controls")

    instance_name = st.sidebar.selectbox(
        "Select planning instance",
        options=["small", "medium", "large"],
        index=0,
    )

    paths = get_instance_paths(instance_name)
    instance = load_json(paths["instance_json"])
    comparison = get_comparison(instance_name)
    validation = get_validation(instance_name)

    if instance is None:
        st.error(
            f"Instance file not found: {paths['instance_json']}\n\n"
            "Run the core pipeline first with `python src/main.py`."
        )
        st.stop()

    st.sidebar.subheader("Generated artifacts")
    st.sidebar.write(f"Instance: `{paths['instance_json'].name}`")
    st.sidebar.write(f"PDDL problem: `{paths['problem_pddl'].name}`")
    st.sidebar.write(f"Plan: `{paths['plan_file'].name}`")

    tabs = st.tabs(
        [
            "Overview",
            "Planner vs Dijkstra",
            "Interactive Map",
            "PDDL+ Files",
            "Plan & Validation",
            "SUMO Simulation",
            "All Results",
        ]
    )

    with tabs[0]:
        st.header("Project Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            display_status_card("Locations", instance["num_locations"])
        with col2:
            display_status_card("Road edges", instance["num_edges"])
        with col3:
            display_status_card("Start", instance["start"])
        with col4:
            display_status_card("Goal", instance["goal"])

        st.subheader("Vehicle configuration")

        vehicle = instance["vehicle"]

        col1, col2, col3 = st.columns(3)

        with col1:
            display_status_card("Initial battery", vehicle["initial_battery"])
        with col2:
            display_status_card("Speed", f"{vehicle['speed_m_per_s']} m/s")
        with col3:
            display_status_card(
                "Battery rate",
                vehicle["battery_consumption_per_meter"],
                "Battery consumed per meter",
            )

        st.subheader("Architecture")

        st.code(
            """
OpenStreetMap
   ↓
OSMnx Map Extractor
   ↓
Graph Processor
   ↓
PDDL+ Generator ─────→ ENHSP Planner ─────→ Plan Validation
   ↓                                           ↓
Dijkstra Baseline ───────────────────────→ Comparison
   ↓                                           ↓
Folium Visualization                    SUMO Simulation
            """.strip(),
            language="text",
        )

    with tabs[1]:
        st.header("Planner vs Dijkstra Baseline")

        if comparison is None:
            st.warning("comparison_results.json not found. Run baseline analysis first.")
        else:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                display_status_card("Planner status", comparison["planner_status"])
            with col2:
                display_status_card("Planner valid", comparison["planner_valid"])
            with col3:
                display_status_card(
                    "Runtime", f"{comparison['planner_runtime_seconds']} s"
                )
            with col4:
                display_status_card(
                    "Distance gap", f"{comparison['distance_gap_percent']}%"
                )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Planner route")
                st.write(f"Moves: `{comparison['planner_num_moves']}`")
                st.write(f"Distance: `{comparison['planner_distance_m']} m`")
                st.write(f"Battery used: `{comparison['planner_battery_used']}`")
                st.write(f"Final battery: `{comparison['planner_final_battery']}`")

            with col2:
                st.subheader("Dijkstra route")
                st.write(f"Moves: `{comparison['dijkstra_num_moves']}`")
                st.write(f"Distance: `{comparison['dijkstra_distance_m']} m`")
                st.write(f"Battery used: `{comparison['dijkstra_battery_used']}`")

            if paths["route_figure"].exists():
                st.subheader("Static route comparison")
                st.image(str(paths["route_figure"]), use_container_width=True)
            else:
                st.warning("Route comparison figure not found.")

    with tabs[2]:
        st.header("Interactive Route Map")

        st.write(
            "Red route = ENHSP planner route. Blue dashed route = Dijkstra route. "
            "For the small instance, they may be identical."
        )

        render_interactive_map(paths["interactive_map"])

    with tabs[3]:
        st.header("Generated PDDL+ Files")

        domain_path = PROJECT_ROOT / "pddl" / "domain.pddl"
        problem_path = paths["problem_pddl"]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Domain file")
            domain_text = load_text(domain_path)
            st.code(domain_text[:12000], language="lisp")
            show_file_download(domain_path, "domain.pddl")

        with col2:
            st.subheader("Problem file")
            problem_text = load_text(problem_path)
            st.code(problem_text[:12000], language="lisp")
            show_file_download(problem_path, problem_path.name)

    with tabs[4]:
        st.header("Plan and Validation")

        plan_text = load_text(paths["plan_file"])

        st.subheader("ENHSP plan")
        if plan_text:
            st.code(plan_text, language="text")
            show_file_download(paths["plan_file"], paths["plan_file"].name)
        else:
            st.warning("Plan file not found.")

        st.subheader("Validation result")

        if validation is None:
            st.warning("Validation result not found.")
        else:
            status = "VALID ✅" if validation["valid"] else "INVALID ❌"
            st.write(f"Status: **{status}**")

            col1, col2, col3 = st.columns(3)

            with col1:
                display_status_card("Total distance", f"{validation['total_distance_m']} m")
            with col2:
                display_status_card("Battery used", validation["battery_used"])
            with col3:
                display_status_card("Final battery", validation["final_battery"])

            if validation.get("errors"):
                st.error("Validation errors:")
                st.write(validation["errors"])

            if validation.get("warnings"):
                st.warning("Validation warnings:")
                st.write(validation["warnings"])

            with st.expander("Route"):
                st.write(" → ".join(validation["route"]))

    with tabs[5]:
        st.header("SUMO Simulation")

        st.write(
            "SUMO is used after PDDL+ planning to simulate the ENHSP-generated route."
        )

        if paths["sumo_config"].exists():
            st.success("SUMO files generated for this instance.")

            st.code(
                f'powershell -ExecutionPolicy Bypass -File "{paths["sumo_open_script"]}"',
                language="powershell",
            )

            st.write("Or manually run:")

            st.code(
                f'& "{config["sumo"]["sumo_gui_path"]}" -c "{paths["sumo_config"]}"',
                language="powershell",
            )

            show_file_download(paths["sumo_config"], paths["sumo_config"].name)
            show_file_download(paths["sumo_route"], paths["sumo_route"].name)
            show_file_download(paths["sumo_network"], paths["sumo_network"].name)
        else:
            st.warning(
                "SUMO files not found. Run `python experiments/run_sumo_simulation.py`."
            )

        sumo_summary = load_json(PROJECT_ROOT / "outputs" / "results" / "sumo_simulation_results.json")

        if sumo_summary:
            st.subheader("SUMO validation summary")
            st.dataframe(pd.DataFrame(sumo_summary), use_container_width=True)

    with tabs[6]:
        st.header("All Result Tables")

        result_files = {
            "Planner results": PROJECT_ROOT / "outputs" / "results" / "planner_results.csv",
            "Validation results": PROJECT_ROOT / "outputs" / "results" / "validation_results.csv",
            "Comparison results": PROJECT_ROOT / "outputs" / "results" / "comparison_results.csv",
        }

        for label, path in result_files.items():
            st.subheader(label)

            df = read_csv_if_exists(path)

            if df.empty:
                st.warning(f"{path.name} not found or empty.")
            else:
                st.dataframe(df, use_container_width=True)
                show_file_download(path, path.name, mime="text/csv")


if __name__ == "__main__":
    main()