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
)
from sumo_traci_runner import run_sumo_traci_simulation

# ---------------------------------------------------------------------------
# Vehicle colour palette
# ---------------------------------------------------------------------------
_PALETTE = [
    {"folium_start": "red",    "folium_goal": "darkred",    "icon": "🔴", "sumo": "255,0,0"},
    {"folium_start": "green",  "folium_goal": "darkgreen",  "icon": "🟢", "sumo": "0,200,0"},
    {"folium_start": "blue",   "folium_goal": "darkblue",   "icon": "🔵", "sumo": "0,100,255"},
    {"folium_start": "orange", "folium_goal": "cadetblue",  "icon": "🟠", "sumo": "255,165,0"},
    {"folium_start": "purple", "folium_goal": "darkpurple", "icon": "🟣", "sumo": "160,32,240"},
]
def _vi(i):  return _PALETTE[i % len(_PALETTE)]["icon"]
def _vfs(i): return _PALETTE[i % len(_PALETTE)]["folium_start"]
def _vfg(i): return _PALETTE[i % len(_PALETTE)]["folium_goal"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        st.error(f"config.yaml not found: {CONFIG_PATH}")
        st.stop()
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _save_config(cfg: dict[str, Any]) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

@st.dialog("⚙️ Settings", width="large")
def _settings_dialog(cfg: dict[str, Any]) -> None:
    st.caption("Changes are saved to `config.yaml` and become the defaults for every run.")

    # --- Map ---
    st.subheader("Map & Extraction")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        place_name = st.text_input("Default location",
            value=cfg["map"].get("place_name", "Rende, Calabria, Italy"))
    with mc2:
        ntype_opts = ["drive", "walk", "bike"]
        network_type = st.selectbox("Routing mode", ntype_opts,
            index=ntype_opts.index(cfg["map"].get("network_type", "drive")))
    with mc3:
        max_nodes = st.number_input("Max graph nodes", 10, 200,
            int(cfg["map"].get("max_nodes_small", 30)), 10)
    with mc4:
        min_chargers = st.number_input("Min charging stations", 0, 10,
            int(cfg.get("planning", {}).get("min_chargers", 2)), 1)

    st.divider()

    # --- Vehicle ---
    st.subheader("Vehicle")
    vc1, vc2, vc3, vc4, vc5, vc6 = st.columns(6)
    with vc1:
        v_name = st.text_input("Name", value=cfg["vehicle"].get("name", "car1"))
    with vc2:
        v_battery = st.number_input("Initial battery", 1.0, 10000.0,
            float(cfg["vehicle"].get("initial_battery", 100.0)), 5.0)
    with vc3:
        v_max_bat = st.number_input("Max battery", 1.0, 10000.0,
            float(cfg["vehicle"].get("max_battery", cfg["vehicle"].get("initial_battery", 100.0))), 5.0)
    with vc4:
        v_speed = st.number_input("Speed (m/s)", 0.5, 80.0,
            float(cfg["vehicle"].get("speed_m_per_s", 13.9)), 0.5)
    with vc5:
        v_cons = st.number_input("Consumption/m", 0.001, 1.0,
            float(cfg["vehicle"].get("battery_consumption_per_meter", 0.10)),
            0.001, format="%.3f")
    with vc6:
        v_charge = st.number_input("Charge rate (u/s)", 0.1, 50.0,
            float(cfg["vehicle"].get("charge_rate", 5.0)), 0.5)

    st.divider()

    # --- Signal Timing ---
    st.subheader("Traffic Signal Timing")
    sc1, sc2, sc3 = st.columns(3)
    _pcfg = cfg.get("planning", {})
    with sc1:
        sig_green  = st.number_input("Green (s)",  5, 300, int(_pcfg.get("signal_green_duration",  45)), 5)
    with sc2:
        sig_yellow = st.number_input("Yellow (s)", 1,  30, int(_pcfg.get("signal_yellow_duration",  5)), 1)
    with sc3:
        sig_red    = st.number_input("Red (s)",    5, 300, int(_pcfg.get("signal_red_duration",    30)), 5)
    st.caption(f"Cycle: {sig_green + sig_yellow + sig_red}s "
               f"(green {sig_green}s + yellow {sig_yellow}s + red {sig_red}s)")

    st.divider()

    # --- Planning ---
    st.subheader("Planning")
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        metric = st.selectbox("Metric", ["distance", "time"],
            index=["distance", "time"].index(_pcfg.get("metric", "distance")))
    with pc2:
        st_cap = st.number_input("Station capacity", 1, 10,
            int(_pcfg.get("station_capacity", 2)), 1)
    with pc3:
        timeout = st.number_input("Planner timeout (s)", 10, 600,
            int(cfg["planner"].get("timeout_seconds", 120)), 10)

    st.divider()

    # --- SUMO ---
    st.subheader("SUMO Simulation")
    su1, su2, su3 = st.columns(3)
    with su1:
        sumo_path = st.text_input("SUMO path", value=cfg["sumo"].get("sumo_path", "sumo"))
    with su2:
        sumo_gui  = st.text_input("SUMO-GUI path", value=cfg["sumo"].get("sumo_gui_path", "sumo-gui"))
    with su3:
        netconv   = st.text_input("netconvert path", value=cfg["sumo"].get("netconvert_path", "netconvert"))
    ss1, ss2, ss3 = st.columns(3)
    with ss1:
        bg_vehs  = st.number_input("Background vehicles", 0, 300,
            int(cfg["sumo"].get("background_vehicles", {}).get("small", 20)), 5)
    with ss2:
        sim_end  = st.number_input("Simulation end time (s)", 100, 99_999_999,
            int(cfg["sumo"].get("simulation_end_time", 3000)), 100)
    with ss3:
        n_lanes  = st.number_input("Lanes per road", 1, 4,
            int(cfg["sumo"].get("num_lanes", 1)), 1)

    st.divider()

    if st.button("💾  Save Settings", type="primary", use_container_width=True):
        cfg["map"]["place_name"]     = place_name
        cfg["map"]["network_type"]   = network_type
        cfg["map"]["max_nodes_small"] = int(max_nodes)
        cfg["vehicle"]["name"]        = v_name
        cfg["vehicle"]["initial_battery"] = float(v_battery)
        cfg["vehicle"]["max_battery"]     = float(v_max_bat)
        cfg["vehicle"]["speed_m_per_s"]   = float(v_speed)
        cfg["vehicle"]["battery_consumption_per_meter"] = float(v_cons)
        cfg["vehicle"]["charge_rate"] = float(v_charge)
        cfg.setdefault("planning", {})
        cfg["planning"]["metric"]                 = metric
        cfg["planning"]["signal_green_duration"]  = int(sig_green)
        cfg["planning"]["signal_yellow_duration"] = int(sig_yellow)
        cfg["planning"]["signal_red_duration"]    = int(sig_red)
        cfg["planning"]["station_capacity"]       = int(st_cap)
        cfg["planning"]["min_chargers"]           = int(min_chargers)
        cfg["planner"]["timeout_seconds"]         = int(timeout)
        cfg["sumo"]["sumo_path"]          = sumo_path
        cfg["sumo"]["sumo_gui_path"]      = sumo_gui
        cfg["sumo"]["netconvert_path"]    = netconv
        cfg["sumo"]["simulation_end_time"]= int(sim_end)
        cfg["sumo"]["num_lanes"]          = int(n_lanes)
        cfg["sumo"].setdefault("background_vehicles", {})["small"] = int(bg_vehs)
        _save_config(cfg)
        st.success("Settings saved. Reload the app to apply changes.")
        st.balloons()

def read_text(path: str | Path) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

def render_html(path: str | Path, height: int = 600) -> None:
    p = Path(path)
    if p.exists():
        components.html(
            p.read_text(encoding="utf-8", errors="replace"),
            height=height,
            scrolling=True,
        )

def find_map_file(result: dict[str, Any], config: dict[str, Any]) -> str | None:
    """Return map file path — checks stored path first then fallback by name."""
    stored = result.get("interactive_map_path")
    if stored and Path(stored).exists():
        return stored
    if result.get("instance"):
        inst_name = result["instance"].get("instance_name", "custom")
        maps_dir = Path(config["outputs"].get("maps_dir", "outputs/maps"))
        fallback = maps_dir / f"{inst_name}_interactive_route_map.html"
        if fallback.exists():
            return str(fallback)
    return None

@st.cache_data(show_spinner=False)
def geocode_center(place: str) -> tuple[float, float]:
    try:
        lat, lon = ox.geocode(place)
        return float(lat), float(lon)
    except Exception:
        return 39.3500, 16.2250

def last_drawn_geometry(draw_data: dict | None) -> dict | None:
    if not draw_data:
        return None
    drawings = draw_data.get("all_drawings")
    if not drawings:
        return None
    last = drawings[-1]
    if not last:
        return None
    geom = last.get("geometry")
    return geom if geom and geom.get("type") in ["Polygon", "MultiPolygon"] else None

def area_map(lat: float, lon: float) -> folium.Map:
    fmap = folium.Map(location=[lat, lon], zoom_start=14,
                      tiles="OpenStreetMap", control_scale=True)
    Draw(export=False,
         draw_options={"polyline": False, "circle": False, "circlemarker": False,
                       "marker": False, "polygon": True, "rectangle": True},
         edit_options={"edit": True, "remove": True}).add_to(fmap)
    return fmap

def node_map(instance: dict[str, Any], n_vehicles: int) -> folium.Map:
    lats = [float(l["lat"]) for l in instance["locations"]]
    lons = [float(l["lon"]) for l in instance["locations"]]
    fmap = folium.Map(location=[sum(lats)/len(lats), sum(lons)/len(lons)],
                      zoom_start=15, tiles="OpenStreetMap", control_scale=True)
    coords = {l["id"]: (float(l["lat"]), float(l["lon"])) for l in instance["locations"]}

    for edge in instance["edges"]:
        s, t = edge["from"], edge["to"]
        if s in coords and t in coords:
            folium.PolyLine([coords[s], coords[t]], color="#aaaaaa",
                            weight=2, opacity=0.5, tooltip=f"{s}→{t}").add_to(fmap)

    last_clicked = st.session_state.get("last_clicked_node")
    starts: dict[str, int] = {}
    goals:  dict[str, int] = {}
    if n_vehicles == 1:
        s = st.session_state.get("selected_start")
        g = st.session_state.get("selected_goal")
        if s: starts[s] = 0
        if g: goals[g]  = 0
    else:
        for i in range(1, n_vehicles + 1):
            s = st.session_state.get(f"mv_start_{i}")
            g = st.session_state.get(f"mv_goal_{i}")
            if s: starts[s] = i - 1
            if g: goals[g]  = i - 1

    for loc in instance["locations"]:
        lid = loc["id"]
        lat, lon = float(loc["lat"]), float(loc["lon"])
        tip = lid
        if loc.get("has_traffic_signal"):   tip += " 🚦 traffic signal"
        if loc.get("has_charging_station"): tip += " ⚡ charging station"

        if lid == last_clicked:
            folium.CircleMarker([lat, lon], radius=10, color="gold",
                                fill=True, fill_color="yellow", fill_opacity=1.0,
                                weight=3, tooltip=tip + " ← selected").add_to(fmap)
        elif lid in starts:
            folium.CircleMarker([lat, lon], radius=9, color=_vfs(starts[lid]),
                                fill=True, fill_opacity=0.9,
                                tooltip=tip + f" [{_vi(starts[lid])} START]").add_to(fmap)
        elif lid in goals:
            folium.CircleMarker([lat, lon], radius=9, color=_vfg(goals[lid]),
                                fill=True, fill_opacity=0.25, weight=3,
                                tooltip=tip + f" [{_vi(goals[lid])} GOAL]").add_to(fmap)
        elif loc.get("has_traffic_signal"):
            # Traffic signal: large orange filled circle with black border.
            # Orange = universally associated with traffic warning signals.
            # CircleMarker is more reliable than DivIcon across all browsers.
            folium.CircleMarker(
                [lat, lon],
                radius=11,
                color="#000000",
                weight=2,
                fill=True,
                fill_color="#e67e22",
                fill_opacity=1.0,
                tooltip=tip,
                popup=folium.Popup(
                    f"<b>&#128680; Traffic Signal</b><br><small>{lid}</small>",
                    max_width=180,
                ),
            ).add_to(fmap)
        elif loc.get("has_charging_station"):
            # Charging station: large teal square (via polygon).
            # Square shape is immediately distinct from round traffic signal.
            folium.CircleMarker(
                [lat, lon],
                radius=11,
                color="#000000",
                weight=2,
                fill=True,
                fill_color="#00b894",
                fill_opacity=1.0,
                tooltip=tip,
                popup=folium.Popup(
                    f"<b>&#9889; Charging Station</b><br><small>{lid}</small>",
                    max_width=180,
                ),
            ).add_to(fmap)
        else:
            folium.CircleMarker(
                [lat, lon], radius=5,
                color="#2471a3", weight=1.5,
                fill=True, fill_color="#5dade2", fill_opacity=0.7,
                tooltip=tip,
            ).add_to(fmap)
    return fmap


# ---------------------------------------------------------------------------
# Result sections (shown inline after run + in Results tab)
# ---------------------------------------------------------------------------

def section_single_metrics(result: dict[str, Any]) -> None:
    v   = result["validation"]
    pr  = result["planner_result"]
    cmp = result["comparison"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distance", f"{v['total_distance_m']} m")
    c2.metric("Final battery", v["final_battery"])
    c3.metric("vs Dijkstra", f"{cmp.get('distance_gap_percent')}%" if cmp.get("distance_gap_percent") is not None else "—")
    c4.metric("Runtime", f"{pr['runtime_seconds']} s")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Moves", v["num_move_actions"])
    c2.metric("Dijkstra distance", f"{cmp['dijkstra_distance_m']} m")
    c3.metric("Same route", "Yes" if cmp.get("same_route_as_dijkstra") else "No")
    c4.metric("Plan valid", "Yes" if v["valid"] else "No")

    if v.get("route"):
        st.caption("Route: " + "  →  ".join(v["route"]))

    if v.get("errors"):
        st.error("Validation errors: " + str(v["errors"]))


def section_multi_metrics(result: dict[str, Any]) -> None:
    pvr       = result.get("per_vehicle_results", [])
    all_valid = result.get("all_routes_valid", False)
    pr        = result["planner_result"]
    n_planned = result.get("num_planned", len(pvr))
    n_excluded = result.get("num_excluded", 0)
    any_repaired = result.get("any_repaired", False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status",         pr.get("status", "—"))
    c2.metric("Runtime",        f"{pr.get('runtime_seconds', 0)} s")
    c3.metric("Total distance", f"{result.get('total_distance_m', 0)} m")
    c4.metric("All valid",      "Yes" if all_valid else "No")

    if any_repaired or n_excluded:
        parts = []
        if any_repaired:
            parts.append("auto-repairs applied to one or more vehicles")
        if n_excluded:
            parts.append(f"{n_excluded} vehicle(s) excluded (no feasible path)")
        st.info("🔧 " + "; ".join(parts).capitalize() + ".")

    st.write("")

    for idx, vr in enumerate(pvr):
        icon     = _vi(idx)
        vid      = vr["vehicle_id"]
        diag     = vr.get("diagnosis", {})
        valid    = vr["route_valid"]
        excluded = vr.get("excluded", False)
        repaired = vr.get("repaired", False)

        # Route header — show original → repaired nodes when they changed
        orig_start = vr.get("original_start", vr["start"])
        orig_goal  = vr.get("original_goal",  vr["goal"])
        route_str  = f"{orig_start} → {orig_goal}"
        if vr["start"] != orig_start or vr["goal"] != orig_goal:
            route_str += f" *(→ {vr['start']} → {vr['goal']})*"

        if excluded:
            badge = "🚫 excluded"
        elif repaired and valid:
            badge = "🔧 ✅"
        elif repaired and not valid:
            badge = "🔧 ❌"
        elif valid:
            badge = "✅"
        elif not diag.get("path_exists", True):
            badge = "❌ no path"
        elif diag.get("battery_sufficient") is False:
            badge = f"🪫 need +{diag.get('battery_shortfall', '?')}"
        else:
            badge = "❌ no route"

        with st.container(border=True):
            st.markdown(f"{icon} **{vid}** &nbsp; {route_str} &nbsp; {badge}")

            # Show repair details when applicable
            repairs = vr.get("repairs_applied", [])
            if repairs:
                for r in repairs:
                    st.caption(f"🔧 {r}")

            if excluded:
                reason = diag.get("reason", "No feasible path could be constructed.")
                st.error(f"Excluded: {reason}")
            elif valid:
                charge_stops = vr.get('charge_stops', [])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('Distance', f"{vr['planner_distance_m']} m")
                c2.metric('Battery used', vr['planner_battery_used'])
                c3.metric('Final battery', vr['final_battery'])
                c4.metric(
                    'vs Dijkstra',
                    f"{vr.get('distance_gap_percent')}%"
                    if vr.get('distance_gap_percent') is not None else '—'
                )
                st.caption('Route: ' + '  →  '.join(vr['route']))
                if charge_stops:
                    for cs in charge_stops:
                        loc = cs.get('location', '?')
                        arr = cs.get('battery_on_arrival', '?')
                        aft = cs.get('battery_after', '?')
                        st.info(
                            f'⚡ Charging stop at `{loc}`: '
                            f'arrived with **{arr}** units, '
                            f'charged to **{aft}** (full).'
                        )
            else:
                if not diag.get("path_exists", True):
                    st.error(
                        f"No path in the graph from `{vr['start']}` to `{vr['goal']}`. "
                        "Choose different nodes."
                    )
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Shortest path",    f"{diag.get('dijkstra_distance_m')} m")
                    c2.metric("Battery available", diag.get("battery_available"))
                    c3.metric("Battery required",  diag.get("battery_required"))
                    if diag.get("battery_sufficient") is False:
                        st.warning(
                            f"Increase battery by **{diag.get('battery_shortfall')}** units "
                            "or choose a closer goal."
                        )
                    elif diag.get("battery_sufficient"):
                        st.info("Battery sufficient. Joint plan may have failed due to routing conflicts.")
                    dr = diag.get("dijkstra_route", [])
                    if dr:
                        st.caption("Dijkstra route: " + "  →  ".join(dr))


def show_inline_result(result: dict[str, Any]) -> None:
    """Shown immediately after clicking Run, inside the Planning tab."""
    if result.get("is_multi_vehicle"):
        all_valid    = result.get("all_routes_valid", False)
        n_planned    = result.get("num_planned", 0)
        n_excluded   = result.get("num_excluded", 0)
        any_repaired = result.get("any_repaired", False)
        n_total      = result.get("num_vehicles", len(result.get("per_vehicle_results", [])))

        if all_valid and not n_excluded:
            msg = f"✅ Joint plan found — all {n_total} vehicles reached their goals."
            if any_repaired:
                msg += " (auto-repairs were applied before planning)"
            st.success(msg)
        elif all_valid and n_excluded:
            st.warning(
                f"⚠️ Joint plan found for {n_planned}/{n_total} vehicles. "
                f"{n_excluded} vehicle(s) excluded — see diagnosis below."
            )
        elif n_planned == 0:
            st.error("🚫 All vehicles excluded — no feasible routes could be constructed.")
        else:
            st.error(
                f"❌ Joint plan failed for {n_planned} planned vehicle(s). "
                "See diagnosis below."
            )
        section_multi_metrics(result)
    else:
        v = result["validation"]
        if v["valid"]:
            st.success("✅ Valid plan found.")
        else:
            st.error("❌ Planner did not produce a valid plan.")
        section_single_metrics(result)


def open_sumo(gui_path: str, cfg_file: str | Path) -> None:
    p = Path(cfg_file)
    subprocess.Popen([gui_path, "-c", p.name], cwd=p.parent)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="PDDL+ Urban Navigation Planning", layout="wide")
    config = load_config()

    _hcol_title, _hcol_btn = st.columns([9, 1])
    with _hcol_title:
        st.title("PDDL+ Urban EV Fleet Planner")
        st.caption("OpenStreetMap extraction · PDDL+ encoding · ENHSP planning · SUMO simulation")
    with _hcol_btn:
        st.write("")   # vertical nudge
        st.write("")
        if st.button("⚙️", help="Open Settings", use_container_width=True):
            _settings_dialog(config)

    for key, val in {
        "base_instance": None, "planning_result": None,
        "drawn_geometry": None, "selected_start": None,
        "selected_goal": None, "last_clicked_node": None,
        "custom_sumo_result": None,
        "custom_traci_report": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ----------------------------------------------------------------
    # Sidebar — map extraction only
    # ----------------------------------------------------------------
    # Read defaults from config (set in Settings page)
    _place_name   = config["map"].get("place_name", "Rende, Calabria, Italy")
    _network_type = config["map"].get("network_type", "drive")
    _max_nodes    = int(config["map"].get("max_nodes_small", 30))
    _speed        = float(config["vehicle"].get("speed_m_per_s", 13.9))
    _min_chargers = int(config.get("planning", {}).get("min_chargers", 2))

    with st.sidebar:
        st.header("Map Extraction")

        place_name   = st.text_input("Location", value=_place_name)
        network_type = st.selectbox(
            "Routing mode", ["drive", "walk", "bike"],
            index=["drive", "walk", "bike"].index(_network_type),
        )
        max_nodes = st.slider("Max graph nodes", 10, 150, _max_nodes, 10)

        if st.button("Extract searched place", use_container_width=True):
            clear_custom_outputs(config)
            for k in ["planning_result", "selected_start", "selected_goal",
                      "last_clicked_node", "custom_sumo_result"]:
                st.session_state[k] = None
            with st.spinner("Extracting..."):
                try:
                    inst = prepare_custom_map(
                        place_name=place_name, network_type=network_type,
                        max_nodes=max_nodes, vehicle_speed_m_per_s=_speed,
                        config=config, min_chargers=_min_chargers,
                    )
                    st.session_state.base_instance = inst
                    st.success(
                        f"{inst['num_locations']} locations · "
                        f"{inst['num_edges']} edges · "
                        f"{inst.get('num_traffic_signals', 0)} 🚦 · "
                        f"{inst.get('num_charging_stations', 0)} ⚡"
                    )
                except Exception as exc:
                    st.error(str(exc))

    # ----------------------------------------------------------------
    # 4 tabs
    # ----------------------------------------------------------------
    tab_area, tab_plan, tab_files, tab_results = st.tabs([
        "📍 Area", "⚙️ Planning", "📄 PDDL+ Files", "📊 Results"
    ])

    # ================================================================
    # Tab 1 — Area selection
    # ================================================================
    with tab_area:
        st.subheader("Draw Map Area")
        st.caption("Draw a rectangle or polygon on the map, then click Extract.")

        clat, clon = geocode_center(place_name)
        draw_data = st_folium(area_map(clat, clon), height=560,
                              use_container_width=True, key="area_map")

        geom = last_drawn_geometry(draw_data)
        if geom:
            st.session_state.drawn_geometry = geom
            st.success("Area drawn.")

        if st.button("Extract graph from drawn area", type="primary", use_container_width=True):
            if not st.session_state.drawn_geometry:
                st.error("Draw an area on the map first.")
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
                            max_nodes=max_nodes, vehicle_speed_m_per_s=_speed,
                            config=config, min_chargers=_min_chargers,
                        )
                        st.session_state.base_instance = inst
                        st.success(
                            f"{inst['num_locations']} locations · "
                            f"{inst['num_edges']} edges · "
                            f"{inst.get('num_traffic_signals', 0)} 🚦 · "
                            f"{inst.get('num_charging_stations', 0)} ⚡"
                        )
                    except Exception as exc:
                        st.error(str(exc))

    base_instance = st.session_state.base_instance

    # ================================================================
    # Tab 2 — Planning configuration
    # ================================================================
    with tab_plan:
        if base_instance is None:
            st.info("Extract a map area first (Area tab or sidebar).")
        else:
            # Stats bar
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Locations", base_instance["num_locations"])
            c2.metric("Edges",     base_instance["num_edges"])
            c3.metric("Mode",      base_instance["network_type"])
            c4.metric("🚦",        base_instance.get("num_traffic_signals", 0))
            c5.metric("⚡",        base_instance.get("num_charging_stations", 0))

            st.write("")

            # ---- Constraints block ----
            with st.container(border=True):
                st.subheader("Constraints")

                n_vehicles = st.number_input(
                    "Number of vehicles", 1, 5, 1, 1,
                    help="1 = single vehicle · 2–5 = multi-vehicle joint planning"
                )
                is_multi = n_vehicles > 1

                if is_multi:
                    legend = "  ".join(f"{_vi(i)} car{i+1}" for i in range(n_vehicles))
                    st.caption(f"{legend} · filled = START · outlined = GOAL · 🟡 = selected node")

                # Vehicle defaults from Settings (config.yaml)
                default_battery = float(config["vehicle"].get("initial_battery", 100.0))
                speed_m_per_s   = float(config["vehicle"].get("speed_m_per_s", 13.9))
                consumption     = float(config["vehicle"].get("battery_consumption_per_meter", 0.10))

                _cfg_metric = config.get("planning", {}).get("metric", "distance")
                metric = st.radio(
                    "Metric", ["distance", "time"], horizontal=True,
                    index=["distance", "time"].index(_cfg_metric),
                )

                use_low_battery = st.checkbox("Low battery scenario")
                if use_low_battery:
                    col_bat, col_max = st.columns(2)
                    with col_bat:
                        initial_battery = st.slider(
                            "Initial battery", 5.0, float(default_battery),
                            min(float(default_battery), 40.0), 5.0,
                            help="Battery available at departure."
                        )
                    with col_max:
                        max_battery_cap = st.number_input(
                            "Max battery capacity", 1.0, 500.0,
                            float(default_battery), 5.0,
                            help="Full-tank capacity — the vehicle charges up to this value at a charging station.",
                        )
                else:
                    initial_battery = float(default_battery)
                    max_battery_cap = float(default_battery)

                selected_blocked = st.multiselect(
                    "Blocked segments",
                    options=list_edge_options(base_instance),
                    help="Models planned road closures known before departure."
                )
                selected_congested = st.multiselect(
                    "Congested segments",
                    options=list_edge_options(base_instance),
                    help="Inflates road distance · planner may reroute."
                )
                congestion_factor = 0.3
                if selected_congested:
                    congestion_factor = st.slider("Congestion factor", 0.1, 0.9, 0.3, 0.1,
                                                  help="0.1 = severe · 0.9 = mild")

                # Signal timing and planning params from Settings (config.yaml)
                _pcfg = config.get("planning", {})
                signal_red_duration    = int(_pcfg.get("signal_red_duration",   30))
                signal_yellow_duration = int(_pcfg.get("signal_yellow_duration",  5))
                signal_green_duration  = int(_pcfg.get("signal_green_duration",  45))
                station_capacity       = int(_pcfg.get("station_capacity",        2))

                n_signals  = base_instance.get("num_traffic_signals",  0)
                n_chargers = base_instance.get("num_charging_stations", 0)
                if n_signals > 0:
                    st.caption(
                        f"🚦 {n_signals} signal(s) · "
                        f"green {signal_green_duration}s · yellow {signal_yellow_duration}s · "
                        f"red {signal_red_duration}s  ·  change in ⚙️ Settings"
                    )
                if n_chargers > 0:
                    st.caption(
                        f"⚡ {n_chargers} charging station(s) · "
                        f"capacity {station_capacity}  ·  change in ⚙️ Settings"
                    )

            blocked_edges   = build_blocked_edge_objects(selected_blocked, base_instance)
            congested_edges = build_congested_edge_objects(
                selected_congested, base_instance, congestion_factor
            )

            st.write("")

            # ---- Start / Goal block ----
            with st.container(border=True):
                st.subheader("Start and Goal States")

                click_data = st_folium(node_map(base_instance, n_vehicles),
                                       height=460, use_container_width=True, key="node_map")

                # Map legend
                legend_items = [
                    ("🔵 small circle", "Road intersection (selectable)"),
                    ("🟠 large orange circle", "Traffic signal intersection"),
                    ("🟢 large teal circle", "EV charging station"),
                    ("🟡 yellow circle", "Last clicked node"),
                ]
                if n_vehicles == 1:
                    legend_items += [
                        ("🔴 filled circle", "Start node"),
                        ("⭕ outlined circle", "Goal node"),
                    ]
                else:
                    for i in range(n_vehicles):
                        legend_items.append((f"{_vi(i)} filled", f"car{i+1} Start"))
                        legend_items.append((f"{_vi(i)} outlined", f"car{i+1} Goal"))
                with st.expander("Map legend", expanded=False):
                    for symbol, meaning in legend_items:
                        st.caption(f"{symbol} — {meaning}")

                clicked = click_data.get("last_clicked") if click_data else None
                if clicked:
                    st.session_state.last_clicked_node = find_nearest_location_id(
                        base_instance, clicked["lat"], clicked["lng"]
                    )

                last_clicked = st.session_state.last_clicked_node
                if last_clicked:
                    st.info(f"Selected node: **`{last_clicked}`**")

                loc_opts = list_location_options(base_instance)
                loc_ids  = [o.split("|")[0].strip() for o in loc_opts]
                n_locs   = len(loc_ids)

                # Single vehicle
                if not is_multi:
                    if last_clicked:
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("Set as START", use_container_width=True):
                                st.session_state.selected_start = last_clicked
                                st.rerun()
                        with c2:
                            if st.button("Set as GOAL", use_container_width=True):
                                st.session_state.selected_goal = last_clicked
                                st.rerun()

                    si = loc_ids.index(st.session_state.selected_start) \
                         if st.session_state.selected_start in loc_ids else 0
                    gi = loc_ids.index(st.session_state.selected_goal) \
                         if st.session_state.selected_goal in loc_ids else n_locs - 1

                    c1, c2 = st.columns(2)
                    with c1:
                        start_label = st.selectbox("Start", loc_opts, index=si)
                    with c2:
                        goal_label  = st.selectbox("Goal",  loc_opts, index=gi)

                    start_location = extract_location_id(start_label)
                    goal_location  = extract_location_id(goal_label)
                    st.session_state.selected_start = start_location
                    st.session_state.selected_goal  = goal_location

                    if start_location == goal_location:
                        st.warning("Start and goal are the same location.")

                # Multi-vehicle
                vehicles: list[dict[str, Any]] = []
                if is_multi:
                    st.caption("Shared starts and goals are allowed.")
                    for i in range(1, n_vehicles + 1):
                        vid  = f"car{i}"
                        icon = _vi(i - 1)
                        with st.container(border=True):
                            st.markdown(f"{icon} **{vid}**")

                            if last_clicked:
                                c1, c2 = st.columns(2)
                                with c1:
                                    if st.button(f"Set `{last_clicked}` as {vid} START",
                                                 key=f"btn_s_{i}", use_container_width=True):
                                        st.session_state[f"mv_start_{i}"] = last_clicked
                                        st.rerun()
                                with c2:
                                    if st.button(f"Set `{last_clicked}` as {vid} GOAL",
                                                 key=f"btn_g_{i}", use_container_width=True):
                                        st.session_state[f"mv_goal_{i}"] = last_clicked
                                        st.rerun()

                            dflt_s = 0
                            dflt_g = min(int((i * n_locs) / (n_vehicles + 1)), n_locs - 1)
                            if dflt_s == dflt_g:
                                dflt_g = min(dflt_g + 1, n_locs - 1)

                            sv_s = st.session_state.get(f"mv_start_{i}")
                            sv_g = st.session_state.get(f"mv_goal_{i}")
                            si = loc_ids.index(sv_s) if sv_s in loc_ids else dflt_s
                            gi = loc_ids.index(sv_g) if sv_g in loc_ids else dflt_g

                            c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
                            with c1:
                                sl = st.selectbox(f"{icon} Start", loc_opts, index=si, key=f"sel_s_{i}")
                            with c2:
                                gl = st.selectbox(f"{icon} Goal",  loc_opts, index=gi, key=f"sel_g_{i}")
                            with c3:
                                v_battery = st.number_input(
                                    "Battery", 1.0, 500.0,
                                    float(st.session_state.get(f"mv_battery_{i}", initial_battery)),
                                    5.0, key=f"mv_battery_{i}"
                                )
                            with c4:
                                v_priority = st.checkbox(
                                    "Priority", key=f"mv_priority_{i}",
                                    help="Priority vehicles bypass red/yellow signals and occupied roads. "
                                         "Approaching a signal forces it green (emergency preemption).",
                                )

                            vehicles.append({
                                "id": vid,
                                "start": extract_location_id(sl),
                                "goal":  extract_location_id(gl),
                                "battery": float(st.session_state.get(f"mv_battery_{i}", initial_battery)),
                                "speed_m_per_s": float(speed_m_per_s),
                                "battery_consumption_per_meter": float(consumption),
                                "max_battery": float(default_battery),
                                "charge_rate": 5.0,
                                "priority": bool(v_priority),
                            })

            st.write("")

            # ---- Run button ----
            btn = f"Run ENHSP — {n_vehicles} vehicle{'s' if n_vehicles > 1 else ''} · {metric}"
            if st.button(btn, type="primary", use_container_width=True):
                st.session_state.custom_sumo_result = None
                st.session_state.custom_traci_report = None
                if is_multi:
                    with st.spinner("Running ENHSP..."):
                        try:
                            result = run_multi_vehicle_planning_job(
                                base_instance=base_instance, vehicles=vehicles,
                                metric=metric, blocked_edges=blocked_edges,
                                congested_edges=congested_edges, config=config,
                                signal_red_duration=signal_red_duration,
                                signal_green_duration=signal_green_duration,
                                signal_yellow_duration=signal_yellow_duration,
                                station_capacity=station_capacity,
                            )
                            st.session_state.planning_result = result
                            show_inline_result(result)
                        except Exception as exc:
                            st.error(f"Planning failed: {exc}")
                else:
                    if start_location == goal_location:
                        st.error("Start and goal must differ.")
                    else:
                        with st.spinner("Running ENHSP..."):
                            try:
                                instance, custom_config = apply_user_choices_to_instance(
                                    base_instance=base_instance,
                                    start=start_location, goal=goal_location,
                                    initial_battery=initial_battery,
                                    max_battery=max_battery_cap,
                                    speed_m_per_s=speed_m_per_s,
                                    battery_consumption_per_meter=consumption,
                                    metric=metric, blocked_edges=blocked_edges,
                                    congested_edges=congested_edges, config=config,
                                    signal_red_duration=signal_red_duration,
                                    signal_green_duration=signal_green_duration,
                                    signal_yellow_duration=signal_yellow_duration,
                                    congestion_sensitivity=congestion_sensitivity,
                                    station_capacity=station_capacity,
                                )
                                result = run_custom_planning_job(
                                    instance=instance, config=custom_config, metric=metric,
                                )
                                st.session_state.planning_result = result
                                show_inline_result(result)
                            except Exception as exc:
                                st.error(f"Planning failed: {exc}")

    result = st.session_state.planning_result

    # ================================================================
    # Tab 3 — Results (map + metrics + SUMO in one place)
    # ================================================================
    with tab_results:
        if not result:
            st.info("Run the planner from the Planning tab first.")
        else:
            is_multi  = result.get("is_multi_vehicle", False)
            all_valid = result.get("all_routes_valid", False) if is_multi else result["validation"]["valid"]
            plan_found = result["planner_result"].get("plan_found", False)

            # ---- Route map (top of results) ----
            map_path = find_map_file(result, config)
            if map_path:
                if is_multi:
                    n_v = result.get("num_vehicles", 1)
                    legend = "  ".join(f"{_vi(i)} car{i+1}" for i in range(n_v))
                    st.caption(f"{legend} · solid = ENHSP · dashed = Dijkstra (toggle layers)")
                render_html(map_path, height=480)
            else:
                st.caption("Route map unavailable.")

            st.divider()

            # ---- Metrics ----
            if is_multi:
                section_multi_metrics(result)
            else:
                section_single_metrics(result)

            st.divider()

            # ---- SUMO section ----
            if all_valid and plan_found:
                st.divider()
                st.subheader("🚗 SUMO Traffic Simulation")

                if is_multi:
                    n_v = result.get("num_vehicles", 1)
                    legend = "  ".join(f"{_vi(i)} car{i+1}" for i in range(n_v))
                    st.caption(f"Planned vehicles: {legend}  ·  🔵 Background traffic")
                else:
                    st.caption("🔴 Planned vehicle (ENHSP route)  ·  🔵 Background traffic")

                if st.button("▶ Open SUMO-GUI", type="primary",
                             use_container_width=True, key="sumo_open"):
                    try:
                        with st.spinner("SUMO-GUI running — close the window to see the execution report..."):
                            traci_report = run_sumo_traci_simulation(
                                result=result,
                                instance=result["instance"],
                                config=config,
                                gui=True,
                            )
                        st.session_state.custom_traci_report = traci_report
                        st.success("SUMO-GUI closed. Execution report ready.")
                    except FileNotFoundError:
                        st.error("SUMO-GUI not found. Check sumo_gui_path in config.yaml.")
                    except Exception as exc:
                        st.error(str(exc))

                    traci_report = st.session_state.get("custom_traci_report")
                    if traci_report:
                        with st.expander("Plan vs Execution Report", expanded=True):
                            for vid, vr in traci_report["vehicle_reports"].items():
                                st.markdown(f"**{vid}** — departed t={vr['depart_time']:.0f} s")
                                rows = [
                                    {
                                        "Node": r["node"],
                                        "Planned (s)": r["planned_t"],
                                        "Actual (s)": r["actual_t"] if r["actual_t"] is not None else "—",
                                        "Δ (s)": f"{r['deviation_s']:+.0f}" if r["deviation_s"] is not None else "—",
                                    }
                                    for r in vr["node_rows"]
                                ]
                                import pandas as pd
                                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                                goal_actual = vr["goal_actual_t"]
                                if goal_actual:
                                    st.caption(f"Goal {vr['goal_node']}: reached t={goal_actual:.0f} s "
                                               f"(planned={vr['goal_planned_t']:.0f} s, "
                                               f"Δ={vr['goal_deviation_s']:+.0f} s)")
                                else:
                                    st.caption(f"Goal {vr['goal_node']}: not reached during simulation.")
                                st.caption(f"Off-route deviations: {vr['num_deviations']}")
                            total = traci_report["total_deviations"]
                            if total == 0:
                                st.success("✓ All vehicles followed their planned routes exactly.")
                            else:
                                st.warning(f"Total off-route deviations: {total}")

                    with st.expander("Output files"):
                        st.code(
                            f"Network  : {sr['network_file']}\n"
                            f"Routes   : {sr['route_file']}\n"
                            f"Tripinfo : {sr.get('tripinfo_file', 'N/A')}\n"
                            f"Edgedata : {sr.get('edgedata_file', 'N/A')}\n"
                            f"Config   : {sr['config_file']}\n"
                            f"SUMO log : {sr['sumo_validation_log']}\n"
                            f"netconv  : {sr['netconvert_log']}",
                            language="text"
                        )

    # ================================================================
    # Tab 4 — Files (PDDL+ domain, problem, plan, log)
    # ================================================================
    with tab_files:
        if not result:
            st.info("Run the planner first.")
        else:
            # PDDL+ files side by side
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Domain")
                st.code(read_text(result["domain_path"])[:15000], language="lisp")
            with c2:
                st.subheader("Problem")
                st.code(read_text(result["problem_path"])[:15000], language="lisp")

            st.divider()

            # Plan and log side by side
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("ENHSP plan")
                plan = read_text(result["planner_result"]["plan_file"])
                st.code(plan if plan.strip() else "No plan found.", language="text")
            with c2:
                st.subheader("Planner log")
                st.code(read_text(result["planner_result"]["log_file"])[:12000],
                        language="text")


if __name__ == "__main__":
    main()