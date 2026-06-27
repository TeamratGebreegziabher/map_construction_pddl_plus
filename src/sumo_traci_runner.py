"""
sumo_traci_runner.py — Step-by-step SUMO plan execution with traci.

Runs the PDDL+ plan inside SUMO-GUI with:
  - Faithful departure times taken from the plan (not hardcoded t=5)
  - Charging stops injected as SUMO `setStop` calls with the correct duration
  - Per-step deviation monitor: expected edge vs actual edge for every vehicle
  - Post-simulation report: planned arrival time vs actual arrival time per node
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


# ── SUMO tools path ──────────────────────────────────────────────────────────

def _ensure_sumo_tools(config: dict) -> None:
    candidates = [
        config.get("sumo", {}).get("tools_path", ""),
        r"C:\Program Files (x86)\Eclipse\Sumo\tools",
        r"C:\Program Files\Eclipse\Sumo\tools",
    ]
    for p in candidates:
        if p and Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)


# ── Edge ID (must match app_pipeline.sumo_edge_id) ───────────────────────────

def _edge_id(src: str, tgt: str) -> str:
    return f"edge__{src}__{tgt}"


# ── Timeline builder ─────────────────────────────────────────────────────────

def build_vehicle_timelines(
    parsed_plan: dict[str, Any],
    per_vehicle_results: list[dict[str, Any]],
) -> dict[str, dict]:
    """
    Convert parsed PDDL+ actions into a per-vehicle execution timeline.

    Timeline keys per vehicle:
      depart_time    — simulation second when the vehicle should first move
      edge_schedule  — list of {t_start, t_end, edge, from, to}
      charges        — list of {location, edge, t_start, t_end, duration}
      arrivals       — {node: planned_arrival_time}
      goal_node      — last node in route
      goal_time      — planned time of goal arrival
    """
    actions = parsed_plan.get("actions", [])

    by_vehicle: dict[str, list] = {}
    for a in actions:
        vid = a.get("vehicle")
        if vid:
            by_vehicle.setdefault(vid, []).append(a)

    route_map = {
        r["vehicle_id"]: r["route"]
        for r in per_vehicle_results
        if not r.get("excluded") and r.get("route")
    }

    timelines: dict[str, dict] = {}

    for vid, acts in by_vehicle.items():
        acts = sorted(acts, key=lambda a: (a.get("time") or 0.0))
        moves         = [a for a in acts if a["action"] == "start-move"]
        charges_raw   = [a for a in acts if a["action"] == "charge"]
        fully_charged = [a for a in acts if a["action"] == "fully-charged"]

        if not moves:
            continue

        # ── Edge schedule ────────────────────────────────────────────────────
        edge_schedule = []
        for i, mv in enumerate(moves):
            t_start = mv.get("time") or 0.0
            t_end   = (moves[i + 1].get("time") or t_start) if i + 1 < len(moves) else t_start + 120
            edge_schedule.append({
                "t_start": t_start,
                "t_end":   t_end,
                "edge":    _edge_id(mv.get("from", ""), mv.get("to", "")),
                "from":    mv.get("from", ""),
                "to":      mv.get("to", ""),
            })

        # Fix last segment's t_end to the last known action time
        all_times = [a.get("time") or 0.0 for a in acts]
        if all_times and edge_schedule:
            edge_schedule[-1]["t_end"] = max(all_times)

        depart_time = edge_schedule[0]["t_start"]
        arrivals    = {seg["to"]: seg["t_end"] for seg in edge_schedule}

        # ── Charging stops ───────────────────────────────────────────────────
        route   = route_map.get(vid, [])
        charges = []
        for ch in charges_raw:
            loc     = ch.get("location", "")
            t_start = ch.get("time") or 0.0
            fc = next(
                (f for f in fully_charged
                 if f.get("vehicle") == vid and (f.get("time") or 0.0) >= t_start),
                None,
            )
            t_end    = (fc.get("time") or (t_start + 1.0)) if fc else t_start + 1.0
            duration = max(1.0, round(t_end - t_start, 1))

            # Edge that enters the charging node (second-to-last → charging node)
            idx      = route.index(loc) if loc in route else -1
            ch_edge  = _edge_id(route[idx - 1], loc) if idx > 0 else ""

            charges.append({
                "location": loc,
                "edge":     ch_edge,
                "t_start":  t_start,
                "t_end":    t_end,
                "duration": duration,
            })

        goal_node = route[-1] if route else (edge_schedule[-1]["to"] if edge_schedule else "")
        goal_time = edge_schedule[-1]["t_end"] if edge_schedule else 0.0

        timelines[vid] = {
            "depart_time":   depart_time,
            "edge_schedule": edge_schedule,
            "charges":       charges,
            "arrivals":      arrivals,
            "goal_node":     goal_node,
            "goal_time":     goal_time,
        }

    return timelines


def _planned_edges(timeline: dict) -> set[str]:
    """Return the set of all SUMO edges that are part of the planned route."""
    return {seg["edge"] for seg in timeline["edge_schedule"]}


# ── Route file patcher ────────────────────────────────────────────────────────

def _patch_depart_times(route_file: Path, timelines: dict[str, dict]) -> None:
    """Rewrite depart= for ENHSP_<vehicle_id> entries using plan timing."""
    tree = ET.parse(route_file)
    root = tree.getroot()
    for v_elem in root.findall("vehicle"):
        vid_attr = v_elem.get("id", "")
        if not vid_attr.startswith("ENHSP_"):
            continue
        vid = vid_attr[len("ENHSP_"):]
        if vid in timelines:
            v_elem.set("depart", str(timelines[vid]["depart_time"]))
    ET.indent(tree, space="  ")
    tree.write(str(route_file), encoding="utf-8", xml_declaration=True)


# ── Colour palette (R, G, B) per vehicle slot ─────────────────────────────────

_PALETTE = [
    (220,  20,  20),   # vehicle_1 — red
    (0,   180,   0),   # vehicle_2 — green
    (30,  100, 240),   # vehicle_3 — blue
    (255, 165,   0),   # vehicle_4 — orange
    (140,  30, 220),   # vehicle_5 — purple
]


# ── Main traci runner ─────────────────────────────────────────────────────────

def run_sumo_traci_simulation(
    result: dict[str, Any],
    instance: dict[str, Any],
    config: dict[str, Any],
    gui: bool = True,
) -> dict[str, Any]:
    """
    Run SUMO via traci with faithful plan timing and charging stops.

    Steps:
      1. Generate SUMO network + route files (reuses existing pipeline)
      2. Patch route file: replace hardcoded depart=5 with plan departure times
      3. Start SUMO-GUI (or headless) via traci
      4. Each step: inject charging stops, detect edge deviations
      5. After simulation: print and return plan-vs-execution report
    """
    _ensure_sumo_tools(config)
    try:
        import traci
    except ImportError:
        raise RuntimeError(
            "traci not found. Add the SUMO tools directory to 'sumo.tools_path' in config.yaml."
        )

    from app_pipeline import generate_custom_sumo_simulation

    parsed_plan        = result.get("parsed_plan", {})
    per_vehicle_results = result.get("per_vehicle_results", [])

    # 1. Build timelines from the PDDL+ plan
    timelines = build_vehicle_timelines(parsed_plan, per_vehicle_results)
    if not timelines:
        print("  No planned vehicle timelines found — nothing to simulate.")
        return {"vehicle_reports": {}, "total_deviations": 0, "deviations": []}

    # 2. Generate SUMO files (netconvert + route XML + sumocfg)
    headless_config = dict(config)
    headless_config["sumo"] = dict(config.get("sumo", {}))
    headless_config["sumo"]["sumo_path"] = "sumo"   # headless for generation step
    print("  Building SUMO network (netconvert + route generation)...")
    sumo_result = generate_custom_sumo_simulation(
        instance=instance,
        comparison=result,
        config=headless_config,
        background_vehicle_count=20,
    )

    # 3. Patch depart times in route file
    route_file = Path(sumo_result["route_file"])
    _patch_depart_times(route_file, timelines)

    # 4. Launch SUMO via traci
    sumo_bin = "sumo-gui" if gui else "sumo"
    sumocfg  = sumo_result["config_file"]
    delay    = "100" if gui else "0"
    print(f"  Launching {'SUMO-GUI' if gui else 'SUMO'} via traci...")

    cmd = [sumo_bin, "-c", sumocfg, "--delay", delay]
    if not gui:
        cmd += ["--quit-on-end", "true"]  # headless only — GUI stays open after sim ends
    else:
        # Extend simulation end to cover the latest vehicle goal time + 5 min buffer
        # so all vehicles depart and arrive before SUMO auto-closes the GUI
        max_goal_t = max((tl["goal_time"] for tl in timelines.values()), default=3600)
        cmd += ["--end", str(int(max_goal_t) + 300)]
    traci.start(cmd)

    # 5. Step-by-step simulation loop
    charge_injected:  set[str]        = set()
    appearance_set:   set[str]        = set()   # vehicles already styled
    deviations:       list[dict]      = []
    arrival_log:      dict[str, dict] = {vid: {} for vid in timelines}
    last_edge:        dict[str, str]  = {}
    vid_order         = {vid: i for i, vid in enumerate(timelines)}
    planned_edge_sets = {vid: _planned_edges(tl) for vid, tl in timelines.items()}

    print()
    print("  ── Live deviation monitor ─────────────────────────────────")

    try:
        while True:
            traci.simulationStep()
            # Headless: stop when all vehicles have arrived
            # GUI: keep stepping so the window stays open; user closes it manually
            if not gui and traci.simulation.getMinExpectedNumber() == 0:
                break
            t = traci.simulation.getTime()

            active = set(traci.vehicle.getIDList())

            for vid, tl in timelines.items():
                sumo_vid = f"ENHSP_{vid}"
                if sumo_vid not in active:
                    continue

                # One-time appearance setup per vehicle
                if sumo_vid not in appearance_set:
                    colour = _PALETTE[vid_order.get(vid, 0) % len(_PALETTE)]
                    traci.vehicle.setColor(sumo_vid, (*colour, 255))
                    traci.vehicle.setLength(sumo_vid, 8.0)  # longer than background (4.5 m)
                    traci.vehicle.setWidth(sumo_vid, 2.8)   # wider than background (1.8 m)
                    try:
                        traci.vehicle.highlight(
                            sumo_vid,
                            color=(*colour, 200),
                            size=12,
                            alphaMax=200,
                            duration=-1,
                            type=0,
                        )
                    except Exception:
                        pass  # highlight not available in older SUMO builds
                    # Camera tracks priority vehicle (index 0)
                    if vid_order.get(vid, 1) == 0:
                        try:
                            traci.gui.trackVehicle("View #0", sumo_vid)
                            traci.gui.setZoom("View #0", 800)
                        except Exception:
                            pass
                    appearance_set.add(sumo_vid)

                # Inject charging stops (once, as soon as vehicle is active)
                if vid not in charge_injected:
                    for ch in tl["charges"]:
                        if not ch["edge"]:
                            continue
                        lane_id = ch["edge"] + "_0"
                        try:
                            lane_len = traci.lane.getLength(lane_id)
                            stop_pos = max(1.0, lane_len - 2.0)
                            traci.vehicle.setStop(
                                sumo_vid,
                                ch["edge"],
                                pos=stop_pos,
                                laneIndex=0,
                                duration=ch["duration"],
                                flags=0,
                            )
                            print(f"    ✓ {vid}: charging stop injected at"
                                  f" {ch['location']} for {ch['duration']:.0f} s"
                                  f"  (plan t={ch['t_start']:.0f}–{ch['t_end']:.0f})")
                        except Exception as e:
                            print(f"    ⚠ {vid}: charging stop failed — {e}")
                    charge_injected.add(vid)

                # Deviation detection — only flag if vehicle is on an edge
                # NOT in its planned route (wrong path, not just early/late)
                current_edge = traci.vehicle.getRoadID(sumo_vid)
                if current_edge.startswith(":"):
                    continue  # skip internal junction edges

                prev = last_edge.get(sumo_vid, "")
                if (current_edge != prev
                        and current_edge not in planned_edge_sets[vid]):
                    deviations.append({
                        "time":        t,
                        "vehicle":     vid,
                        "actual_edge": current_edge,
                    })
                    print(f"    ⚠  t={t:5.0f} s  {vid}"
                          f"  off-route: {current_edge}")

                last_edge[sumo_vid] = current_edge

                # Record first time we see the vehicle leave each segment
                for seg in tl["edge_schedule"]:
                    node = seg["to"]
                    if node in arrival_log[vid]:
                        continue
                    if current_edge != seg["edge"] and t > seg["t_start"] + 1:
                        arrival_log[vid][node] = round(t, 1)

    except Exception as e:
        print(f"  traci error during simulation: {e}")
    finally:
        try:
            traci.close()
        except Exception:
            pass

    # 6. Build and print report
    report = _build_report(timelines, arrival_log, deviations)
    _print_report(report)
    return report


# ── Report ────────────────────────────────────────────────────────────────────

def _build_report(
    timelines: dict[str, dict],
    arrival_log: dict[str, dict],
    deviations: list[dict],
) -> dict[str, Any]:
    vehicle_reports: dict[str, dict] = {}

    for vid, tl in timelines.items():
        rows = []
        for seg in tl["edge_schedule"]:
            node      = seg["to"]
            planned_t = seg["t_end"]
            actual_t  = arrival_log.get(vid, {}).get(node)
            rows.append({
                "node":       node,
                "planned_t":  planned_t,
                "actual_t":   actual_t,
                "deviation_s": round(actual_t - planned_t, 1) if actual_t is not None else None,
            })

        goal_actual = arrival_log.get(vid, {}).get(tl["goal_node"])
        vehicle_reports[vid] = {
            "depart_time":     tl["depart_time"],
            "goal_node":       tl["goal_node"],
            "goal_planned_t":  tl["goal_time"],
            "goal_actual_t":   goal_actual,
            "goal_deviation_s": round(goal_actual - tl["goal_time"], 1)
                                if goal_actual is not None else None,
            "node_rows":       rows,
            "num_deviations":  sum(1 for d in deviations if d["vehicle"] == vid),
        }

    return {
        "vehicle_reports":  vehicle_reports,
        "total_deviations": len(deviations),
        "deviations":       deviations,
    }


def _print_report(report: dict) -> None:
    print()
    print("=" * 64)
    print("  PLAN vs EXECUTION REPORT")
    print("=" * 64)

    for vid, vr in report["vehicle_reports"].items():
        print(f"\n  {vid}   (plan departure t={vr['depart_time']:.0f} s)")
        print(f"  {'Node':<14}  {'Plan (s)':>9}  {'Actual (s)':>10}  {'Δ (s)':>7}")
        print("  " + "─" * 46)
        for row in vr["node_rows"]:
            a_str = f"{row['actual_t']:.0f}"  if row["actual_t"]   is not None else "—"
            d_str = f"{row['deviation_s']:+.0f}" if row["deviation_s"] is not None else "—"
            flag  = "  ⚠" if row["deviation_s"] is not None and abs(row["deviation_s"]) > 5 else ""
            print(f"  {row['node']:<14}  {row['planned_t']:>9.0f}"
                  f"  {a_str:>10}  {d_str:>7}{flag}")

        gpa = vr["goal_actual_t"]
        gpd = vr["goal_deviation_s"]
        if gpa is not None:
            flag = "  ⚠ behind schedule" if gpd and gpd > 10 else (
                   "  ✓ on schedule"     if gpd is not None and abs(gpd) <= 5 else "")
            print(f"\n  Goal {vr['goal_node']}: reached at t={gpa:.0f} s"
                  f"  (planned={vr['goal_planned_t']:.0f} s, Δ={gpd:+.0f} s){flag}")
        else:
            print(f"\n  Goal {vr['goal_node']}: not reached during simulation")
        print(f"  Route deviations: {vr['num_deviations']}")

    print()
    if report["total_deviations"] == 0:
        print("  ✓ All vehicles followed their planned routes exactly.")
    else:
        print(f"  Total off-route deviations: {report['total_deviations']}")
        for d in report["deviations"]:
            print(f"    t={d['time']:.0f} s  {d['vehicle']}  →  {d['actual_edge']}")
    print("=" * 64)
