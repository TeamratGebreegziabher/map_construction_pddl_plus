from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def extract_time_from_line(line: str) -> float | None:
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*:", line)
    if not match:
        return None
    return float(match.group(1))


def extract_parenthesized_action(line: str) -> str | None:
    match = re.search(r"\(([^()]*)\)", line)
    if not match:
        return None
    return match.group(1).strip()


def parse_action_line(line: str) -> dict[str, Any] | None:
    """
    Parse one planner action line.

    Supports:
        0.000: (start-move car1 loc_2 loc_1)
        0.000: (start-move car2 loc_2 loc_5)
        170.264: (arrive car1 loc_1)
        0.000: (charge car1 loc_3)
    """
    action_text = extract_parenthesized_action(line)
    if not action_text:
        return None

    tokens = action_text.split()
    if not tokens:
        return None

    action_name = tokens[0].lower()
    args = tokens[1:]

    parsed: dict[str, Any] = {
        "raw": line.strip(),
        "time": extract_time_from_line(line),
        "action": action_name,
        "args": args,
        "vehicle": None,
    }

    if action_name == "start-move" and len(args) >= 3:
        parsed["vehicle"] = args[0]
        parsed["from"] = args[1]
        parsed["to"] = args[2]

    elif action_name == "arrive" and len(args) >= 2:
        parsed["vehicle"] = args[0]
        # ENHSP emits: (arrive <vehicle> <from> <to>) — 3 args
        if len(args) >= 3:
            parsed["from"] = args[1]
            parsed["to"]   = args[2]
        else:
            parsed["to"] = args[1]

    elif action_name == "charge" and len(args) >= 2:
        parsed["vehicle"]  = args[0]
        parsed["location"] = args[1]

    elif action_name == "fully-charged" and len(args) >= 2:
        parsed["vehicle"]  = args[0]
        parsed["location"] = args[1]

    return parsed


def build_route_from_moves(
    move_actions: list[dict[str, Any]],
) -> list[str]:
    """
    Reconstruct ordered location list from start-move actions.
    Works for both single-vehicle and per-vehicle filtered lists.
    """
    route: list[str] = []
    if move_actions:
        route.append(move_actions[0]["from"])
        for move in move_actions:
            route.append(move["to"])
    return route


def build_per_vehicle_routes(
    actions: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """
    NEW — group start-move actions by vehicle and build a route per vehicle.

    Returns:
        {"car1": ["loc_2", "loc_5", "loc_8"],
         "car2": ["loc_2", "loc_1", "loc_19"]}

    Vehicles are discovered dynamically from the plan —
    no hardcoded vehicle names.
    """
    # Collect vehicle ids in order of first appearance
    vehicle_ids: list[str] = []
    seen: set[str] = set()

    for action in actions:
        vid = action.get("vehicle")
        if vid and action["action"] == "start-move" and vid not in seen:
            vehicle_ids.append(vid)
            seen.add(vid)

    routes: dict[str, list[str]] = {}

    for vid in vehicle_ids:
        vehicle_moves = [
            a for a in actions
            if a["action"] == "start-move" and a.get("vehicle") == vid
        ]
        routes[vid] = build_route_from_moves(vehicle_moves)

    return routes


def parse_plan_file(plan_path: str | Path) -> dict[str, Any]:
    """
    Parse one saved ENHSP plan file.

    Works for both single-vehicle and multi-vehicle plans.
    Per-vehicle routes are extracted automatically by vehicle name.
    """
    path = Path(plan_path)
    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()

    actions: list[dict[str, Any]] = []
    for line in lines:
        parsed = parse_action_line(line)
        if parsed is not None:
            actions.append(parsed)

    move_actions = [a for a in actions if a["action"] == "start-move"]
    arrive_events = [a for a in actions if a["action"] == "arrive"]
    charge_actions = [a for a in actions if a["action"] == "charge"]

    # Single-vehicle route (backward compatible)
    route = build_route_from_moves(move_actions)

    # NEW — per-vehicle routes for multi-vehicle plans
    per_vehicle_routes = build_per_vehicle_routes(actions)

    # Detect whether this is a multi-vehicle plan
    vehicle_ids = list(per_vehicle_routes.keys())
    is_multi_vehicle = len(vehicle_ids) > 1

    return {
        "plan_file": str(path),
        "num_actions": len(actions),
        "num_move_actions": len(move_actions),
        "num_arrive_events": len(arrive_events),
        "num_charge_actions": len(charge_actions),
        "actions": actions,
        "move_actions": move_actions,
        "arrive_events": arrive_events,
        "charge_actions": charge_actions,
        # Single-vehicle (backward compatible)
        "route": route,
        # NEW — multi-vehicle
        "is_multi_vehicle": is_multi_vehicle,
        "vehicle_ids": vehicle_ids,
        "per_vehicle_routes": per_vehicle_routes,
    }


def get_plan_files(config: dict[str, Any]) -> list[Path]:
    plans_dir = Path(config["outputs"]["plans_dir"])
    return [
        plans_dir / "small_plan.txt",
        plans_dir / "medium_plan.txt",
        plans_dir / "large_plan.txt",
    ]


def save_parsed_plan(
    parsed_plan: dict[str, Any],
    instance_name: str,
    config: dict[str, Any],
) -> Path:
    results_dir = ensure_directory(config["outputs"]["results_dir"])
    output_path = results_dir / f"{instance_name}_parsed_plan.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(parsed_plan, file, indent=2)
    return output_path


def run_plan_parsing(config: dict[str, Any]) -> dict[str, Any]:
    parsed_outputs = []
    for plan_file in get_plan_files(config):
        instance_name = plan_file.stem.replace("_plan", "")
        print(f"Parsing plan for instance: {instance_name}")
        parsed_plan = parse_plan_file(plan_file)
        parsed_path = save_parsed_plan(parsed_plan, instance_name, config)
        parsed_outputs.append(
            {
                "instance_name": instance_name,
                "plan_file": str(plan_file),
                "parsed_file": str(parsed_path),
                "num_actions": parsed_plan["num_actions"],
                "num_move_actions": parsed_plan["num_move_actions"],
                "route_length": len(parsed_plan["route"]),
                "is_multi_vehicle": parsed_plan["is_multi_vehicle"],
                "vehicle_ids": parsed_plan["vehicle_ids"],
            }
        )
        print(
            f"Parsed {instance_name}: "
            f"{parsed_plan['num_move_actions']} move actions, "
            f"route length {len(parsed_plan['route'])}, "
            f"vehicles: {parsed_plan['vehicle_ids']}"
        )
    return {"parsed_outputs": parsed_outputs}