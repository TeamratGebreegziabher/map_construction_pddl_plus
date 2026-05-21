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
    """
    Extract timestamp from lines like:
        0.000: (start-move car1 loc_2 loc_1)
    """
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*:", line)

    if not match:
        return None

    return float(match.group(1))


def extract_parenthesized_action(line: str) -> str | None:
    """
    Extract action content from the first (...) expression.

    Example:
        0.000: (start-move car1 loc_2 loc_1)
    returns:
        start-move car1 loc_2 loc_1
    """
    match = re.search(r"\(([^()]*)\)", line)

    if not match:
        return None

    return match.group(1).strip()


def parse_action_line(line: str) -> dict[str, Any] | None:
    """
    Parse one planner action line.

    Supports:
        0.000: (start-move car1 loc_2 loc_1)
        (start-move car1 loc_2 loc_1)
        170.264: (arrive car1 loc_1)
    """
    action_text = extract_parenthesized_action(line)

    if not action_text:
        return None

    tokens = action_text.split()

    if not tokens:
        return None

    action_name = tokens[0].lower()
    args = tokens[1:]

    parsed = {
        "raw": line.strip(),
        "time": extract_time_from_line(line),
        "action": action_name,
        "args": args,
    }

    if action_name == "start-move" and len(args) >= 3:
        parsed["vehicle"] = args[0]
        parsed["from"] = args[1]
        parsed["to"] = args[2]

    elif action_name == "arrive" and len(args) >= 2:
        parsed["vehicle"] = args[0]
        parsed["to"] = args[1]

    return parsed


def parse_plan_file(plan_path: str | Path) -> dict[str, Any]:
    """
    Parse one saved plan file.
    """
    path = Path(plan_path)

    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()

    actions = []

    for line in lines:
        parsed = parse_action_line(line)

        if parsed is not None:
            actions.append(parsed)

    move_actions = [
        action for action in actions if action["action"] == "start-move"
    ]

    arrive_events = [
        action for action in actions if action["action"] == "arrive"
    ]

    route = []

    if move_actions:
        route.append(move_actions[0]["from"])

        for move in move_actions:
            route.append(move["to"])

    return {
        "plan_file": str(path),
        "num_actions": len(actions),
        "num_move_actions": len(move_actions),
        "num_arrive_events": len(arrive_events),
        "actions": actions,
        "move_actions": move_actions,
        "arrive_events": arrive_events,
        "route": route,
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
    """
    Parse small, medium, and large plan files.
    """
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
            }
        )

        print(
            f"Parsed {instance_name}: "
            f"{parsed_plan['num_move_actions']} move actions, "
            f"route length {len(parsed_plan['route'])}"
        )

    return {
        "parsed_outputs": parsed_outputs,
    }