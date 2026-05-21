from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_number(value: float) -> str:
    """
    Format numbers for PDDL.

    Avoid scientific notation because some planners are sensitive to it.
    """
    formatted = f"{float(value):.6f}"
    formatted = formatted.rstrip("0").rstrip(".")

    if formatted == "-0":
        formatted = "0"

    return formatted


def load_instance(instance_path: str | Path) -> dict[str, Any]:
    """Load one processed JSON planning instance."""
    path = Path(instance_path)

    if not path.exists():
        raise FileNotFoundError(f"Instance file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_all_instances(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load small, medium, and large processed instances."""
    processed_data_dir = Path(config["outputs"]["processed_data_dir"])

    instance_files = [
        processed_data_dir / "small_instance.json",
        processed_data_dir / "medium_instance.json",
        processed_data_dir / "large_instance.json",
    ]

    instances = []

    for instance_file in instance_files:
        if not instance_file.exists():
            raise FileNotFoundError(
                f"Missing processed instance: {instance_file}\n"
                "Run Stage Three before Stage Four."
            )

        instances.append(load_instance(instance_file))

    return instances


def generate_pddlplus_domain() -> str:
    """
    Generate the generic PDDL+ domain.

    This domain is independent of a specific map.
    The map-specific data goes into the problem file.
    """
    return """(define (domain osm-urban-navigation-pddlplus)

  (:requirements
    :typing
    :negative-preconditions
    :numeric-fluents
    :processes
    :events
  )

  (:types
    vehicle
    location
  )

  (:predicates
    (at ?v - vehicle ?l - location)
    (blocked ?from ?to - location)
    (visited ?l - location)
    (connected ?from ?to - location)
    (moving ?v - vehicle)
    (moving-to ?v - vehicle ?to - location)
  )
  (:functions
    (road-distance ?from ?to - location)
    (speed ?v - vehicle)
    (battery ?v - vehicle)
    (battery-rate ?v - vehicle)
    (battery-consumption-per-meter ?v - vehicle)
    (remaining-distance ?v - vehicle)
    (total-distance)
    (total-time)
  )

  (:action start-move
    :parameters (?v - vehicle ?from ?to - location)
    :precondition
      (and
        (at ?v ?from)
        (connected ?from ?to)
        (not (blocked ?from ?to))
        (not (moving ?v))
        (not (visited ?to))
        (>= (battery ?v)
            (* (road-distance ?from ?to)
               (battery-consumption-per-meter ?v)))
      )
    :effect
      (and
        (not (at ?v ?from))
        (moving ?v)
        (moving-to ?v ?to)
        (assign (remaining-distance ?v)
                (road-distance ?from ?to))
        (increase (total-distance)
                  (road-distance ?from ?to))
      )
  )

  (:process travelling
    :parameters (?v - vehicle)
    :precondition
      (moving ?v)
    :effect
      (and
        (decrease (remaining-distance ?v)
                  (* #t (speed ?v)))
        (decrease (battery ?v)
                  (* #t (battery-rate ?v)))
        (increase (total-time)
                  (* #t 1))
      )
  )

  (:event arrive
    :parameters (?v - vehicle ?to - location)
    :precondition
      (and
        (moving ?v)
        (moving-to ?v ?to)
        (<= (remaining-distance ?v) 0)
      )
    :effect
      (and
        (not (moving ?v))
        (not (moving-to ?v ?to))
        (at ?v ?to)
        (assign (remaining-distance ?v) 0)
        (visited ?to)
      )
  )
)
"""


def generate_objects_section(instance: dict[str, Any]) -> str:
    """Generate PDDL objects section."""
    vehicle_name = instance["vehicle"]["name"]
    location_ids = [location["id"] for location in instance["locations"]]

    location_lines = []
    line = "    "

    for location_id in location_ids:
        candidate = line + location_id + " "

        if len(candidate) > 78:
            location_lines.append(line.rstrip())
            line = "    " + location_id + " "
        else:
            line = candidate

    if line.strip():
        location_lines.append(line.rstrip())

    locations_text = "\n".join(location_lines)

    return f"""  (:objects
    {vehicle_name} - vehicle
{locations_text} - location
  )
"""


def generate_init_section(instance: dict[str, Any]) -> str:
    """Generate PDDL initial state."""
    vehicle = instance["vehicle"]
    vehicle_name = vehicle["name"]
    start_location = instance["start"]

    initial_battery = float(vehicle["initial_battery"])
    speed = float(vehicle["speed_m_per_s"])
    consumption_per_meter = float(vehicle["battery_consumption_per_meter"])
    battery_rate = speed * consumption_per_meter

    lines = []

    lines.append(f"    (at {vehicle_name} {start_location})")
    lines.append(f"    (visited {start_location})")
    lines.append("")
    lines.append(f"    (= (battery {vehicle_name}) {format_number(initial_battery)})")
    lines.append(f"    (= (speed {vehicle_name}) {format_number(speed)})")
    lines.append(
        f"    (= (battery-consumption-per-meter {vehicle_name}) "
        f"{format_number(consumption_per_meter)})"
    )
    lines.append(f"    (= (battery-rate {vehicle_name}) {format_number(battery_rate)})")
    lines.append(f"    (= (remaining-distance {vehicle_name}) 0)")
    lines.append(f"    (= (total-distance) 0)")
    lines.append(f"    (= (total-time) 0)")
    lines.append("")

    for edge in instance["edges"]:
        from_location = edge["from"]
        to_location = edge["to"]
        distance = float(edge["distance_m"])

        lines.append(f"    (connected {from_location} {to_location})")
        lines.append(
            f"    (= (road-distance {from_location} {to_location}) "
            f"{format_number(distance)})"
        )
    
    blocked_edges = instance.get("blocked_edges", [])

    if blocked_edges:
        lines.append("")
        lines.append("    ; Blocked road segments")

        for blocked_edge in blocked_edges:
            from_location = blocked_edge["from"]
            to_location = blocked_edge["to"]
            lines.append(f"    (blocked {from_location} {to_location})")

    init_text = "\n".join(lines)

    return f"""  (:init
{init_text}
  )
"""


def generate_goal_section(instance: dict[str, Any]) -> str:
    """Generate PDDL goal section."""
    vehicle_name = instance["vehicle"]["name"]
    goal_location = instance["goal"]

    return f"""  (:goal
    (and
      (at {vehicle_name} {goal_location})
      (not (moving {vehicle_name}))
    )
  )
"""


def generate_metric_section(config: dict[str, Any]) -> str:
    """
    Generate PDDL metric section.

    Supported metrics:
    - distance
    - time
    """
    metric = config["planning"].get("metric", "distance").lower()

    if metric == "time":
        return "  (:metric minimize (total-time))\n"

    return "  (:metric minimize (total-distance))\n"


def generate_problem(instance: dict[str, Any], config: dict[str, Any]) -> str:
    """Generate one PDDL+ problem file."""
    instance_name = instance["instance_name"]
    problem_name = f"osm-navigation-{instance_name}"

    objects = generate_objects_section(instance)
    init = generate_init_section(instance)
    goal = generate_goal_section(instance)
    metric = generate_metric_section(config)

    return f"""(define (problem {problem_name})
  (:domain osm-urban-navigation-pddlplus)

{objects}
{init}
{goal}
{metric})
"""


def save_domain(config: dict[str, Any]) -> Path:
    """Save the PDDL+ domain file."""
    domain_path = Path(config["planning"]["domain_file"])
    ensure_directory(domain_path.parent)

    domain_text = generate_pddlplus_domain()

    with domain_path.open("w", encoding="utf-8") as file:
        file.write(domain_text)

    print(f"Saved PDDL+ domain to: {domain_path}")

    return domain_path


def save_problem(
    instance: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """Save one PDDL+ problem file."""
    problem_dir = ensure_directory(config["planning"]["problem_dir"])
    instance_name = instance["instance_name"]

    problem_path = problem_dir / f"problem_{instance_name}.pddl"

    problem_text = generate_problem(instance, config)

    with problem_path.open("w", encoding="utf-8") as file:
        file.write(problem_text)

    print(f"Saved PDDL+ problem to: {problem_path}")

    return problem_path


def save_generation_summary(
    instances: list[dict[str, Any]],
    domain_path: Path,
    problem_paths: list[Path],
    config: dict[str, Any],
) -> Path:
    """Save a JSON summary of generated PDDL files."""
    results_dir = ensure_directory(config["outputs"]["results_dir"])
    output_path = results_dir / "pddl_generation_summary.json"

    summary = {
        "domain_file": str(domain_path),
        "problem_files": [str(path) for path in problem_paths],
        "instances": [
            {
                "instance_name": instance["instance_name"],
                "place_name": instance["place_name"],
                "num_locations": instance["num_locations"],
                "num_edges": instance["num_edges"],
                "start": instance["start"],
                "goal": instance["goal"],
                "estimated_shortest_distance_m": instance[
                    "estimated_shortest_distance_m"
                ],
                "vehicle": instance["vehicle"],
            }
            for instance in instances
        ],
        "metric": config["planning"].get("metric", "distance"),
        "encoding": "PDDL+ with action, process, and event",
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved PDDL generation summary to: {output_path}")

    return output_path


def run_pddl_generation(config: dict[str, Any]) -> dict[str, Any]:
    """
    Complete Stage Four:
    - load processed JSON instances
    - generate PDDL+ domain
    - generate PDDL+ problem files
    - save generation summary
    """
    print("Loading processed instances...")
    instances = load_all_instances(config)

    print("Generating PDDL+ domain...")
    domain_path = save_domain(config)

    print("Generating PDDL+ problem files...")
    problem_paths = []

    for instance in instances:
        problem_path = save_problem(instance, config)
        problem_paths.append(problem_path)

    summary_path = save_generation_summary(
        instances=instances,
        domain_path=domain_path,
        problem_paths=problem_paths,
        config=config,
    )

    return {
        "domain_path": str(domain_path),
        "problem_paths": [str(path) for path in problem_paths],
        "summary_path": str(summary_path),
    }