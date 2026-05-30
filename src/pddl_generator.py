from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_number(value: float) -> str:
    formatted = f"{float(value):.6f}"
    formatted = formatted.rstrip("0").rstrip(".")
    if formatted == "-0":
        formatted = "0"
    return formatted


def load_instance(instance_path: str | Path) -> dict[str, Any]:
    path = Path(instance_path)
    if not path.exists():
        raise FileNotFoundError(f"Instance file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_all_instances(config: dict[str, Any]) -> list[dict[str, Any]]:
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
                "Run Stage Two before Stage Three."
            )
        instances.append(load_instance(instance_file))
    return instances


def generate_pddlplus_domain() -> str:
    return """\
(define (domain osm-urban-navigation-pddlplus)

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
    (moving ?v - vehicle)
    (moving-to ?v - vehicle ?l - location)
    (visited ?v - vehicle ?l - location)
    (connected ?from ?to - location)
    (blocked ?from ?to - location)
    (charging-station ?l - location)
    (charging ?v - vehicle)
    (has-signal ?l - location)
    (signal-red ?l - location)
    (signal-green ?l - location)
    (congested ?from ?to - location)
  )

  (:functions
    (road-distance ?from ?to - location)
    (remaining-distance ?v - vehicle)
    (total-distance)
    (total-time)
    (speed ?v - vehicle)
    (battery ?v - vehicle)
    (max-battery ?v - vehicle)
    (battery-rate ?v - vehicle)
    (battery-consumption-per-meter ?v - vehicle)
    (charge-rate ?v - vehicle)
    (signal-timer ?l - location)
    (red-duration ?l - location)
    (green-duration ?l - location)
  )

  (:action start-move
    :parameters (?v - vehicle ?from ?to - location)
    :precondition
      (and
        (at ?v ?from)
        (connected ?from ?to)
        (not (blocked ?from ?to))
        (not (moving ?v))
        (not (charging ?v))
        (not (visited ?v ?to))
        (not (signal-red ?from))
        (>= (battery ?v)
            (* (road-distance ?from ?to)
               (battery-consumption-per-meter ?v)))
      )
    :effect
      (and
        (not (at ?v ?from))
        (moving ?v)
        (moving-to ?v ?to)
        (assign (remaining-distance ?v) (road-distance ?from ?to))
        (increase (total-distance) (road-distance ?from ?to))
      )
  )

  (:action charge
    :parameters (?v - vehicle ?l - location)
    :precondition
      (and
        (at ?v ?l)
        (charging-station ?l)
        (not (moving ?v))
        (not (charging ?v))
        (< (battery ?v) (max-battery ?v))
      )
    :effect
      (charging ?v)
  )

  (:action stop-charging
    :parameters (?v - vehicle ?l - location)
    :precondition
      (and
        (at ?v ?l)
        (charging ?v)
        (not (moving ?v))
      )
    :effect
      (not (charging ?v))
  )

  (:process travelling
    :parameters (?v - vehicle)
    :precondition
      (moving ?v)
    :effect
      (and
        (decrease (remaining-distance ?v) (* #t (speed ?v)))
        (decrease (battery ?v) (* #t (battery-rate ?v)))
        (increase (total-time) (* #t 1))
      )
  )

  (:process charging-process
    :parameters (?v - vehicle)
    :precondition
      (charging ?v)
    :effect
      (increase (battery ?v) (* #t (charge-rate ?v)))
  )

  (:process signal-cycling
    :parameters (?l - location)
    :precondition
      (has-signal ?l)
    :effect
      (decrease (signal-timer ?l) (* #t 1))
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
        (visited ?v ?to)
      )
  )

  (:event fully-charged
    :parameters (?v - vehicle)
    :precondition
      (and
        (charging ?v)
        (>= (battery ?v) (max-battery ?v))
      )
    :effect
      (and
        (not (charging ?v))
        (assign (battery ?v) (max-battery ?v))
      )
  )

  (:event signal-switch-to-red
    :parameters (?l - location)
    :precondition
      (and
        (has-signal ?l)
        (signal-green ?l)
        (<= (signal-timer ?l) 0)
      )
    :effect
      (and
        (not (signal-green ?l))
        (signal-red ?l)
        (assign (signal-timer ?l) (red-duration ?l))
      )
  )

  (:event signal-switch-to-green
    :parameters (?l - location)
    :precondition
      (and
        (has-signal ?l)
        (signal-red ?l)
        (<= (signal-timer ?l) 0)
      )
    :effect
      (and
        (not (signal-red ?l))
        (signal-green ?l)
        (assign (signal-timer ?l) (green-duration ?l))
      )
  )
)
"""


def generate_road_graph_lines(instance: dict[str, Any]) -> list[str]:
    congested_pairs: dict[tuple[str, str], float] = {}
    for ce in instance.get("congested_edges", []):
        factor = float(ce.get("congestion_factor", 0.3))
        if factor <= 0:
            factor = 0.1
        congested_pairs[(ce["from"], ce["to"])] = factor

    lines = []
    for edge in instance["edges"]:
        f = edge["from"]
        t = edge["to"]
        real_distance = float(edge["distance_m"])
        factor = congested_pairs.get((f, t))

        if factor is not None:
            effective_distance = real_distance / factor
            lines.append(f"    (connected {f} {t})")
            lines.append(f"    ; congested: real={real_distance}m factor={factor}")
            lines.append(f"    (= (road-distance {f} {t}) {format_number(effective_distance)})")
            lines.append(f"    (congested {f} {t})")
        else:
            lines.append(f"    (connected {f} {t})")
            lines.append(f"    (= (road-distance {f} {t}) {format_number(real_distance)})")

    return lines


def generate_vehicle_fluents(
    vehicle_name: str,
    initial_battery: float,
    speed: float,
    consumption_per_meter: float,
    max_battery: float,
    charge_rate: float,
) -> list[str]:
    battery_rate = speed * consumption_per_meter
    return [
        f"    (= (battery {vehicle_name}) {format_number(initial_battery)})",
        f"    (= (max-battery {vehicle_name}) {format_number(max_battery)})",
        f"    (= (speed {vehicle_name}) {format_number(speed)})",
        f"    (= (battery-consumption-per-meter {vehicle_name}) {format_number(consumption_per_meter)})",
        f"    (= (battery-rate {vehicle_name}) {format_number(battery_rate)})",
        f"    (= (charge-rate {vehicle_name}) {format_number(charge_rate)})",
        f"    (= (remaining-distance {vehicle_name}) 0)",
    ]


def generate_location_feature_lines(instance: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    signal_lines: list[str] = []
    charger_lines: list[str] = []

    for loc in instance["locations"]:
        lid = loc["id"]

        if loc.get("has_traffic_signal", False):
            signal_lines.append(f"    ; traffic signal at {lid} (OSM: {loc.get('osm_highway_tag', '')})")
            signal_lines.append(f"    (has-signal {lid})")
            signal_lines.append(f"    (signal-green {lid})")
            signal_lines.append(f"    (= (signal-timer {lid}) 45)")
            signal_lines.append(f"    (= (red-duration {lid}) 30)")
            signal_lines.append(f"    (= (green-duration {lid}) 45)")

        if loc.get("has_charging_station", False):
            charger_lines.append(f"    ; charging station at {lid} (OSM: {loc.get('osm_amenity_tag', '')})")
            charger_lines.append(f"    (charging-station {lid})")

    if signal_lines:
        lines.append("")
        lines.append("    ; --- traffic signals (OSM highway=traffic_signals) ---")
        lines.extend(signal_lines)

    if charger_lines:
        lines.append("")
        lines.append("    ; --- charging stations (OSM amenity=charging_station) ---")
        lines.extend(charger_lines)

    return lines


def generate_objects_section(instance: dict[str, Any]) -> str:
    vehicle_name = instance["vehicle"]["name"]
    location_ids = [loc["id"] for loc in instance["locations"]]

    location_lines: list[str] = []
    line = "    "
    for lid in location_ids:
        candidate = line + lid + " "
        if len(candidate) > 78:
            location_lines.append(line.rstrip())
            line = "    " + lid + " "
        else:
            line = candidate
    if line.strip():
        location_lines.append(line.rstrip())

    return (
        "  (:objects\n"
        f"    {vehicle_name} - vehicle\n"
        + "\n".join(location_lines) + " - location\n"
        "  )\n"
    )


def generate_init_section(instance: dict[str, Any]) -> str:
    vehicle = instance["vehicle"]
    vehicle_name = vehicle["name"]
    start_location = instance["start"]

    initial_battery = float(vehicle["initial_battery"])
    speed = float(vehicle["speed_m_per_s"])
    consumption = float(vehicle["battery_consumption_per_meter"])
    max_battery = float(vehicle.get("max_battery", initial_battery))
    charge_rate = float(vehicle.get("charge_rate", 5.0))

    lines: list[str] = []
    lines.append(f"    (at {vehicle_name} {start_location})")
    lines.append(f"    (visited {vehicle_name} {start_location})")
    lines.append("")
    lines.extend(
        generate_vehicle_fluents(
            vehicle_name, initial_battery, speed,
            consumption, max_battery, charge_rate,
        )
    )
    lines.append("    (= (total-distance) 0)")
    lines.append("    (= (total-time) 0)")
    lines.append("")
    lines.extend(generate_road_graph_lines(instance))
    lines.extend(generate_location_feature_lines(instance))

    blocked_edges = instance.get("blocked_edges", [])
    if blocked_edges:
        lines.append("")
        lines.append("    ; --- blocked road segments ---")
        for be in blocked_edges:
            lines.append(f"    (blocked {be['from']} {be['to']})")

    return "  (:init\n" + "\n".join(lines) + "\n  )\n"


def generate_goal_section(instance: dict[str, Any]) -> str:
    vehicle_name = instance["vehicle"]["name"]
    goal_location = instance["goal"]
    return (
        "  (:goal\n"
        "    (and\n"
        f"      (at {vehicle_name} {goal_location})\n"
        f"      (not (moving {vehicle_name}))\n"
        "    )\n"
        "  )\n"
    )


def generate_metric_section(metric: str = "distance") -> str:
    if metric == "time":
        return "  (:metric minimize (total-time))\n"
    return "  (:metric minimize (total-distance))\n"


def generate_problem(instance: dict[str, Any], config: dict[str, Any]) -> str:
    instance_name = instance["instance_name"]
    problem_name = f"osm-navigation-{instance_name}"
    metric = config["planning"].get("metric", "distance")

    return (
        f"(define (problem {problem_name})\n"
        "  (:domain osm-urban-navigation-pddlplus)\n\n"
        + generate_objects_section(instance) + "\n"
        + generate_init_section(instance) + "\n"
        + generate_goal_section(instance) + "\n"
        + generate_metric_section(metric)
        + ")\n"
    )


def generate_multi_vehicle_objects(
    vehicles: list[dict[str, Any]],
    instance: dict[str, Any],
) -> str:
    vehicle_ids = " ".join(v["id"] for v in vehicles)
    location_ids = [loc["id"] for loc in instance["locations"]]

    location_lines: list[str] = []
    line = "    "
    for lid in location_ids:
        candidate = line + lid + " "
        if len(candidate) > 78:
            location_lines.append(line.rstrip())
            line = "    " + lid + " "
        else:
            line = candidate
    if line.strip():
        location_lines.append(line.rstrip())

    return (
        "  (:objects\n"
        f"    {vehicle_ids} - vehicle\n"
        + "\n".join(location_lines) + " - location\n"
        "  )\n"
    )


def generate_multi_vehicle_init(
    vehicles: list[dict[str, Any]],
    instance: dict[str, Any],
) -> str:
    lines: list[str] = []

    for v in vehicles:
        vid = v["id"]
        battery = float(v.get("battery", 100.0))
        speed = float(v.get("speed_m_per_s", 10.0))
        consumption = float(v.get("battery_consumption_per_meter", 0.01))
        max_battery = float(v.get("max_battery", battery))
        charge_rate = float(v.get("charge_rate", 5.0))

        lines.append(f"    ; --- {vid} ---")
        lines.append(f"    (at {vid} {v['start']})")
        lines.append(f"    (visited {vid} {v['start']})")
        lines.extend(
            generate_vehicle_fluents(
                vid, battery, speed, consumption, max_battery, charge_rate,
            )
        )
        lines.append("")

    lines.append("    (= (total-distance) 0)")
    lines.append("    (= (total-time) 0)")
    lines.append("")
    lines.extend(generate_road_graph_lines(instance))
    lines.extend(generate_location_feature_lines(instance))

    blocked_edges = instance.get("blocked_edges", [])
    if blocked_edges:
        lines.append("")
        lines.append("    ; --- blocked road segments ---")
        for be in blocked_edges:
            lines.append(f"    (blocked {be['from']} {be['to']})")

    return "  (:init\n" + "\n".join(lines) + "\n  )\n"


def generate_multi_vehicle_goal(vehicles: list[dict[str, Any]]) -> str:
    conditions = []
    for v in vehicles:
        conditions.append(f"      (at {v['id']} {v['goal']})")
        conditions.append(f"      (not (moving {v['id']}))")
    return (
        "  (:goal\n"
        "    (and\n"
        + "\n".join(conditions) + "\n"
        "    )\n"
        "  )\n"
    )


def generate_multi_vehicle_problem(
    vehicles: list[dict[str, Any]],
    instance: dict[str, Any],
    metric: str = "distance",
) -> str:
    instance_name = instance["instance_name"]
    n = len(vehicles)
    problem_name = f"osm-navigation-{instance_name}-{n}vehicles"
    vehicle_summary = ", ".join(
        f"{v['id']}: {v['start']}to{v['goal']}" for v in vehicles
    )

    return (
        f"(define (problem {problem_name})\n"
        "  (:domain osm-urban-navigation-pddlplus)\n\n"
        f"  ; Multi-vehicle planning problem\n"
        f"  ; Vehicles : {vehicle_summary}\n"
        f"  ; Instance : {instance_name} "
        f"({instance['num_locations']} locations, {instance['num_edges']} edges)\n\n"
        + generate_multi_vehicle_objects(vehicles, instance) + "\n"
        + generate_multi_vehicle_init(vehicles, instance) + "\n"
        + generate_multi_vehicle_goal(vehicles) + "\n"
        + generate_metric_section(metric)
        + ")\n"
    )


def save_domain(config: dict[str, Any]) -> Path:
    domain_path = Path(config["planning"]["domain_file"])
    ensure_directory(domain_path.parent)
    with domain_path.open("w", encoding="utf-8") as file:
        file.write(generate_pddlplus_domain())
    print(f"Saved PDDL+ domain to: {domain_path}")
    return domain_path


def save_problem(instance: dict[str, Any], config: dict[str, Any]) -> Path:
    problem_dir = ensure_directory(config["planning"]["problem_dir"])
    problem_path = problem_dir / f"problem_{instance['instance_name']}.pddl"
    with problem_path.open("w", encoding="utf-8") as file:
        file.write(generate_problem(instance, config))
    print(f"Saved PDDL+ problem to: {problem_path}")
    return problem_path


def save_multi_vehicle_problem(
    vehicles: list[dict[str, Any]],
    instance: dict[str, Any],
    config: dict[str, Any],
    metric: str = "distance",
) -> Path:
    problem_dir = ensure_directory(config["planning"]["problem_dir"])
    n = len(vehicles)
    problem_path = problem_dir / f"problem_{instance['instance_name']}_{n}vehicles.pddl"
    with problem_path.open("w", encoding="utf-8") as file:
        file.write(generate_multi_vehicle_problem(vehicles, instance, metric))
    print(f"Saved multi-vehicle PDDL+ problem to: {problem_path}")
    return problem_path


def save_generation_summary(
    instances: list[dict[str, Any]],
    domain_path: Path,
    problem_paths: list[Path],
    config: dict[str, Any],
) -> Path:
    results_dir = ensure_directory(config["outputs"]["results_dir"])
    output_path = results_dir / "pddl_generation_summary.json"

    summary = {
        "domain_file": str(domain_path),
        "problem_files": [str(p) for p in problem_paths],
        "instances": [
            {
                "instance_name": inst["instance_name"],
                "place_name": inst["place_name"],
                "num_locations": inst["num_locations"],
                "num_edges": inst["num_edges"],
                "num_traffic_signals": inst.get("num_traffic_signals", 0),
                "num_charging_stations": inst.get("num_charging_stations", 0),
                "start": inst["start"],
                "goal": inst["goal"],
                "estimated_shortest_distance_m": inst["estimated_shortest_distance_m"],
                "vehicle": inst["vehicle"],
            }
            for inst in instances
        ],
        "metric": config["planning"].get("metric", "distance"),
        "encoding": "PDDL+ with action, process, and event",
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved PDDL generation summary to: {output_path}")
    return output_path


def run_pddl_generation(config: dict[str, Any]) -> dict[str, Any]:
    print("Loading processed instances...")
    instances = load_all_instances(config)

    print("Generating PDDL+ domain...")
    domain_path = save_domain(config)

    print("Generating single-vehicle PDDL+ problem files...")
    problem_paths: list[Path] = []
    metric = config["planning"].get("metric", "distance")

    for instance in instances:
        problem_paths.append(save_problem(instance, config))

    print("Generating multi-vehicle PDDL+ problem files...")
    for instance in instances:
        locations = instance["locations"]
        n = len(locations)

        vehicle_defaults = {
            "speed_m_per_s": float(instance["vehicle"].get("speed_m_per_s", 10.0)),
            "battery_consumption_per_meter": float(
                instance["vehicle"].get("battery_consumption_per_meter", 0.01)
            ),
            "max_battery": float(
                instance["vehicle"].get(
                    "max_battery",
                    instance["vehicle"].get("initial_battery", 100.0),
                )
            ),
            "charge_rate": 5.0,
        }

        vehicles_2 = [
            {"id": "car1", "start": locations[0]["id"],
             "goal": locations[n // 2]["id"], "battery": 100.0, **vehicle_defaults},
            {"id": "car2", "start": locations[0]["id"],
             "goal": locations[n - 1]["id"], "battery": 100.0, **vehicle_defaults},
        ]

        if vehicles_2[0]["start"] == vehicles_2[0]["goal"]:
            vehicles_2[0]["goal"] = locations[min(n // 2 + 1, n - 1)]["id"]
        if vehicles_2[1]["start"] == vehicles_2[1]["goal"]:
            vehicles_2[1]["goal"] = locations[max(n - 2, 0)]["id"]

        problem_paths.append(
            save_multi_vehicle_problem(vehicles_2, instance, config, metric)
        )

    summary_path = save_generation_summary(
        instances, domain_path, problem_paths, config
    )

    return {
        "domain_path": str(domain_path),
        "problem_paths": [str(p) for p in problem_paths],
        "summary_path": str(summary_path),
    }