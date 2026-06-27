"""Tests for pddl_generator.py — signal timing, PDDL structure, and congestion."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pddl_generator import (
    generate_location_feature_lines,
    generate_road_graph_lines,
    generate_problem,
    generate_multi_vehicle_problem,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _signal_instance(red: int = 30, green: int = 45) -> dict:
    return {
        "signal_red_duration": red,
        "signal_green_duration": green,
        "locations": [
            {"id": "loc_1", "has_traffic_signal": True, "osm_highway_tag": "traffic_signals"},
            {"id": "loc_2", "has_traffic_signal": False},
            {"id": "loc_3", "has_charging_station": True, "osm_amenity_tag": "charging_station"},
        ],
    }


def _base_instance() -> dict:
    locs = [
        {"id": "loc_1", "lat": 39.35, "lon": 16.22, "has_traffic_signal": False},
        {"id": "loc_2", "lat": 39.36, "lon": 16.23, "has_traffic_signal": False},
    ]
    edges = [
        {"from": "loc_1", "to": "loc_2", "distance_m": 200.0, "travel_time_s": 20.0},
    ]
    return {
        "instance_name": "test",
        "signal_red_duration": 30,
        "signal_green_duration": 45,
        "num_locations": len(locs),
        "num_edges": len(edges),
        "locations": locs,
        "edges": edges,
        "blocked_edges": [],
        "congested_edges": [],
        "vehicle": {
            "name": "car1",
            "initial_battery": 100.0,
            "max_battery": 100.0,
            "speed_m_per_s": 10.0,
            "battery_consumption_per_meter": 0.01,
            "charge_rate": 5.0,
        },
        "start": "loc_1",
        "goal": "loc_2",
    }


def _base_config() -> dict:
    return {"planning": {"metric": "distance"}}


# ---------------------------------------------------------------------------
# Signal timing tests (M1)
# ---------------------------------------------------------------------------

class TestSignalTiming:
    def test_default_timing_values(self):
        lines = generate_location_feature_lines(_signal_instance(30, 45))
        text = "\n".join(lines)
        assert "(= (red-duration loc_1) 30)" in text
        assert "(= (green-duration loc_1) 45)" in text
        assert "(= (signal-timer loc_1) 45)" in text  # timer starts at green

    def test_custom_timing_values(self):
        lines = generate_location_feature_lines(_signal_instance(20, 60))
        text = "\n".join(lines)
        assert "(= (red-duration loc_1) 20)" in text
        assert "(= (green-duration loc_1) 60)" in text
        assert "(= (signal-timer loc_1) 60)" in text

    def test_timing_fallback_when_keys_absent(self):
        """Instance without signal_*_duration keys falls back to 30/45."""
        inst = {"locations": [{"id": "loc_A", "has_traffic_signal": True}]}
        lines = generate_location_feature_lines(inst)
        text = "\n".join(lines)
        assert "(= (red-duration loc_A) 30)" in text
        assert "(= (green-duration loc_A) 45)" in text
        assert "(= (signal-timer loc_A) 45)" in text

    def test_signal_node_predicates_present(self):
        lines = generate_location_feature_lines(_signal_instance())
        text = "\n".join(lines)
        assert "(has-signal loc_1)" in text
        assert "(signal-green loc_1)" in text

    def test_non_signal_node_has_no_signal_lines(self):
        lines = generate_location_feature_lines(_signal_instance())
        text = "\n".join(lines)
        assert "loc_2" not in text  # plain node should not appear

    def test_charging_station_present(self):
        lines = generate_location_feature_lines(_signal_instance())
        text = "\n".join(lines)
        assert "(charging-station loc_3)" in text

    def test_signal_timer_equals_green_duration(self):
        """signal-timer initial value must always equal green-duration."""
        for red, green in [(10, 20), (30, 45), (60, 90)]:
            lines = generate_location_feature_lines(_signal_instance(red, green))
            text = "\n".join(lines)
            assert f"(= (signal-timer loc_1) {green})" in text, (
                f"signal-timer should be {green} for green_duration={green}"
            )

    def test_timing_written_into_pddl_problem(self):
        """End-to-end: custom timing reaches the generated PDDL problem text."""
        inst = _base_instance()
        inst["locations"][0]["has_traffic_signal"] = True
        inst["signal_red_duration"] = 15
        inst["signal_green_duration"] = 55
        problem = generate_problem(inst, _base_config())
        assert "(= (red-duration loc_1) 15)" in problem
        assert "(= (green-duration loc_1) 55)" in problem
        assert "(= (signal-timer loc_1) 55)" in problem

    def test_timing_written_into_multi_vehicle_problem(self):
        inst = _base_instance()
        inst["instance_name"] = "mv_test"
        inst["locations"][0]["has_traffic_signal"] = True
        inst["signal_red_duration"] = 25
        inst["signal_green_duration"] = 70
        vehicles = [
            {"id": "car1", "start": "loc_1", "goal": "loc_2",
             "battery": 100.0, "speed_m_per_s": 10.0,
             "battery_consumption_per_meter": 0.01,
             "max_battery": 100.0, "charge_rate": 5.0},
        ]
        problem = generate_multi_vehicle_problem(vehicles, inst, metric="distance")
        assert "(= (red-duration loc_1) 25)" in problem
        assert "(= (green-duration loc_1) 70)" in problem


# ---------------------------------------------------------------------------
# Road graph lines (congestion)
# ---------------------------------------------------------------------------

class TestRoadGraphLines:
    def test_uncongested_edge(self):
        inst = {
            "edges": [{"from": "loc_1", "to": "loc_2", "distance_m": 100.0}],
            "congested_edges": [],
        }
        lines = generate_road_graph_lines(inst)
        text = "\n".join(lines)
        assert "(connected loc_1 loc_2)" in text
        assert "(= (road-distance loc_1 loc_2) 100)" in text
        assert "congested" not in text

    def test_congested_edge_inflates_distance(self):
        inst = {
            "edges": [{"from": "loc_1", "to": "loc_2", "distance_m": 100.0}],
            "congested_edges": [
                {"from": "loc_1", "to": "loc_2", "congestion_factor": 0.5}
            ],
        }
        lines = generate_road_graph_lines(inst)
        text = "\n".join(lines)
        # effective distance = 100 / 0.5 = 200
        assert "(= (road-distance loc_1 loc_2) 200)" in text
        assert "congested" not in text.replace("; congested:", "")

    def test_blocked_edge_in_init(self):
        inst = _base_instance()
        inst["blocked_edges"] = [{"from": "loc_1", "to": "loc_2"}]
        problem = generate_problem(inst, _base_config())
        assert "(blocked loc_1 loc_2)" in problem
