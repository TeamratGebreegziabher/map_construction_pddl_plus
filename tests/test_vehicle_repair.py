"""Tests for vehicle auto-repair functions in app_pipeline.py."""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app_pipeline import (
    haversine_m,
    check_vehicle_feasibility,
    _build_excluded_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _linear_graph() -> nx.DiGraph:
    """A → B → C → D (linear, directed)."""
    g = nx.DiGraph()
    edges = [("A", "B", 100.0), ("B", "C", 200.0), ("C", "D", 150.0)]
    for u, v, d in edges:
        g.add_edge(u, v, distance_m=d)
    return g


def _instance_with_locations(ids_latlons: dict[str, tuple[float, float]]) -> dict:
    """Build a minimal instance with the given location ids and coordinates."""
    return {
        "locations": [
            {"id": lid, "lat": lat, "lon": lon}
            for lid, (lat, lon) in ids_latlons.items()
        ],
        "edges": [],
        "blocked_edges": [],
        "congested_edges": [],
    }


def _vehicle(vid="car1", start="A", goal="D", battery=100.0,
             max_battery=100.0, consumption=0.01) -> dict:
    return {
        "id": vid,
        "start": start,
        "goal": goal,
        "battery": battery,
        "max_battery": max_battery,
        "battery_consumption_per_meter": consumption,
        "speed_m_per_s": 10.0,
        "charge_rate": 5.0,
    }


# ---------------------------------------------------------------------------
# haversine_m
# ---------------------------------------------------------------------------

class TestHaversineM:
    def test_same_point_is_zero(self):
        assert haversine_m(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        # Equatorial degree ≈ 111_320 m; 1° lat difference from (0,0) to (1,0)
        d = haversine_m(0.0, 0.0, 1.0, 0.0)
        assert 110_000 < d < 112_000

    def test_symmetric(self):
        d1 = haversine_m(39.35, 16.22, 39.40, 16.28)
        d2 = haversine_m(39.40, 16.28, 39.35, 16.22)
        assert d1 == pytest.approx(d2, rel=1e-6)


# ---------------------------------------------------------------------------
# check_vehicle_feasibility
# ---------------------------------------------------------------------------

class TestDiagnoseAndRepair:
    def _base_instance(self, consumption: float = 0.01) -> dict:
        return {
            "locations": [
                {"id": "A", "lat": 0.0,  "lon": 0.0},
                {"id": "B", "lat": 0.01, "lon": 0.0},
                {"id": "C", "lat": 0.02, "lon": 0.0},
                {"id": "D", "lat": 0.03, "lon": 0.0},
            ],
            "edges": [
                {"from": "A", "to": "B", "distance_m": 100.0, "travel_time_s": 10.0},
                {"from": "B", "to": "C", "distance_m": 200.0, "travel_time_s": 20.0},
                {"from": "C", "to": "D", "distance_m": 150.0, "travel_time_s": 15.0},
            ],
            "blocked_edges": [],
            "congested_edges": [],
            "vehicle": {
                "name": "car1",
                "initial_battery": 100.0,
                "max_battery": 100.0,
                "speed_m_per_s": 10.0,
                "battery_consumption_per_meter": consumption,
                "charge_rate": 5.0,
            },
            "start": "A",
            "goal":  "D",
        }

    def test_feasible_vehicle_has_no_repair(self):
        """Vehicle that can reach goal with enough battery is left unchanged."""
        g = _linear_graph()
        inst = self._base_instance()
        v = _vehicle(start="A", goal="D", battery=10.0, consumption=0.01)
        result = check_vehicle_feasibility(v, inst, g)
        assert result["repaired"] is False
        assert result["excluded"] is False

    def test_battery_insufficient_excludes_vehicle(self):
        """When battery cannot cover the route and no charger, vehicle is excluded."""
        g = _linear_graph()
        # total distance A→D = 100+200+150 = 450 m; at 0.1/m need 45 units; only have 5
        v = _vehicle(start="A", goal="D", battery=5.0, max_battery=5.0, consumption=0.1)
        inst = self._base_instance(consumption=0.1)
        result = check_vehicle_feasibility(v, inst, g)
        assert result["excluded"] is True
        assert result["repaired"] is False
        assert result["diagnosis"]["failure"] == "battery_insufficient"
        assert result["repaired_vehicle"]["battery"] == 5.0   # unchanged

    def test_no_path_excludes_vehicle(self):
        """Vehicle with no directed path to goal is excluded; start and goal unchanged."""
        g = nx.DiGraph()
        g.add_edge("A", "B", distance_m=100.0)
        g.add_edge("B", "C", distance_m=200.0)
        g.add_node("D")  # D disconnected
        inst = self._base_instance()
        v = _vehicle(start="A", goal="D", battery=100.0)
        result = check_vehicle_feasibility(v, inst, g)
        assert result["excluded"] is True
        assert result["repaired"] is False
        assert result["repaired_vehicle"]["goal"] == "D"   # goal never changed
        assert result["repaired_vehicle"]["start"] == "A"  # start never changed
        assert result["diagnosis"]["failure"] == "no_path"

    def test_isolated_start_excludes_vehicle(self):
        """Vehicle with isolated start is excluded; start and goal unchanged."""
        g = nx.DiGraph()
        g.add_edge("B", "C", distance_m=200.0)
        g.add_edge("C", "D", distance_m=150.0)
        g.add_node("A")  # A cannot reach anything
        inst = self._base_instance()
        v = _vehicle(start="A", goal="D", battery=100.0)
        result = check_vehicle_feasibility(v, inst, g)
        assert result["excluded"] is True
        assert result["repaired"] is False
        assert result["repaired_vehicle"]["start"] == "A"
        assert result["diagnosis"]["failure"] == "no_path"

    def test_truly_disconnected_excludes_vehicle(self):
        """When both goal and start fixes fail, vehicle is excluded."""
        g = nx.DiGraph()
        g.add_node("A")
        g.add_node("D")  # No edges at all
        inst = {
            "locations": [
                {"id": "A", "lat": 0.0, "lon": 0.0},
                {"id": "D", "lat": 0.1, "lon": 0.0},
            ],
            "edges": [],
            "blocked_edges": [],
            "congested_edges": [],
        }
        v = _vehicle(start="A", goal="D", battery=100.0)
        result = check_vehicle_feasibility(v, inst, g)
        assert result["excluded"] is True
        assert result["repaired"] is False
        assert result["diagnosis"]["failure"] == "no_path"

    def test_original_nodes_always_reflect_user_input(self):
        """original_start and original_goal always reflect the user's input, even on exclusion."""
        g = nx.DiGraph()
        g.add_edge("A", "B", distance_m=100.0)
        g.add_node("D")  # D disconnected
        inst = self._base_instance()
        v = _vehicle(start="A", goal="D", battery=100.0)
        result = check_vehicle_feasibility(v, inst, g)
        assert result["original_start"] == "A"
        assert result["original_goal"] == "D"
        assert result["repaired_vehicle"]["start"] == "A"
        assert result["repaired_vehicle"]["goal"] == "D"


# ---------------------------------------------------------------------------
# _build_excluded_result
# ---------------------------------------------------------------------------

class TestBuildExcludedResult:
    def test_structure(self):
        v = _vehicle("car2", "A", "D")
        repair = {
            "original_start": "A",
            "original_goal":  "D",
            "diagnosis": {"failure": "no_path", "reason": "Disconnected"},
        }
        r = _build_excluded_result(v, repair)
        assert r["vehicle_id"] == "car2"
        assert r["excluded"] is True
        assert r["route_valid"] is False
        assert r["route"] == []
        assert r["repaired"] is False
        assert r["diagnosis"]["failure"] == "no_path"
