"""Tests for graph_processor.py — SCC, node mapping, edge extraction."""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_processor import (
    keep_largest_strongly_connected_component,
    create_pddl_location_mapping,
    extract_edges,
    extract_locations,
)


def _simple_graph() -> nx.MultiDiGraph:
    """
    Six-node strongly connected cycle (nodes 1-6) plus one dangling node (7).
    keep_largest_strongly_connected_component requires >= 5 nodes to keep the SCC;
    the six-node cycle satisfies that threshold.
    """
    g = nx.MultiDiGraph()
    for n in range(1, 8):
        g.add_node(n, x=float(n), y=float(n), street_count=2)
    # SCC: 1->2->3->4->5->6->1
    cycle_edges = [(1,2),(2,3),(3,4),(4,5),(5,6),(6,1)]
    for a, b in cycle_edges:
        g.add_edge(a, b, length=100.0, speed_kph=36.0, travel_time=10.0,
                   name="Road", highway="residential", oneway=False)
    # Dangling node 7 — only reachable from 1, no return path
    g.add_edge(1, 7, length=50.0, speed_kph=36.0, travel_time=5.0,
               name="", highway="residential", oneway=False)
    return g


class TestKeepLargestSCC:
    def test_removes_dangling_node(self):
        g = _simple_graph()
        scc = keep_largest_strongly_connected_component(g)
        assert 7 not in scc.nodes
        assert set(scc.nodes) == {1, 2, 3, 4, 5, 6}

    def test_all_edges_within_scc(self):
        g = _simple_graph()
        scc = keep_largest_strongly_connected_component(g)
        for u, v, _ in scc.edges:
            assert u in scc.nodes and v in scc.nodes


class TestCreatePddlLocationMapping:
    def test_mapping_uses_loc_prefix(self):
        g = _simple_graph()
        scc = keep_largest_strongly_connected_component(g)
        mapping = create_pddl_location_mapping(scc)
        for pddl_id in mapping.values():
            assert pddl_id.startswith("loc_")

    def test_mapping_is_bijective(self):
        g = _simple_graph()
        scc = keep_largest_strongly_connected_component(g)
        mapping = create_pddl_location_mapping(scc)
        assert len(mapping) == len(set(mapping.values()))


class TestExtractEdges:
    def test_edges_have_required_keys(self):
        g = _simple_graph()
        scc = keep_largest_strongly_connected_component(g)
        mapping = create_pddl_location_mapping(scc)
        edges = extract_edges(scc, mapping, vehicle_speed_m_per_s=10.0)
        for edge in edges:
            assert "from" in edge
            assert "to" in edge
            assert "distance_m" in edge
            assert "travel_time_s" in edge

    def test_travel_time_uses_vehicle_speed(self):
        g = _simple_graph()
        scc = keep_largest_strongly_connected_component(g)
        mapping = create_pddl_location_mapping(scc)
        edges = extract_edges(scc, mapping, vehicle_speed_m_per_s=10.0)
        for edge in edges:
            expected = round(edge["distance_m"] / 10.0, 3)
            assert abs(edge["travel_time_s"] - expected) < 0.01


class TestExtractLocations:
    def test_locations_have_lat_lon(self):
        g = _simple_graph()
        scc = keep_largest_strongly_connected_component(g)
        mapping = create_pddl_location_mapping(scc)
        locs = extract_locations(scc, mapping)
        for loc in locs:
            assert "lat" in loc
            assert "lon" in loc
            assert "id" in loc

    def test_traffic_signal_flagged(self):
        g = _simple_graph()
        scc = keep_largest_strongly_connected_component(g)
        mapping = create_pddl_location_mapping(scc)
        signal_ids = {mapping[1]}
        locs = extract_locations(scc, mapping, traffic_signal_ids=signal_ids)
        signal_locs = [l for l in locs if l.get("has_traffic_signal")]
        assert len(signal_locs) == 1
        assert signal_locs[0]["id"] == mapping[1]

    def test_charging_station_flagged(self):
        g = _simple_graph()
        scc = keep_largest_strongly_connected_component(g)
        mapping = create_pddl_location_mapping(scc)
        charger_ids = {mapping[2]}
        locs = extract_locations(scc, mapping, charging_station_ids=charger_ids)
        charger_locs = [l for l in locs if l.get("has_charging_station")]
        assert len(charger_locs) == 1
        assert charger_locs[0]["id"] == mapping[2]
