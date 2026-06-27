"""Tests for plan_parser.py."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plan_parser import parse_plan_file, parse_action_line, build_per_vehicle_routes


class TestParseActionLine:
    def test_start_move(self):
        line = "0.000: (start-move car1 loc_2 loc_1)"
        action = parse_action_line(line)
        assert action is not None
        assert action["action"] == "start-move"
        assert action["vehicle"] == "car1"
        assert action["from"] == "loc_2"
        assert action["to"] == "loc_1"
        assert action["time"] == 0.0

    def test_arrive(self):
        line = "170.264: (arrive car1 loc_1)"
        action = parse_action_line(line)
        assert action["action"] == "arrive"
        assert action["vehicle"] == "car1"
        assert action["to"] == "loc_1"
        assert abs(action["time"] - 170.264) < 1e-9

    def test_charge(self):
        line = "0.000: (charge car1 loc_3)"
        action = parse_action_line(line)
        assert action["action"] == "charge"
        assert action["vehicle"] == "car1"
        assert action["location"] == "loc_3"

    def test_non_action_line_returns_none(self):
        assert parse_action_line("") is None
        assert parse_action_line("some plain text") is None
        assert parse_action_line("Plan found!") is None


class TestBuildPerVehicleRoutes:
    def _make_move(self, vehicle, frm, to, t=0.0):
        return {"action": "start-move", "vehicle": vehicle,
                "from": frm, "to": to, "time": t}

    def test_single_vehicle(self):
        actions = [
            self._make_move("car1", "loc_1", "loc_2"),
            self._make_move("car1", "loc_2", "loc_3"),
        ]
        routes = build_per_vehicle_routes(actions)
        assert routes == {"car1": ["loc_1", "loc_2", "loc_3"]}

    def test_multi_vehicle(self):
        actions = [
            self._make_move("car1", "loc_1", "loc_2"),
            self._make_move("car2", "loc_5", "loc_6"),
            self._make_move("car1", "loc_2", "loc_3"),
        ]
        routes = build_per_vehicle_routes(actions)
        assert routes["car1"] == ["loc_1", "loc_2", "loc_3"]
        assert routes["car2"] == ["loc_5", "loc_6"]

    def test_empty_actions(self):
        assert build_per_vehicle_routes([]) == {}


class TestParsePlanFile:
    def _write_plan(self, content):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_single_vehicle_plan(self):
        path = self._write_plan(
            "0.000: (start-move car1 loc_1 loc_2)\n"
            "20.000: (arrive car1 loc_2)\n"
            "20.000: (start-move car1 loc_2 loc_3)\n"
            "40.000: (arrive car1 loc_3)\n"
        )
        parsed = parse_plan_file(path)
        assert parsed["num_move_actions"] == 2
        assert parsed["route"] == ["loc_1", "loc_2", "loc_3"]
        assert not parsed["is_multi_vehicle"]
        path.unlink()

    def test_multi_vehicle_plan(self):
        path = self._write_plan(
            "0.000: (start-move car1 loc_1 loc_2)\n"
            "0.000: (start-move car2 loc_5 loc_6)\n"
            "20.000: (arrive car1 loc_2)\n"
            "20.000: (arrive car2 loc_6)\n"
        )
        parsed = parse_plan_file(path)
        assert parsed["is_multi_vehicle"]
        assert parsed["per_vehicle_routes"]["car1"] == ["loc_1", "loc_2"]
        assert parsed["per_vehicle_routes"]["car2"] == ["loc_5", "loc_6"]
        path.unlink()

    def test_plan_with_charge_action(self):
        path = self._write_plan(
            "0.000: (start-move car1 loc_1 loc_3)\n"
            "50.000: (arrive car1 loc_3)\n"
            "50.000: (charge car1 loc_3)\n"
            "100.000: (start-move car1 loc_3 loc_5)\n"
            "150.000: (arrive car1 loc_5)\n"
        )
        parsed = parse_plan_file(path)
        assert parsed["num_charge_actions"] == 1
        assert parsed["charge_actions"][0]["location"] == "loc_3"
        assert parsed["route"] == ["loc_1", "loc_3", "loc_5"]
        path.unlink()

    def test_empty_plan_file(self):
        path = self._write_plan("")
        parsed = parse_plan_file(path)
        assert parsed["num_actions"] == 0
        assert parsed["route"] == []
        path.unlink()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_plan_file(Path("nonexistent_plan.txt"))
