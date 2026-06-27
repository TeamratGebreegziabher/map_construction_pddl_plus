# Map Construction in PDDL+ for Urban Electric Vehicle Fleet Navigation

An automated planning system that converts real-world OpenStreetMap road networks into PDDL+ navigation problems, solves them with the ENHSP numeric planner, and executes the resulting plans inside SUMO traffic simulation.

---

## What It Does

Given a map area, the system:

1. Downloads the road network from OpenStreetMap
2. Generates a PDDL+ problem encoding roads, distances, battery consumption, charging stations, and traffic signals
3. Runs the ENHSP planner to find a valid plan
4. Visualises the planned routes on an interactive Folium map
5. Executes the plan in SUMO via TraCI, with plan-faithful departure times, charging stops, and deviation monitoring

Supports multi-vehicle fleets, with per-vehicle start and goal nodes.

---

## Prerequisites

| Dependency | Notes |
|---|---|
| Python 3.10+ | |
| Java JRE 11+ | Required to run `planners/enhsp.jar` |
| SUMO 1.18+ | `netconvert`, `sumo`, and `sumo-gui` must be on PATH or configured in `config.yaml` |
| Internet access | OSMnx downloads map tiles from OpenStreetMap on first use; results are cached |

---

## Installation

```bash
pip install -r requirements.txt
```

The ENHSP planner jar is included at `planners/enhsp.jar`. No additional installation is required for it.

---

## Running

### Interactive CLI (demo)

```bash
python demo.py
```

Walks through all steps interactively: map selection, vehicle configuration, planning, visualisation, and SUMO simulation.

### Streamlit web app

```bash
streamlit run app\planning_app.py
```

Provides the same workflow through a browser-based UI with map drawing, polygon selection, and results panels.


## Project Structure

```
map_construction_pddl_plus/
│
├── demo.py                      # Interactive CLI entry point
├── config.yaml                  # All configuration
├── requirements.txt
│
├── app/
│   └── planning_app.py          # Streamlit web app
│
├── src/
│   ├── app_pipeline.py          # Full pipeline orchestration (used by Streamlit)
│   ├── map_extractor.py         # OSMnx download and caching
│   ├── graph_processor.py       # Road graph construction and PDDL ID mapping
│   ├── pddl_generator.py        # PDDL+ domain and problem generation
│   ├── planner_runner.py        # ENHSP subprocess execution with timeout
│   ├── plan_parser.py           # ENHSP output parsing to structured JSON
│   ├── evaluator.py             # Route validation and battery metrics
│   ├── interactive_visualizer.py# Folium HTML map generation
│   └── sumo_traci_runner.py     # SUMO TraCI execution and deviation monitoring
│
├── pddl/
│   ├── domain.pddl              # PDDL+ domain (movement, battery, charging, signals)
│   └── problems/                # Auto-generated problem files per vehicle count
│
├── planners/
│   └── enhsp.jar                # ENHSP numeric planner
│
├── data/
│   └── processed/               # Saved graph instances (JSON)
│
├── outputs/
│   ├── logs/                    # Planner logs
│   ├── maps/                    # Folium HTML route maps
│   ├── plans/                   # Raw ENHSP plan text files
│   ├── results/                 # Parsed plan and evaluation JSON
│   └── sumo/                    # SUMO network, route, and config files
│
└── tests/
    ├── test_graph_processor.py
    ├── test_pddl_generator.py
    ├── test_plan_parser.py
    └── test_vehicle_repair.py
```

---

## Pipeline Overview

```
OpenStreetMap
      │
      ▼
map_extractor.py        Download and cache road network
      │
      ▼
graph_processor.py      Build directed graph, assign PDDL location IDs
      │
      ▼
pddl_generator.py       Write domain.pddl + problem.pddl
      │
      ▼
planner_runner.py       Run ENHSP (java -jar enhsp.jar -o domain -f problem)
      │
      ▼
plan_parser.py          Parse timed actions into structured JSON
      │
      ▼
evaluator.py            Validate routes, compute battery usage
      │
      ▼
interactive_visualizer.py   Generate animated Folium route map
      │
      ▼
sumo_traci_runner.py    Execute plan in SUMO, report plan vs actual
```

---

## PDDL+ Model

The domain models:

- **Movement** — `start-move` action, continuous `driving` process, triggered `arrive` event
- **Battery** — drained continuously during driving, modelled as a numeric fluent
- **Charging** — `charge` action at designated stations; `fully-charged` event ends the process
- **Traffic signals** — green/yellow/red cycle times modelled as timed events
- **Congestion** — road distances inflated by a congestion factor (higher factor = slower effective travel)

The search strategy defaults to GBFS (`search_strategy: gbfs` in config), which finds a plan quickly but does not guarantee optimality. Switch to `wastar` for cost-optimal plans on smaller instances.

---

## SUMO Integration

After planning, `sumo_traci_runner.py` executes the plan step-by-step via TraCI:

- Departure times are read from the PDDL+ plan and written into the SUMO route file
- Charging stops are injected via `traci.vehicle.setStop()`
- Planned vehicles are highlighted (enlarged, coloured halo) and the camera tracks the first vehicle
- Off-route deviations are detected when a vehicle's current edge is not part of its planned route
- On completion, a **Plan vs Execution** table is printed showing planned and actual arrival times per node

---

## Running Tests

```bash
pytest tests/
```

