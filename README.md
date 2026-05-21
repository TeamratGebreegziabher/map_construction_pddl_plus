# Overview
This project is an interactive Automated Planning prototype that converts a user-selected OpenStreetMap area into a PDDL+ urban navigation planning problem. The user selects a map area, chooses the initial and goal states, sets vehicle/resource constraints, runs the ENHSP planner, validates the generated plan, compares it with Dijkstra, visualizes the route on a map, and exports the resulting plan to SUMO for traffic simulation.

## Usage steps
1. Open the Streamlit app.
2. Search for a place.
3. Draw a small polygon/rectangle.
4. Extract graph.
5. Select start and goal nodes.
6. Run normal planning.
7. Show valid plan and route map.
8. Run blocked-road scenario.
9. Show that the planner avoids blocked roads.
10. Generate SUMO simulation.
11. Click **Open SUMO Simulation**.
12. Explain that congestion and extended evaluation are next-stage improvements.
