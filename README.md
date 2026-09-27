# TSP Route Optimizer — Streamlit Dashboard

A Streamlit dashboard that solves the Travelling Salesman Problem (TSP) from
user-uploaded location data and visualizes the optimized route on an
interactive Folium map.

## Features

- **Upload CSV or JSON** of locations (`name, lat, lon` or `name, x, y`)
- **Two solvers**
  - **Exact (PuLP + CBC)** — MILP with MTZ subtour elimination, guarantees the
    optimal route. Practical up to ~12 locations.
  - **Heuristic (Nearest Neighbor + 2-opt)** — scales to hundreds of
    locations, near-optimal in practice.
  - **Auto** mode picks the right one based on how many locations you upload.
- **Interactive Folium map** with the optimized route drawn, numbered stops,
  and a highlighted start point.
- **Optimized distance** (km, using the haversine formula for real
  coordinates) and **optional cost estimate** (distance × cost-per-unit).
- **Downloadable route** as CSV.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input format

CSV example:

```csv
name,lat,lon
Delhi,28.6139,77.2090
Mumbai,19.0760,72.8777
Bengaluru,12.9716,77.5946
```

JSON example:

```json
[
  {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
  {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
  {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946}
]
```

If you don't have geographic coordinates, you can instead use planar `x`/`y`
columns (e.g. factory floor or warehouse coordinates) — the app will detect
this automatically, use Euclidean distance, and show the route as a line
chart instead of a map (Folium requires real lat/lon).

A ready-to-use `sample_locations.csv` (10 Indian cities) is included for
testing.

## How it works

1. **Parsing** — flexible column-name detection (`name`/`city`/`id`,
   `lat`/`latitude`, `lon`/`lng`/`longitude`, or `x`/`y`).
2. **Distance matrix** — haversine (great-circle, km) for lat/lon, Euclidean
   for x/y.
3. **Solving**
   - Exact: PuLP MILP with binary arc variables and MTZ subtour-elimination
     constraints, solved with the bundled CBC solver.
   - Heuristic: Nearest-Neighbor construction followed by 2-opt local search
     until no improving swap remains.
4. **Visualization** — Folium `PolyLine` for the route, numbered
   `CircleMarker`s for stops, red marker for the start/end point.

## Notes

- The exact solver's runtime grows quickly with the number of locations
  (MTZ MILP is NP-hard) — a time limit (default 30s) prevents it from
  hanging; if not solved to proven optimality within that window it returns
  the best incumbent found.
- For anything beyond ~12–15 locations, use the heuristic solver.
