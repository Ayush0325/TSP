"""
TSP Route Optimizer — Streamlit Dashboard
==========================================
Upload a CSV/JSON of locations, solve the Travelling Salesman Problem
(exact MILP via PuLP for small instances, Nearest-Neighbor + 2-opt
heuristic for larger ones), and visualize the optimized route on an
interactive Folium map.

Run with:
    streamlit run app.py
"""

import io
import json
import math
import time
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import pulp

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="TSP Route Optimizer",
   
    layout="wide",
)

EXACT_SOLVER_LIMIT = 12  # beyond this many nodes, MILP becomes too slow


# --------------------------------------------------------------------------
# Data parsing
# --------------------------------------------------------------------------
def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Parse CSV or JSON upload into a normalized DataFrame with
    columns: name, lat, lon (geographic) OR name, x, y (planar)."""

    filename = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if filename.endswith(".json"):
        data = json.loads(raw_bytes.decode("utf-8"))
        if isinstance(data, dict):
            data = data.get("locations", data.get("points", data.get("data", data)))
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(io.BytesIO(raw_bytes))

    df.columns = [str(c).strip().lower() for c in df.columns]

    # normalize name column
    name_col = None
    for candidate in ["name", "city", "location", "id", "label"]:
        if candidate in df.columns:
            name_col = candidate
            break
    if name_col is None:
        df["name"] = [f"Point {i+1}" for i in range(len(df))]
        name_col = "name"

    # detect lat/lon
    lat_col = next((c for c in ["lat", "latitude"] if c in df.columns), None)
    lon_col = next((c for c in ["lon", "lng", "long", "longitude"] if c in df.columns), None)

    if lat_col and lon_col:
        out = pd.DataFrame({
            "name": df[name_col].astype(str),
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        })
        out.attrs["mode"] = "geo"
    else:
        x_col = next((c for c in ["x", "x_coord", "easting"] if c in df.columns), None)
        y_col = next((c for c in ["y", "y_coord", "northing"] if c in df.columns), None)
        if not (x_col and y_col):
            raise ValueError(
                "Could not find coordinate columns. Provide either "
                "'lat'/'lon' (geographic) or 'x'/'y' (planar) columns."
            )
        out = pd.DataFrame({
            "name": df[name_col].astype(str),
            "x": pd.to_numeric(df[x_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
        })
        out.attrs["mode"] = "planar"

    out = out.dropna().reset_index(drop=True)
    if len(out) < 3:
        raise ValueError("Need at least 3 valid locations to solve a TSP route.")
    return out


# --------------------------------------------------------------------------
# Distance matrix
# --------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088  # km
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_distance_matrix(df: pd.DataFrame, mode: str) -> np.ndarray:
    n = len(df)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if mode == "geo":
                d = haversine(df.lat[i], df.lon[i], df.lat[j], df.lon[j])
            else:
                d = math.hypot(df.x[i] - df.x[j], df.y[i] - df.y[j])
            mat[i, j] = mat[j, i] = d
    return mat


# --------------------------------------------------------------------------
# Exact solver (PuLP, MTZ subtour elimination formulation)
# --------------------------------------------------------------------------
def solve_tsp_exact(dist: np.ndarray, time_limit=30):
    n = len(dist)
    prob = pulp.LpProblem("TSP", pulp.LpMinimize)

    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
         for i in range(n) for j in range(n) if i != j}
    u = {i: pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n, cat="Integer")
         for i in range(1, n)}

    prob += pulp.lpSum(dist[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)

    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(n) if j != i) == 1
        prob += pulp.lpSum(x[j, i] for j in range(n) if j != i) == 1

    # MTZ subtour elimination
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i, j] <= n - 1

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)

    # reconstruct route
    route = [0]
    current = 0
    visited = {0}
    for _ in range(n - 1):
        nxt = next(j for j in range(n) if j != current and pulp.value(x[current, j]) > 0.5)
        route.append(nxt)
        visited.add(nxt)
        current = nxt
    route.append(0)

    total = sum(dist[route[k]][route[k + 1]] for k in range(len(route) - 1))
    status = pulp.LpStatus[prob.status]
    return route, total, status


# --------------------------------------------------------------------------
# Heuristic solver: Nearest Neighbor + 2-opt
# --------------------------------------------------------------------------
def nearest_neighbor_route(dist: np.ndarray, start: int = 0):
    n = len(dist)
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[current][j])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    route.append(start)
    return route


def route_length(route, dist):
    return sum(dist[route[k]][route[k + 1]] for k in range(len(route) - 1))


def two_opt(route, dist, max_iter=2000):
    best = route[:]
    best_len = route_length(best, dist)
    improved = True
    iters = 0
    n = len(best)
    while improved and iters < max_iter:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                if j - i == 1:
                    continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                new_len = route_length(new_route, dist)
                if new_len < best_len - 1e-9:
                    best, best_len = new_route, new_len
                    improved = True
            iters += 1
            if iters >= max_iter:
                break
    return best, best_len


def solve_tsp_heuristic(dist: np.ndarray):
    nn_route = nearest_neighbor_route(dist, 0)
    best_route, best_len = two_opt(nn_route, dist)
    return best_route, best_len


# --------------------------------------------------------------------------
# Map rendering
# --------------------------------------------------------------------------
def render_map(df: pd.DataFrame, route: list, mode: str):
    if mode != "geo":
        return None  # folium needs real geo coordinates

    lats = df.lat.tolist()
    lons = df.lon.tolist()
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]

    m = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")

    coords = [(df.lat[i], df.lon[i]) for i in route]
    folium.PolyLine(coords, color="#2563eb", weight=4, opacity=0.85).add_to(m)

    for order, idx in enumerate(route[:-1]):
        is_start = order == 0
        folium.CircleMarker(
            location=(df.lat[idx], df.lon[idx]),
            radius=9 if is_start else 7,
            color="#dc2626" if is_start else "#1d4ed8",
            fill=True,
            fill_color="#dc2626" if is_start else "#1d4ed8",
            fill_opacity=0.95,
            popup=f"{order + 1}. {df.name[idx]}",
        ).add_to(m)
        folium.Marker(
            location=(df.lat[idx], df.lon[idx]),
            icon=folium.DivIcon(html=f"""
                <div style="font-size:11px;font-weight:600;color:#111;
                            transform: translate(10px,-6px);">
                    {order + 1}. {df.name[idx]}
                </div>""")
        ).add_to(m)

    m.fit_bounds(coords)
    return m


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🗺️ TSP Route Optimizer")
st.caption(
    "Upload a list of locations (CSV or JSON), solve the shortest closed "
    "route, and view it on an interactive map."
)

with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader("CSV or JSON file", type=["csv", "json"])

    st.header("2. Solver settings")
    solver_choice = st.radio(
        "Solver",
        ["Auto (recommended)", "Exact (MILP - PuLP/CBC)", "Heuristic (Nearest Neighbor + 2-opt)"],
        help=f"Exact solver is only practical up to ~{EXACT_SOLVER_LIMIT} locations.",
    )
    time_limit = st.slider("Exact solver time limit (seconds)", 5, 120, 30)

    st.header("3. Cost settings")
    cost_per_unit = st.number_input(
        "Cost per distance unit (e.g. $/km)", min_value=0.0, value=0.0, step=0.5,
        help="Optional. Set to 0 to skip cost calculation."
    )

    solve_clicked = st.button("🚀 Solve TSP", type="primary", use_container_width=True)

if uploaded is None:
    st.info("👈 Upload a CSV or JSON file of locations to get started.")
    st.stop()

try:
    df = parse_uploaded_file(uploaded)
    mode = df.attrs.get("mode", "geo")
except Exception as e:
    st.error(f"Could not parse file: {e}")
    st.stop()

n = len(df)

# Display locations table full width (preview map removed)
st.subheader(f"📍 Locations ({n})")
st.dataframe(df, use_container_width=True, height=250)

st.divider()

# --------------------------------------------------------------------------
# Solve (persisted in session_state so results survive Streamlit reruns,
# e.g. the rerun that st_folium itself triggers on interaction)
# --------------------------------------------------------------------------
data_signature = (uploaded.name, uploaded.size, n, mode)

if solve_clicked:
    dist_matrix = build_distance_matrix(df, mode)

    use_exact = (
        solver_choice.startswith("Exact")
        or (solver_choice.startswith("Auto") and n <= EXACT_SOLVER_LIMIT)
    )

    if solver_choice.startswith("Exact") and n > EXACT_SOLVER_LIMIT:
        st.warning(
            f"⚠️ {n} locations is large for an exact MILP solve — this may be slow "
            f"or hit the {time_limit}s time limit without proving optimality."
        )

    with st.spinner("Solving..."):
        t0 = time.time()
        if use_exact:
            route, total_dist, status = solve_tsp_exact(dist_matrix, time_limit=time_limit)
            solver_used = f"Exact MILP (PuLP/CBC) — status: {status}"
        else:
            route, total_dist = solve_tsp_heuristic(dist_matrix)
            solver_used = "Heuristic (Nearest Neighbor + 2-opt)"
        elapsed = time.time() - t0

    st.session_state["tsp_result"] = {
        "signature": data_signature,
        "route": route,
        "total_dist": total_dist,
        "solver_used": solver_used,
        "elapsed": elapsed,
        "dist_matrix": dist_matrix,
        "cost_per_unit": cost_per_unit,
    }

result = st.session_state.get("tsp_result")

if result is None:
    st.info("Adjust settings in the sidebar, then click **Solve TSP**.")
    st.stop()

if result["signature"] != data_signature:
    st.info(
        "The uploaded data or its settings changed since the last solve. "
        "Click **Solve TSP** again to re-optimize for the current file."
    )
    st.stop()

route = result["route"]
total_dist = result["total_dist"]
solver_used = result["solver_used"]
elapsed = result["elapsed"]
dist_matrix = result["dist_matrix"]
cost_per_unit = result["cost_per_unit"]

st.success(f"Solved with **{solver_used}** in {elapsed:.2f}s")

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
unit_label = "km" if mode == "geo" else "units"
m1, m2, m3 = st.columns(3)
m1.metric("Locations visited", n)
m2.metric(f"Optimized distance ({unit_label})", f"{total_dist:,.2f}")
if cost_per_unit > 0:
    m3.metric("Estimated cost", f"{total_dist * cost_per_unit:,.2f}")
else:
    m3.metric("Estimated cost", "— set cost/unit in sidebar")

route_df = pd.DataFrame({
    "Order": range(1, len(route) + 1),
    "Location": [df.name[i] for i in route],
})
route_df["Leg distance"] = [0.0] + [
    dist_matrix[route[k]][route[k + 1]] for k in range(len(route) - 1)
]

col_a, col_b = st.columns([1, 2])
with col_a:
    st.subheader("Optimized route")
    st.dataframe(route_df, use_container_width=True, height=380)
    st.download_button(
        "⬇️ Download route as CSV",
        route_df.to_csv(index=False).encode("utf-8"),
        file_name="optimized_tsp_route.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_b:
    st.subheader("Optimized route map")
    if mode == "geo":
        result_map = render_map(df, route, mode)
        st_folium(result_map, height=460, use_container_width=True, key="result_map")
    else:
        ordered = df.iloc[route].reset_index(drop=True)
        st.line_chart(ordered, x="x", y="y")
        st.caption("Planar coordinates — showing route as an ordered line chart.")

st.caption(
    "Route starts and ends at the first location in your file. "
    "Exact solving uses a MILP formulation (PuLP + CBC) with subtour elimination; "
    "large instances automatically fall back to a Nearest-Neighbor + 2-opt heuristic."
)
