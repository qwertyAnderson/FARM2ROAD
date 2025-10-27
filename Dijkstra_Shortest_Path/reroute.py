# ...existing code...
#!/usr/bin/env python3
"""
reroute.py — Weather-Based Rerouting Module for Farm2Road

Features:
- convert_path_to_edges: node path -> edge keys (A-B)
- fetch_weather_for_edge: pluggable stub (simulate or replace with real API)
- get_weather_affected_edges: returns edges with severity >= threshold
- is_road_blocked_due_to_weather: quick membership check
- calculate_alternate_route: Dijkstra avoiding blocked edges (adj-list input)
- adjust_eta_for_weather: adjust ETA using severity scores
- simulate_weather_blockages: demo random blocker generator
"""
from typing import Dict, List, Tuple, Iterable, Union, Any
import heapq
import random
import math

# Demo simulated blocked roads. Keys are edge identifiers "A-B"
SIMULATED_BLOCKED_ROADS: Dict[str, str] = {
    "A-B": "Heavy Rain",
    "C-D": "Landslide",
    "E-F": "Flood Alert",
}

# Severity mapping (higher = worse)
_SEVERITY_SCORE: Dict[str, int] = {
    "Clear": 0,
    "Light Rain": 1,
    "Rain": 2,
    "Heavy Rain": 3,
    "Flood Alert": 4,
    "Landslide": 5,
}


def convert_path_to_edges(node_path: List[str]) -> List[str]:
    """Convert node path ['A','B','C'] -> ['A-B','B-C']"""
    if not node_path or len(node_path) < 2:
        return []
    return [f"{node_path[i]}-{node_path[i+1]}" for i in range(len(node_path) - 1)]


def fetch_weather_for_edge(edge: str, use_simulation: bool = True) -> Tuple[str, int]:
    """
    Fetch weather label and severity score for an edge.
    Replace simulation with real API calls when coordinates and API key available.
    """
    if use_simulation:
        if edge in SIMULATED_BLOCKED_ROADS:
            label = SIMULATED_BLOCKED_ROADS[edge]
        else:
            label = random.choices(
                ["Clear", "Light Rain", "Rain", "Clear"],
                weights=[0.82, 0.10, 0.04, 0.04],
                k=1
            )[0]
    else:
        label = "Clear"
    score = _SEVERITY_SCORE.get(label, 0)
    return label, score


def get_weather_affected_edges(
    path_edges: List[str],
    severity_threshold: int = 2,
    use_simulation: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Check edges in path_edges and return mapping edge -> {'label','severity'}
    Only includes edges with severity >= severity_threshold.
    """
    affected: Dict[str, Dict[str, Any]] = {}
    for edge in path_edges:
        label, score = fetch_weather_for_edge(edge, use_simulation=use_simulation)
        if score >= severity_threshold:
            affected[edge] = {"label": label, "severity": score}
    return affected


def is_road_blocked_due_to_weather(
    path_edges: List[str],
    blocked_roads: Union[Dict[str, str], Iterable[str], None] = None
) -> bool:
    """
    Return True if any edge in path_edges is present in blocked_roads.
    If blocked_roads is None, uses SIMULATED_BLOCKED_ROADS.
    """
    if blocked_roads is None:
        blocked_keys = set(SIMULATED_BLOCKED_ROADS.keys())
    elif isinstance(blocked_roads, dict):
        blocked_keys = set(blocked_roads.keys())
    else:
        blocked_keys = set(blocked_roads)

    for edge in path_edges:
        if edge in blocked_keys:
            return True
    return False


def calculate_alternate_route(
    graph: Dict[str, List[Tuple[str, float]]],
    source: str,
    target: str,
    blocked_roads: Union[Dict[str, str], Iterable[str]]
) -> Tuple[List[str], float]:
    """
    Compute shortest path from source -> target on `graph` (adj-list) excluding blocked_roads.
    Returns (path_list, total_distance). If none found returns ([], inf).
    """
    if isinstance(blocked_roads, dict):
        blocked_keys = set(blocked_roads.keys())
    else:
        blocked_keys = set(blocked_roads)

    modified: Dict[str, List[Tuple[str, float]]] = {}
    for node in graph:
        modified[node] = []
        for neighbor, w in graph.get(node, []):
            edge = f"{node}-{neighbor}"
            rev = f"{neighbor}-{node}"
            if edge not in blocked_keys and rev not in blocked_keys:
                try:
                    weight = float(w)
                except Exception:
                    weight = 1.0
                modified[node].append((neighbor, weight))

    pq: List[Tuple[float, str, List[str]]] = [(0.0, source, [])]
    visited: Dict[str, float] = {}

    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in visited and cost >= visited[node] - 1e-9:
            continue
        visited[node] = cost
        new_path = path + [node]

        if node == target:
            return new_path, float(cost)

        for neigh, w in modified.get(node, []):
            next_cost = cost + float(w)
            heapq.heappush(pq, (next_cost, neigh, new_path))

    return [], float("inf")


def adjust_eta_for_weather(
    distance_km: float,
    base_speed_kmh: float,
    severity_scores: Iterable[int]
) -> Tuple[int, int]:
    """Adjust ETA using average severity; reduces effective speed per severity point."""
    if distance_km <= 0 or base_speed_kmh <= 0:
        return 0, 0

    scores = list(severity_scores)
    avg_sev = float(sum(scores) / len(scores)) if scores else 0.0
    multiplier = max(0.25, 1.0 - 0.05 * avg_sev)
    effective_speed = base_speed_kmh * multiplier
    if effective_speed <= 0:
        effective_speed = max(0.1, base_speed_kmh * 0.25)

    total_hours = distance_km / effective_speed
    hrs = int(math.floor(total_hours))
    mins = int(round((total_hours - hrs) * 60))
    if mins == 60:
        hrs += 1
        mins = 0
    return hrs, mins


def simulate_weather_blockages(graph_edges: List[str], chance: float = 0.15) -> Dict[str, str]:
    """Randomly mark some edges as blocked for demo/testing."""
    blocked: Dict[str, str] = {}
    choices = ["Light Rain", "Rain", "Heavy Rain", "Flood Alert", "Landslide"]
    for e in graph_edges:
        if random.random() < chance:
            blocked[e] = random.choice(choices)
    return blocked


if __name__ == "__main__":
    sample_graph = {
        "A": [("B", 4), ("C", 2)],
        "B": [("C", 1), ("D", 5)],
        "C": [("D", 8)],
        "D": []
    }
    source, target = "A", "D"
    shortest_path = ["A", "B", "D"]
    path_edges = convert_path_to_edges(shortest_path)
    print("Original Path:", shortest_path)
    affected = get_weather_affected_edges(path_edges, severity_threshold=2, use_simulation=True)
    print("Weather-affected edges:", affected)
    if affected:
        alt_path, dist = calculate_alternate_route(sample_graph, source, target, affected.keys())
        print("Alternate Path:", alt_path)
        print("New Distance:", dist)
        if dist < float("inf"):
            hours, mins = adjust_eta_for_weather(dist, 40.0, [v["severity"] for v in affected.values()])
            print(f"Adjusted ETA: {hours}h {mins}m")
    else:
        print("✅ All clear — current route safe.")
# ...existing code...