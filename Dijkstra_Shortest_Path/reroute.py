# ...existing code...
#!/usr/bin/env python3
"""
reroute.py — Weather-Based Rerouting Module (compatible with app.py)

Provides:
- SIMULATED_BLOCKED_ROADS
- convert_path_to_edges(node_path: List[str]) -> List[str]
- get_weather_affected_edges(path_edges: List[str], severity_threshold=2) -> Dict[str, dict]
- is_road_blocked_due_to_weather(path_edges, blocked_roads=None) -> bool
- calculate_alternate_route(graph_adj_or_nx, source, target, blocked_roads) -> (path, distance)
- adjust_eta_for_weather(distance_km, base_speed_kmh, severity_scores) -> (hrs, mins)
"""
from typing import Dict, List, Tuple, Iterable, Union, Any
import heapq
import random
import math

# Simulated blocked roads (edge key -> reason)
SIMULATED_BLOCKED_ROADS: Dict[str, str] = {
    "A-B": "Heavy Rain",
    "C-D": "Landslide",
    # Add verbose names matching your sample_data.csv if desired:
    "VillageA-VillageD": "Heavy Rain",
    "VillageD-Market": "Rain",
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
    Return (label, severity) for an edge string like "VillageA-VillageD".
    Uses SIMULATED_BLOCKED_ROADS when use_simulation=True, otherwise returns random mild weather.
    """
    if use_simulation:
        # direct match (exact)
        if edge in SIMULATED_BLOCKED_ROADS:
            label = SIMULATED_BLOCKED_ROADS[edge]
        else:
            # also try reversed key
            parts = edge.split('-')
            if len(parts) == 2:
                rev = f"{parts[1]}-{parts[0]}"
                if rev in SIMULATED_BLOCKED_ROADS:
                    label = SIMULATED_BLOCKED_ROADS[rev]
                else:
                    label = random.choices(["Clear", "Light Rain", "Rain", "Clear"],
                                           weights=[0.82, 0.10, 0.04, 0.04], k=1)[0]
            else:
                label = "Clear"
    else:
        label = "Clear"
    return label, _SEVERITY_SCORE.get(label, 0)


def get_weather_affected_edges(
    path_edges: List[str],
    severity_threshold: int = 2,
    use_simulation: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Check each edge in path_edges and return mapping edge -> {'label','severity'}
    Only edges with severity >= severity_threshold are included.
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
    Accepts blocked_roads as dict or iterable of keys.
    """
    if blocked_roads is None:
        blocked_keys = set(SIMULATED_BLOCKED_ROADS.keys())
    elif isinstance(blocked_roads, dict):
        blocked_keys = set(blocked_roads.keys())
    else:
        blocked_keys = set(blocked_roads)

    for edge in path_edges:
        # check forward or reverse ordering
        if edge in blocked_keys:
            return True
        parts = edge.split('-')
        if len(parts) == 2:
            rev = f"{parts[1]}-{parts[0]}"
            if rev in blocked_keys:
                return True
    return False


def calculate_alternate_route(
    graph: Union[Dict[str, List[Tuple[str, float]]], Any],
    source: str,
    target: str,
    blocked_roads: Union[Dict[str, str], Iterable[str]]
) -> Tuple[List[str], float]:
    """
    Compute shortest path from source -> target while excluding blocked_roads.
    graph may be an adjacency dict {node: [(nbr, weight), ...]} (this is what app.py passes)
    or a NetworkX Graph (function will detect adjacency).
    blocked_roads: dict or iterable of edge keys like 'A-B'. Both directions treated blocked.
    Returns: (path_list, total_distance) or ([], inf) if none found.
    """
    # normalize blocked list
    if isinstance(blocked_roads, dict):
        blocked_list = set(blocked_roads.keys())
    else:
        blocked_list = set(blocked_roads)

    # If user passed a NetworkX Graph, convert to adj-list
    try:
        import networkx as nx  # type: ignore
        if hasattr(graph, 'nodes') and hasattr(graph, 'neighbors'):
            adj: Dict[str, List[Tuple[str, float]]] = {}
            for u in graph.nodes():
                adj[str(u)] = []
                for v in graph.neighbors(u):
                    weight = graph[u][v].get('weight', 1.0)
                    adj[str(u)].append((str(v), float(weight)))
        else:
            adj = graph  # assume dict
    except Exception:
        adj = graph  # assume dict

    # Build modified adjacency excluding blocked edges
    modified: Dict[str, List[Tuple[str, float]]] = {}
    for node in adj:
        modified[node] = []
        for neighbor, w in adj.get(node, []):
            edge = f"{node}-{neighbor}"
            rev = f"{neighbor}-{node}"
            if edge in blocked_list or rev in blocked_list:
                continue
            try:
                weight = float(w)
            except Exception:
                weight = 1.0
            modified[node].append((neighbor, weight))

    # Dijkstra
    pq: List[Tuple[float, str, List[str]]] = [(0.0, source, [])]
    best: Dict[str, float] = {}

    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in best and cost >= best[node] - 1e-9:
            continue
        best[node] = cost
        new_path = path + [node]
        if node == target:
            return new_path, float(cost)
        for neigh, w in modified.get(node, []):
            heapq.heappush(pq, (cost + float(w), neigh, new_path))

    return [], float("inf")


def adjust_eta_for_weather(
    distance_km: float,
    base_speed_kmh: float,
    severity_scores: Iterable[int]
) -> Tuple[int, int]:
    """
    Adjust ETA (hours, minutes) for a route distance given base speed and list of severity scores.
    Simple model: reduce speed by 5% per severity point, minimum 25% of base speed.
    """
    if distance_km <= 0 or base_speed_kmh <= 0:
        return 0, 0
    scores = list(severity_scores)
    avg_sev = float(sum(scores) / len(scores)) if scores else 0.0
    multiplier = max(0.25, 1.0 - 0.05 * avg_sev)
    effective_speed = base_speed_kmh * multiplier
    if effective_speed <= 0:
        effective_speed = base_speed_kmh * 0.25
    total_hours = distance_km / effective_speed
    hrs = int(math.floor(total_hours))
    mins = int(round((total_hours - hrs) * 60))
    if mins == 60:
        hrs += 1
        mins = 0
    return hrs, mins
# ...existing code...