# reroute.py
"""
Weather-Aware Route Rerouting Module
Author: Ayushi Bisht
Date: October 2025

This module detects blocked routes due to weather and finds an alternate
path using Dijkstra's algorithm on the modified graph.
"""

import networkx as nx
import random
import folium
import numpy as np

# ---------- Simulated Weather-Based Blockages ----------
def get_weather_affected_edges(G, severity_level: int = 2):
    """
    Randomly simulate edges that are blocked due to bad weather.
    Higher severity means more roads are affected.
    """
    all_edges = list(G.edges())
    num_to_block = max(1, int(len(all_edges) * (0.1 * severity_level)))
    blocked = random.sample(all_edges, min(num_to_block, len(all_edges)))
    return blocked


# ---------- Alternate Route Calculation ----------
def find_alternate_route(G, source: str, target: str, blocked_edges):
    """
    Removes blocked edges and finds alternate shortest path.
    Returns both new path and updated map visualization.
    """
    # Create a copy so original graph stays safe
    G_copy = G.copy()
    G_copy.remove_edges_from(blocked_edges)

    # Try finding a new path
    try:
        alt_path = nx.dijkstra_path(G_copy, source, target, weight='weight')
        alt_distance = nx.dijkstra_path_length(G_copy, source, target, weight='weight')
    except nx.NetworkXNoPath:
        return [], float('inf')

    return alt_path, alt_distance


# ---------- Map Visualization ----------
def create_weather_reroute_map(G, original_path, alt_path, blocked_edges, source, target, map_theme="OpenStreetMap"):
    """
    Creates a Folium map showing:
    - Original route (red)
    - Alternate route (blue)
    - Blocked edges (gray dotted)
    """

    # Generate random coordinates similar to main app
    base_lat, base_lon = 40.7128, -74.0060
    coords = {}
    for i, node in enumerate(sorted(G.nodes())):
        lat_offset = (i % 3 - 1) * 0.02 + np.random.uniform(-0.005, 0.005)
        lon_offset = (i // 3 - 1) * 0.02 + np.random.uniform(-0.005, 0.005)
        coords[node] = (base_lat + lat_offset, base_lon + lon_offset)

    m = folium.Map(location=[base_lat, base_lon], zoom_start=12, tiles=map_theme)

    # Plot all edges as light gray
    for u, v in G.edges():
        folium.PolyLine([coords[u], coords[v]], color="lightgray", weight=2, opacity=0.4).add_to(m)

    # Mark blocked edges
    for u, v in blocked_edges:
        folium.PolyLine([coords[u], coords[v]], color="gray", weight=3, opacity=0.6, dash_array='5, 10',
                        popup=f"Blocked: {u} - {v}").add_to(m)

    # Plot original route (red)
    if len(original_path) > 1:
        orig_coords = [coords[n] for n in original_path if n in coords]
        folium.PolyLine(orig_coords, color="red", weight=5, opacity=0.8, popup="Original Path").add_to(m)

    # Plot alternate route (blue)
    if len(alt_path) > 1:
        alt_coords = [coords[n] for n in alt_path if n in coords]
        folium.PolyLine(alt_coords, color="blue", weight=5, opacity=0.8, popup="Alternate Path").add_to(m)

    # Add markers
    for node in G.nodes():
        lat, lon = coords[node]
        if node == source:
            folium.Marker([lat, lon], popup=f"Start: {node}", icon=folium.Icon(color="green")).add_to(m)
        elif node == target:
            folium.Marker([lat, lon], popup=f"Destination: {node}", icon=folium.Icon(color="red")).add_to(m)
        else:
            folium.Marker([lat, lon], popup=node, icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)

    return m
