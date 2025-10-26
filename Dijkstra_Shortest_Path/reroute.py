#!/usr/bin/env python3
"""
Farm-to-Market Route Optimizer - Streamlit Web Application
Enhanced with Ayushi's Weather-Based Alternate Route Feature
"""

import streamlit as st
import networkx as nx
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
import io
from validation import RouteDataValidator, display_validation_help, display_validation_help_simple

# Page setup
st.set_page_config(page_title="Farm-to-Market Route Optimizer", layout="wide")

# ----------------- FUNCTIONS FROM HIMADRI'S CODE -----------------
def load_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in df.iterrows():
        if not pd.isna(row["source"]) and not pd.isna(row["target"]) and not pd.isna(row["weight"]):
            G.add_edge(str(row["source"]).strip(), str(row["target"]).strip(), weight=float(row["weight"]))
    return G


def calculate_shortest_path(G, source, target):
    try:
        path = nx.dijkstra_path(G, source, target, weight="weight")
        distance = nx.dijkstra_path_length(G, source, target, weight="weight")
        return path, distance
    except:
        return [], float("inf")


def calculate_eta(distance, speed):
    if distance == 0 or speed <= 0:
        return 0, 0
    total_hours = distance / speed
    return int(total_hours), int((total_hours - int(total_hours)) * 60)


def generate_node_coordinates(nodes):
    base_lat, base_lon = 40.7128, -74.0060
    coords = {}
    for i, node in enumerate(sorted(nodes)):
        lat_offset = (i % 3 - 1) * 0.02 + np.random.uniform(-0.005, 0.005)
        lon_offset = (i // 3 - 1) * 0.02 + np.random.uniform(-0.005, 0.005)
        coords[node] = (base_lat + lat_offset, base_lon + lon_offset)
    return coords


def create_folium_map(G, path, source, target, theme="OpenStreetMap"):
    coords = generate_node_coordinates(list(G.nodes()))
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=12, tiles=theme)
    for edge in G.edges(data=True):
        u, v = edge[0], edge[1]
        folium.PolyLine([coords[u], coords[v]], color="lightgray", weight=2).add_to(m)
    if len(path) > 1:
        folium.PolyLine([coords[n] for n in path], color="red", weight=5, opacity=0.8).add_to(m)
    for n in G.nodes():
        lat, lon = coords[n]
        if n == source:
            color, icon = "green", "play"
        elif n == target:
            color, icon = "red", "stop"
        elif n in path:
            color, icon = "orange", "info-sign"
        else:
            color, icon = "lightblue", "info-sign"
        folium.Marker(location=[lat, lon], icon=folium.Icon(color=color, icon=icon), tooltip=n).add_to(m)
    return m
# ------------------------------------------------------------------


def main():
    st.title("🌾 Farm-to-Market Route Optimizer")
    st.write("Find the optimal route from farm to market using Dijkstra's Algorithm")

    validator = RouteDataValidator()

    # Load sample CSV
    df = pd.read_csv("sample_data.csv")
    G = load_graph(df)

    nodes = sorted(list(G.nodes()))
    source = st.sidebar.selectbox("Select Source", nodes, index=0)
    target = st.sidebar.selectbox("Select Destination", nodes, index=len(nodes)-1)
    speed = st.sidebar.slider("Average Speed (km/h)", 10, 100, 40)
    map_theme = st.sidebar.selectbox("Map Theme", ["OpenStreetMap", "Stamen Terrain", "CartoDB Positron", "CartoDB Dark"])

    if st.sidebar.button("Calculate Optimal Route"):
        path, distance = calculate_shortest_path(G, source, target)
        if not path:
            st.error("No path found between selected locations.")
        else:
            st.success("Optimal route found successfully!")
            hours, mins = calculate_eta(distance, speed)
            st.metric("Total Distance", f"{distance:.2f} km")
            st.metric("Estimated Time", f"{hours}h {mins}m")
            st.write("Optimal Path:", " → ".join(path))
            st.subheader("🗺️ Route Map")
            st_folium(create_folium_map(G, path, source, target, map_theme), width=700, height=500)

    # ---- AYUSHI'S WEATHER-BASED REROUTE FEATURE ----
    from reroute import find_alternate_route

    if st.sidebar.button("☁️ Find Alternate Route (Weather Aware)"):
        st.info("Simulating weather and finding safe alternate route...")
        alt_path, alt_distance, blocked_edges = find_alternate_route(G, source, target)

        if not alt_path:
            st.error("⚠️ No alternate route found due to severe weather.")
            if blocked_edges:
                st.write("Blocked routes:", blocked_edges)
        else:
            st.success("✅ Alternate weather-safe route found!")
            st.metric("Alternate Distance", f"{alt_distance:.2f} km")
            st.write("Blocked Routes:", blocked_edges)
            st.write("Alternate Path:", " → ".join(alt_path))

            # Create map with alternate route (blue)
            coords = generate_node_coordinates(list(G.nodes()))
            alt_map = create_folium_map(G, [], source, target, map_theme)

            # Add blocked roads in red
            for (u, v) in blocked_edges:
                if u in coords and v in coords:
                    folium.PolyLine([coords[u], coords[v]], color="red", weight=4, opacity=0.8,
                                    popup=f"Blocked: {u}-{v}").add_to(alt_map)

            # Add alternate path in blue
            if len(alt_path) > 1:
                folium.PolyLine([coords[n] for n in alt_path], color="blue", weight=5,
                                opacity=0.8, popup="Alternate Route").add_to(alt_map)

            st.subheader("🗺️ Weather-Aware Alternate Route Map")
            st_folium(alt_map, width=700, height=500)
    # -------------------------------------------------


#!/usr/bin/env python3
"""
Reroute Module - Weather and Condition Aware Alternate Route Finder

Simulates real-time weather or road blockages and finds an alternate route.
Integrates seamlessly with existing Dijkstra shortest path logic.
"""

import networkx as nx
import random

def simulate_weather_conditions(G):
    """
    Simulate random bad weather or blocked routes by increasing edge weights or disabling edges.

    Args:
        G (nx.Graph): The original graph.

    Returns:
        nx.Graph: Modified graph with some edges blocked.
        list: List of blocked edges.
    """
    G_copy = G.copy()
    blocked_edges = []

    for u, v, data in list(G_copy.edges(data=True)):
        # 25% chance that an edge is blocked due to bad weather
        if random.random() < 0.25:
            blocked_edges.append((u, v))
            G_copy.remove_edge(u, v)

    return G_copy, blocked_edges


def find_alternate_route(G, source, target):
    """
    Compute alternate route considering simulated bad weather or blocked edges.

    Args:
        G (nx.Graph): The original road graph.
        source (str): Starting node.
        target (str): Destination node.

    Returns:
        tuple: (alternate_path, total_distance, blocked_edges)
    """
    modified_graph, blocked_edges = simulate_weather_conditions(G)

    try:
        alternate_path = nx.dijkstra_path(modified_graph, source, target, weight="weight")
        total_distance = nx.dijkstra_path_length(modified_graph, source, target, weight="weight")
        return alternate_path, total_distance, blocked_edges
    except nx.NetworkXNoPath:
        return [], float('inf'), blocked_edges



if __name__ == "__main__":
    main()
