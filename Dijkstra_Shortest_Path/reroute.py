#!/usr/bin/env python3
"""
Alternate Route & Real-Time Weather Module

Handles live weather conditions to detect blocked routes
and compute alternate paths dynamically using Dijkstra’s algorithm.

Author: Ayushi
Date: October 2025
"""

import requests
import networkx as nx
import random
import time

# ========================
# 🔑 CONFIG
# ========================
API_KEY = "4a4c11792a697499ca90523b200d7aae"  # Replace with your OpenWeatherMap API key
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


# ========================
# 🌦️ Get Real-Time Weather
# ========================
def get_weather_condition(lat: float, lon: float) -> str:
    """Fetch real-time weather condition using OpenWeatherMap API."""
    try:
        params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
        response = requests.get(WEATHER_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        condition = data["weather"][0]["main"].lower()
        return condition
    except Exception:
        return "clear"  # Default fallback


# ========================
# ⚠️ Determine Blocked Edges
# ========================
def detect_blocked_edges(graph: nx.Graph, weather_condition: str):
    """
    Identify blocked routes based on weather.
    - Rain/storm => randomly block 1-2 edges
    """
    edges = list(graph.edges())
    blocked = []

    if "rain" in weather_condition or "storm" in weather_condition or "snow" in weather_condition:
        # Randomly block 1–2 edges to simulate affected routes
        num_blocks = random.randint(1, min(2, len(edges)))
        blocked = random.sample(edges, num_blocks)

    return blocked


# ========================
# 🗺️ Compute Alternate Route
# ========================
def compute_alternate_route(graph: nx.Graph, source: str, target: str, blocked_edges):
    """Find alternate route avoiding blocked edges."""
    temp_graph = graph.copy()
    for edge in blocked_edges:
        if temp_graph.has_edge(*edge):
            temp_graph.remove_edge(*edge)

    try:
        path = nx.shortest_path(temp_graph, source, target, weight="weight")
        distance = nx.shortest_path_length(temp_graph, source, target, weight="weight")
        return path, distance
    except nx.NetworkXNoPath:
        return None, None