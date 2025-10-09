#!/usr/bin/env python3
"""
Farm-to-Market Route Optimizer - Streamlit Web Application

Interactive web interface for calculating optimal routes using Dijkstra's algorithm
with real-time map visualization powered by Folium.

Author: AI Assistant
Date: October 2025
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


# Configure Streamlit page
st.set_page_config(
    page_title="🚜 Farm-to-Market Route Optimizer",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .path-container {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Load graph data from DataFrame and create a NetworkX graph.
    
    Args:
        df (pd.DataFrame): DataFrame containing edge data with columns: source, target, weight
        
    Returns:
        nx.Graph: NetworkX graph object with weighted edges
    """
    try:
        G = nx.Graph()
        
        # Validate required columns
        required_cols = ['source', 'target', 'weight']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {', '.join(required_cols)}")
            return None
        
        # Add edges with weights to the graph
        for _, row in df.iterrows():
            if pd.isna(row['source']) or pd.isna(row['target']) or pd.isna(row['weight']):
                continue
            G.add_edge(str(row['source']).strip(), str(row['target']).strip(), weight=float(row['weight']))
        
        return G
        
    except Exception as e:
        st.error(f"Error loading graph: {str(e)}")
        return None


def calculate_shortest_path(G: nx.Graph, source: str, target: str) -> Tuple[List[str], float]:
    """
    Calculate the shortest path between two nodes using Dijkstra's algorithm.
    
    Args:
        G (nx.Graph): NetworkX graph object
        source (str): Starting node
        target (str): Ending node
        
    Returns:
        Tuple[List[str], float]: Shortest path as list of nodes and total distance
    """
    try:
        if source == target:
            return [source], 0.0
            
        shortest_path = nx.dijkstra_path(G, source, target, weight='weight')
        shortest_distance = nx.dijkstra_path_length(G, source, target, weight='weight')
        
        return shortest_path, shortest_distance
        
    except nx.NetworkXNoPath:
        return [], float('inf')
    except nx.NodeNotFound:
        return [], float('inf')
    except Exception:
        return [], float('inf')


def calculate_eta(distance: float, speed: float) -> Tuple[int, int]:
    """
    Calculate estimated travel time.
    
    Args:
        distance (float): Distance in kilometers
        speed (float): Average speed in km/h
        
    Returns:
        Tuple[int, int]: Hours and minutes
    """
    if distance == 0 or speed <= 0:
        return 0, 0
    
    total_hours = distance / speed
    hours = int(total_hours)
    minutes = int((total_hours - hours) * 60)
    
    return hours, minutes


def generate_node_coordinates(nodes: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Generate realistic coordinates for nodes based on their names.
    
    Args:
        nodes (List[str]): List of node names
        
    Returns:
        Dict[str, Tuple[float, float]]: Dictionary mapping node names to (lat, lon) coordinates
    """
    # Base coordinates (around a rural farming area)
    base_lat, base_lon = 40.7128, -74.0060  # New York area as example
    
    coordinates = {}
    
    # Assign specific coordinate patterns based on node types
    for i, node in enumerate(sorted(nodes)):
        # Create a grid-like pattern with some randomness
        lat_offset = (i % 3 - 1) * 0.02 + np.random.uniform(-0.005, 0.005)
        lon_offset = (i // 3 - 1) * 0.02 + np.random.uniform(-0.005, 0.005)
        
        # Special positioning for common node types
        if 'farm' in node.lower():
            lat_offset -= 0.01  # Farms slightly south
        elif 'market' in node.lower():
            lat_offset += 0.01  # Markets slightly north
        elif 'village' in node.lower():
            # Villages spread around the middle
            pass
            
        coordinates[node] = (base_lat + lat_offset, base_lon + lon_offset)
    
    return coordinates


def create_folium_map(G: nx.Graph, shortest_path: List[str], source: str, target: str, 
                     map_theme: str = "OpenStreetMap") -> folium.Map:
    """
    Create a Folium map visualization of the graph and shortest path.
    
    Args:
        G (nx.Graph): NetworkX graph object
        shortest_path (List[str]): List of nodes in the shortest path
        source (str): Starting node
        target (str): Ending node
        map_theme (str): Map tile theme
        
    Returns:
        folium.Map: Folium map object
    """
    # Generate coordinates for all nodes
    coordinates = generate_node_coordinates(list(G.nodes()))
    
    # Calculate map center
    if coordinates:
        center_lat = np.mean([coord[0] for coord in coordinates.values()])
        center_lon = np.mean([coord[1] for coord in coordinates.values()])
    else:
        center_lat, center_lon = 40.7128, -74.0060
    
    # Map themes
    tile_options = {
        "OpenStreetMap": "OpenStreetMap",
        "Stamen Terrain": "Stamen Terrain",
        "CartoDB Positron": "cartodb positron",
        "CartoDB Dark": "cartodb dark_matter"
    }
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles=tile_options.get(map_theme, "OpenStreetMap")
    )
    
    # Add all edges as light gray lines
    for edge in G.edges(data=True):
        node1, node2 = edge[0], edge[1]
        if node1 in coordinates and node2 in coordinates:
            folium.PolyLine(
                locations=[coordinates[node1], coordinates[node2]],
                color='lightgray',
                weight=2,
                opacity=0.6
            ).add_to(m)
    
    # Highlight shortest path
    if len(shortest_path) > 1:
        path_coordinates = [coordinates[node] for node in shortest_path if node in coordinates]
        if len(path_coordinates) > 1:
            folium.PolyLine(
                locations=path_coordinates,
                color='red',
                weight=5,
                opacity=0.8,
                popup=f"Shortest Path: {' → '.join(shortest_path)}"
            ).add_to(m)
    
    # Add node markers
    for node in G.nodes():
        if node in coordinates:
            lat, lon = coordinates[node]
            
            # Determine marker color and icon
            if node == source:
                color = 'green'
                icon = 'play'
                popup_text = f"🚜 START: {node}"
            elif node == target:
                color = 'red'
                icon = 'stop'
                popup_text = f"🏬 DESTINATION: {node}"
            elif node in shortest_path:
                color = 'orange'
                icon = 'info-sign'
                popup_text = f"🏘️ PATH: {node}"
            else:
                color = 'lightblue'
                icon = 'info-sign'
                popup_text = f"📍 {node}"
            
            folium.Marker(
                location=[lat, lon],
                popup=popup_text,
                tooltip=node,
                icon=folium.Icon(color=color, icon=icon, prefix='glyphicon')
            ).add_to(m)
    
    return m


def format_path_display(path: List[str]) -> str:
    """
    Format the path for display with emoji indicators.
    
    Args:
        path (List[str]): List of nodes in the path
        
    Returns:
        str: Formatted path string with emojis
    """
    if not path:
        return "No path found"
    
    if len(path) == 1:
        return f"🚜 {path[0]}"
    
    formatted_path = []
    for i, node in enumerate(path):
        if i == 0:
            formatted_path.append(f"🚜 {node}")
        elif i == len(path) - 1:
            formatted_path.append(f"🏬 {node}")
        else:
            formatted_path.append(f"🏘️ {node}")
    
    return " → ".join(formatted_path)


def main():
    """
    Main Streamlit application function.
    """
    # Header
    st.markdown('<h1 class="main-header">🚜 Farm-to-Market Route Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("**Find the optimal route from farm to market using Dijkstra's Algorithm with interactive map visualization**")
    
    # Sidebar
    st.sidebar.header("🎛️ Route Configuration")
    
    # File upload option
    st.sidebar.subheader("📁 Data Source")
    use_default = st.sidebar.radio(
        "Choose data source:",
        ["Use default sample data", "Upload custom CSV file"]
    )
    
    # Load data
    if use_default == "Upload custom CSV file":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="CSV should have columns: source, target, weight"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Uploaded: {len(df)} edges")
            except Exception as e:
                st.sidebar.error(f"❌ Error reading file: {str(e)}")
                df = None
        else:
            df = None
    else:
        # Use default data
        try:
            df = pd.read_csv("sample_data.csv")
            st.sidebar.success(f"✅ Default data: {len(df)} edges")
        except FileNotFoundError:
            st.sidebar.error("❌ Default sample_data.csv not found!")
            df = None
        except Exception as e:
            st.sidebar.error(f"❌ Error loading default data: {str(e)}")
            df = None
    
    if df is not None:
        # Create graph
        with st.spinner("🔄 Loading network..."):
            G = load_graph(df)
        
        if G is not None and G.number_of_nodes() > 0:
            # Display network info
            st.sidebar.subheader("📊 Network Info")
            st.sidebar.info(f"""
            **Nodes:** {G.number_of_nodes()}  
            **Edges:** {G.number_of_edges()}  
            **Connected:** {'✅ Yes' if nx.is_connected(G) else '❌ No'}
            """)
            
            # Route selection
            st.sidebar.subheader("🗺️ Route Selection")
            
            nodes = sorted(list(G.nodes()))
            
            # Source selection
            default_source = next((node for node in nodes if 'farm' in node.lower()), nodes[0])
            source_idx = nodes.index(default_source) if default_source in nodes else 0
            source = st.sidebar.selectbox(
                "📍 Start Location:",
                nodes,
                index=source_idx,
                help="Select the starting point for your route"
            )
            
            # Target selection
            default_target = next((node for node in nodes if 'market' in node.lower()), nodes[-1])
            target_idx = nodes.index(default_target) if default_target in nodes else len(nodes)-1
            target = st.sidebar.selectbox(
                "🎯 Destination:",
                nodes,
                index=target_idx,
                help="Select the destination for your route"
            )
            
            # Speed configuration
            st.sidebar.subheader("⚡ Travel Settings")
            speed = st.sidebar.slider(
                "🚗 Average Speed (km/h):",
                min_value=10,
                max_value=100,
                value=40,
                step=5,
                help="Average travel speed for ETA calculation"
            )
            
            # Map theme selection
            st.sidebar.subheader("🗺️ Map Settings")
            map_theme = st.sidebar.selectbox(
                "🎨 Map Theme:",
                ["OpenStreetMap", "Stamen Terrain", "CartoDB Positron", "CartoDB Dark"],
                help="Choose the visual style for the map"
            )
            
            # Calculate button
            if st.sidebar.button("🧮 Calculate Optimal Route", type="primary"):
                
                # Validation
                if source == target:
                    st.sidebar.warning("⚠️ Start and destination are the same!")
                else:
                    # Show progress
                    progress_placeholder = st.empty()
                    progress_placeholder.info("🔄 Calculating shortest path...")
                    
                    # Calculate shortest path
                    with st.spinner("Computing optimal route..."):
                        time.sleep(0.5)  # Brief delay for UX
                        shortest_path, distance = calculate_shortest_path(G, source, target)
                    
                    progress_placeholder.empty()
                    
                    # Main content area
                    if shortest_path and distance != float('inf'):
                        # Success message
                        st.markdown(f'''
                        <div class="success-message">
                            <strong>✅ Route Successfully Calculated!</strong><br>
                            Optimal path found from <strong>{source}</strong> to <strong>{target}</strong>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Metrics row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                label="📏 Total Distance",
                                value=f"{distance:.1f} km"
                            )
                        
                        with col2:
                            hours, minutes = calculate_eta(distance, speed)
                            eta_text = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                            st.metric(
                                label="⏱️ Estimated Time",
                                value=eta_text
                            )
                        
                        with col3:
                            st.metric(
                                label="🛑 Number of Stops",
                                value=len(shortest_path) - 1
                            )
                        
                        with col4:
                            st.metric(
                                label="🚗 Average Speed",
                                value=f"{speed} km/h"
                            )
                        
                        # Path display
                        st.markdown(f'''
                        <div class="path-container">
                            <h3>🗺️ Optimal Route Path</h3>
                            <p style="font-size: 1.2rem; font-weight: bold;">{format_path_display(shortest_path)}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Detailed route breakdown
                        if len(shortest_path) > 1:
                            st.subheader("📋 Step-by-Step Directions")
                            
                            for i in range(len(shortest_path) - 1):
                                current = shortest_path[i]
                                next_node = shortest_path[i + 1]
                                
                                # Get edge weight
                                edge_weight = G[current][next_node]['weight']
                                step_hours, step_minutes = calculate_eta(edge_weight, speed)
                                step_eta = f"{step_hours}h {step_minutes}m" if step_hours > 0 else f"{step_minutes}m"
                                
                                st.write(f"**Step {i+1}:** {current} → {next_node} ({edge_weight:.1f} km, ~{step_eta})")
                        
                        # Interactive map
                        st.subheader("🗺️ Interactive Route Map")
                        st.write("Explore the route on the interactive map below:")
                        
                        with st.spinner("🗺️ Generating map..."):
                            folium_map = create_folium_map(G, shortest_path, source, target, map_theme)
                        
                        # Display map
                        map_data = st_folium(
                            folium_map,
                            width=700,
                            height=500,
                            returned_objects=["last_object_clicked"]
                        )
                        
                        # Map legend
                        st.markdown("""
                        **Map Legend:**
                        - 🟢 **Green Marker**: Start location
                        - 🔴 **Red Marker**: Destination
                        - 🟠 **Orange Markers**: Route waypoints
                        - 🔴 **Red Line**: Optimal route
                        - ⚪ **Gray Lines**: Alternative routes
                        """)
                        
                        # Additional analysis
                        with st.expander("📊 Advanced Route Analysis"):
                            st.write("**Alternative Routes Analysis:**")
                            
                            # Calculate all shortest paths to all nodes
                            try:
                                all_distances = nx.single_source_dijkstra_path_length(G, source, weight='weight')
                                
                                st.write("Distances from start to all locations:")
                                for node, dist in sorted(all_distances.items()):
                                    if node != source:
                                        h, m = calculate_eta(dist, speed)
                                        eta = f"{h}h {m}m" if h > 0 else f"{m}m"
                                        st.write(f"- **{node}**: {dist:.1f} km (~{eta})")
                                        
                            except Exception as e:
                                st.write(f"Could not calculate alternative routes: {str(e)}")
                        
                        # Export options
                        with st.expander("💾 Export Route Data"):
                            # Create route summary
                            route_data = {
                                'Step': [f"Step {i+1}" for i in range(len(shortest_path)-1)],
                                'From': shortest_path[:-1],
                                'To': shortest_path[1:],
                                'Distance_km': [G[shortest_path[i]][shortest_path[i+1]]['weight'] for i in range(len(shortest_path)-1)]
                            }
                            
                            route_df = pd.DataFrame(route_data)
                            route_df['Cumulative_Distance'] = route_df['Distance_km'].cumsum()
                            
                            st.write("**Route Summary Table:**")
                            st.dataframe(route_df)
                            
                            # Download button
                            csv = route_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Route Summary (CSV)",
                                data=csv,
                                file_name=f"route_{source}_to_{target}.csv",
                                mime="text/csv"
                            )
                    
                    else:
                        # No path found
                        st.markdown(f'''
                        <div class="error-message">
                            <strong>❌ No Route Found!</strong><br>
                            There is no path between <strong>{source}</strong> and <strong>{target}</strong>.
                            Check if the network is properly connected.
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Still show the map for network visualization
                        st.subheader("🗺️ Network Overview")
                        folium_map = create_folium_map(G, [], source, target, map_theme)
                        st_folium(folium_map, width=700, height=500)
        
        else:
            st.error("❌ Could not load graph data. Please check your CSV format.")
            st.write("**Expected CSV format:**")
            st.code("""source,target,weight
Farm,VillageA,4
Farm,VillageB,2
VillageA,Market,5""")
    
    else:
        st.info("👆 Please select a data source from the sidebar to begin route optimization.")
        
        # Show sample data format
        st.subheader("📋 Expected Data Format")
        st.write("Your CSV file should contain the following columns:")
        
        sample_data = pd.DataFrame({
            'source': ['Farm', 'Farm', 'VillageA', 'VillageB'],
            'target': ['VillageA', 'VillageB', 'Market', 'Market'],
            'weight': [4, 2, 5, 3]
        })
        
        st.dataframe(sample_data)
        
        st.write("""
        **Column descriptions:**
        - **source**: Starting location name
        - **target**: Ending location name  
        - **weight**: Distance between locations (in kilometers)
        """)


if __name__ == "__main__":
    main()