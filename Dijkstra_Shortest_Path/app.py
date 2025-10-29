# ...existing code...
#!/usr/bin/env python3
"""
Farm-to-Market Route Optimizer - Streamlit Web Application

Interactive web interface for calculating optimal routes using Dijkstra's algorithm
with real-time map visualization powered by Folium.
"""

from reroute import (
    is_road_blocked_due_to_weather,
    calculate_alternate_route,
    convert_path_to_edges,
    SIMULATED_BLOCKED_ROADS,
    get_weather_affected_edges,
    adjust_eta_for_weather 
)
 
import streamlit as st # pyright: ignore[reportMissingImports]
import networkx as nx # pyright: ignore[reportMissingModuleSource]
import folium # pyright: ignore[reportMissingImports]
from streamlit_folium import st_folium # pyright: ignore[reportMissingImports]

import streamlit as st # pyright: ignore[reportMissingImports]
import networkx as nx # pyright: ignore[reportMissingModuleSource]
import pandas as pd
import folium # pyright: ignore[reportMissingImports]
from streamlit_folium import st_folium # pyright: ignore[reportMissingImports]
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
import io
import hashlib
#from cost_estimation import calculate_cost

# Import validation module
from validation import RouteDataValidator, validate_route_data, display_validation_help, display_validation_help_simple


# Configure Streamlit page
st.set_page_config(
    page_title="  Farm-to-Market Route Optimizer",
    page_icon=" ",
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
    """
    try:
        G = nx.Graph()
        required_cols = ['source', 'target', 'weight']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {', '.join(required_cols)}")
            return None

        st.write(f"  **Debug Info:** Processing {len(df)} rows from CSV")

        skipped_rows = 0
        for _, row in df.iterrows():
            if pd.isna(row['source']) or pd.isna(row['target']) or pd.isna(row['weight']):
                skipped_rows += 1
                continue
            G.add_edge(str(row['source']).strip(), str(row['target']).strip(), weight=float(row['weight']))

        st.write(f"  **Graph built:** {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        if skipped_rows > 0:
            st.warning(f"   Skipped {skipped_rows} rows with missing data")

        return G

    except Exception as e:
        st.error(f"Error loading graph: {str(e)}")
        import traceback
        st.error(f"  **Stack trace:** {traceback.format_exc()}")
        return None


def calculate_shortest_path(G: nx.Graph, source: str, target: str) -> Tuple[List[str], float]:
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
    if distance == 0 or speed <= 0:
        return 0, 0
    total_hours = distance / speed
    hours = int(total_hours)
    minutes = int((total_hours - hours) * 60)
    return hours, minutes


@st.cache_data
def generate_node_coordinates(nodes: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Generate deterministic coordinates for nodes using hash-based offsets.
    Cached to keep coordinates stable across Streamlit reruns.
    """
    base_lat, base_lon = 40.7128, -74.0060  # NYC center
    coordinates = {}
    
    # Sort nodes for consistent ordering
    for i, node in enumerate(sorted(nodes)):
        # Use node name hash for stable random-like offsets
        node_hash = int(hashlib.md5(node.encode()).hexdigest(), 16)
        
        # Deterministic offsets based on hash
        lat_offset = ((node_hash & 0xFFF) / 0xFFF - 0.5) * 0.02
        lon_offset = ((node_hash >> 12 & 0xFFF) / 0xFFF - 0.5) * 0.02
        
        # Grid placement plus hash-based offset
        grid_lat = (i % 3 - 1) * 0.02
        grid_lon = (i // 3 - 1) * 0.02
        
        final_lat = grid_lat + lat_offset
        final_lon = grid_lon + lon_offset
        
        # Semantic adjustments
        if 'farm' in node.lower():
            final_lat -= 0.01
        elif 'market' in node.lower():
            final_lat += 0.01
            
        coordinates[node] = (base_lat + final_lat, base_lon + final_lon)
    
    return coordinates


def create_folium_map(G: nx.Graph, shortest_path: List[str], source: str, target: str,
                     map_theme: str = "OpenStreetMap") -> folium.Map:
    coordinates = generate_node_coordinates(list(G.nodes()))
    if coordinates:
        center_lat = np.mean([coord[0] for coord in coordinates.values()])
        center_lon = np.mean([coord[1] for coord in coordinates.values()])
    else:
        center_lat, center_lon = 40.7128, -74.0060

    tile_options = {
        "OpenStreetMap": "OpenStreetMap",
        "Stamen Terrain": "Stamen Terrain",
        "CartoDB Positron": "cartodb positron",
        "CartoDB Dark": "cartodb dark_matter"
    }

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles=tile_options.get(map_theme, "OpenStreetMap")
    )

    for edge in G.edges(data=True):
        node1, node2 = edge[0], edge[1]
        if node1 in coordinates and node2 in coordinates:
            folium.PolyLine(
                locations=[coordinates[node1], coordinates[node2]],
                color='lightgray',
                weight=2,
                opacity=0.6
            ).add_to(m)

    if len(shortest_path) > 1:
        path_coordinates = [coordinates[node] for node in shortest_path if node in coordinates]
        if len(path_coordinates) > 1:
            folium.PolyLine(
                locations=path_coordinates,
                color='red',
                weight=5,
                opacity=0.8,
                popup=f"Shortest Path: {'   '.join(shortest_path)}"
            ).add_to(m)

    for node in G.nodes():
        if node in coordinates:
            lat, lon = coordinates[node]
            if node == source:
                color = 'green'
                icon = 'play'
                popup_text = f"  START: {node}"
            elif node == target:
                color = 'red'
                icon = 'stop'
                popup_text = f"  DESTINATION: {node}"
            elif node in shortest_path:
                color = 'orange'
                icon = 'info-sign'
                popup_text = f"   PATH: {node}"
            else:
                color = 'lightblue'
                icon = 'info-sign'
                popup_text = f"  {node}"

            folium.Marker(
                location=[lat, lon],
                popup=popup_text,
                tooltip=node,
                icon=folium.Icon(color=color, icon=icon, prefix='glyphicon')
            ).add_to(m)

    return m


def format_path_display(path: List[str]) -> str:
    if not path:
        return "No path found"
    if len(path) == 1:
        return f"  {path[0]}"
    formatted_path = []
    for i, node in enumerate(path):
        if i == 0:
            formatted_path.append(f"  {node}")
        elif i == len(path) - 1:
            formatted_path.append(f"  {node}")
        else:
            formatted_path.append(f"   {node}")
    return "   ".join(formatted_path)


def networkx_to_adj_list(G: nx.Graph) -> Dict[str, List[Tuple[str, float]]]:
    adj = {}
    for u in G.nodes():
        adj[str(u)] = []
        for v in G.neighbors(u):
            weight = G[u][v].get('weight', 1.0)
            adj[str(u)].append((str(v), float(weight)))
    return adj


def main():
    st.markdown('<h1 class="main-header">  Farm-to-Market Route Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("**Find the optimal route from farm to market using Dijkstra's Algorithm with interactive map visualization**")

    with st.sidebar.expander("  **Data Validation Help**"):
        display_validation_help_simple()

    st.sidebar.header("   Route Configuration")
    st.sidebar.subheader("  Data Source")
    use_default = st.sidebar.radio(
        "Choose data source:",
        ["Use default sample data", "Upload custom CSV file"]
    )

    validator = RouteDataValidator()

    if use_default == "Upload custom CSV file":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="CSV should have columns: source, target, weight (optionally: lat, lon)"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"  Uploaded: {len(df)} edges")
                st.subheader("  Data Validation")
                if not validator.validate_route_data(df):
                    st.error("  **Data validation failed.** Please fix the issues above and try again.")
                    st.stop()
            except Exception as e:
                st.sidebar.error(f"  Error reading file: {str(e)}")
                st.error(f"  **File Reading Error:** {str(e)}")
                st.info("  Please ensure your file is a valid CSV format.")
                st.stop()
        else:
            st.info("  Please upload a CSV file to begin route optimization.")
            st.subheader("  Expected Data Format")
            display_validation_help()
            st.stop()
    else:
        try:
            df = pd.read_csv("sample_data.csv")
            st.sidebar.success(f"  Default data: {len(df)} edges")
            st.subheader("  Data Validation")
            if not validator.validate_data_only(df):
                st.error("  **Default data validation failed.** Please check sample_data.csv file.")
                st.stop()
        except FileNotFoundError:
            st.sidebar.error("  Default sample_data.csv not found!")
            st.error("  **Default data file missing:** sample_data.csv not found!")
            st.info("  Please upload a custom CSV file instead.")
            st.stop()
        except Exception as e:
            st.sidebar.error(f"  Error loading default data: {str(e)}")
            st.error(f"  **Default Data Error:** {str(e)}")
            st.stop()

    with st.spinner("  Building network graph..."):
        G = load_graph(df)

    if G is not None and G.number_of_nodes() > 0:
        st.sidebar.subheader("  Network Info")
        validation_report = validator.get_validation_report()

        connectivity_status = "  Connected" if nx.is_connected(G) else "   Disconnected"
        st.sidebar.info(f"""
        **Nodes:** {G.number_of_nodes()}  
        **Edges:** {G.number_of_edges()}  
        **Connected:** {connectivity_status}
        """)

        if validation_report['warnings']:
            with st.sidebar.expander("   **Validation Warnings**"):
                for warning in validation_report['warnings']:
                    st.write(f"  {warning}")

        st.sidebar.subheader("   Route Selection")
        nodes = sorted(list(G.nodes()))
        default_source = next((node for node in nodes if 'farm' in node.lower()), nodes[0])
        source_idx = nodes.index(default_source) if default_source in nodes else 0
        source = st.sidebar.selectbox("  Start Location:", nodes, index=source_idx, help="Select the starting point for your route")
        default_target = next((node for node in nodes if 'market' in node.lower()), nodes[-1])
        target_idx = nodes.index(default_target) if default_target in nodes else len(nodes)-1
        target = st.sidebar.selectbox("  Destination:", nodes, index=target_idx, help="Select the destination for your route")

        if not validator.validate_user_inputs(G, source, target):
            st.stop()

        st.sidebar.subheader("  Travel Settings")
        speed = st.sidebar.slider("  Average Speed (km/h):", min_value=10, max_value=100, value=40, step=5, help="Average travel speed for ETA calculation")

        st.sidebar.subheader("   Map Settings")
        map_theme = st.sidebar.selectbox("  Map Theme:", ["OpenStreetMap", "Stamen Terrain", "CartoDB Positron", "CartoDB Dark"], help="Choose the visual style for the map")

        if st.sidebar.button("  Calculate Optimal Route", type="primary"):
            if source == target:
                st.sidebar.warning("   Start and destination are the same!")
                st.warning("   **Invalid Route:** Start and destination locations are identical.")
                st.info("  Please select different start and destination locations.")
            else:
                progress_placeholder = st.empty()
                progress_placeholder.info("  Calculating shortest path...")
                with st.spinner("Computing optimal route..."):
                    time.sleep(0.5)
                    shortest_path, distance = calculate_shortest_path(G, source, target)
                progress_placeholder.empty()

                if not shortest_path or distance == float('inf'):
                    st.error("  **No Route Found!**")
                    st.write(f"There is no path between **{source}** and **{target}**.")
                    if not nx.is_connected(G):
                        st.info("  **Possible Cause:** The network has disconnected components.")
                        components = list(nx.connected_components(G))
                        st.write(f"Your network has {len(components)} separate groups of connected locations.")
                    st.subheader("   Network Overview")
                    folium_map = create_folium_map(G, [], source, target, map_theme)
                    st_folium(folium_map, width=700, height=500)
                    st.stop()

                if shortest_path and distance != float('inf'):
                    st.session_state['shortest_path'] = shortest_path
                    st.session_state['distance'] = distance
                    st.session_state['source'] = source
                    st.session_state['target'] = target
                    st.session_state['map_theme'] = map_theme

                    st.markdown(f'''
                    <div class="success-message">
                        <strong>  Route Successfully Calculated!</strong><br>
                        Optimal path found from <strong>{source}</strong> to <strong>{target}</strong>
                    </div>
                    ''', unsafe_allow_html=True)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(label="  Total Distance", value=f"{distance:.1f} km")
                    with col2:
                        hours, minutes = calculate_eta(distance, speed)
                        eta_text = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                        st.metric(label="   Estimated Time", value=eta_text)
                    with col3:
                        st.metric(label="  Number of Stops", value=len(shortest_path) - 1)
                    with col4:
                        st.metric(label="  Average Speed", value=f"{speed} km/h")

                    st.markdown(f'''
                    <div class="path-container">
                        <h3>   Optimal Route Path</h3>
                        <p style="font-size: 1.2rem; font-weight: bold;">{format_path_display(shortest_path)}</p>
                    </div>
                    ''', unsafe_allow_html=True)

                    if len(shortest_path) > 1:
                        st.subheader("  Step-by-Step Directions")
                        for i in range(len(shortest_path) - 1):
                            current = shortest_path[i]
                            next_node = shortest_path[i + 1]
                            edge_weight = G[current][next_node]['weight']
                            step_hours, step_minutes = calculate_eta(edge_weight, speed)
                            step_eta = f"{step_hours}h {step_minutes}m" if step_hours > 0 else f"{step_minutes}m"
                            st.write(f"**Step {i+1}:** {current}   {next_node} ({edge_weight:.1f} km, ~{step_eta})")

                    st.subheader("   Interactive Route Map")
                    st.write("Explore the route on the interactive map below:")
                    with st.spinner("   Generating map..."):
                        folium_map = create_folium_map(G, shortest_path, source, target, map_theme)
                    map_data = st_folium(folium_map, width=700, height=500, returned_objects=["last_object_clicked"])

                    st.markdown("""
                    **Map Legend:**
                    -   **Green Marker**: Start location
                    -   **Red Marker**: Destination
                    -   **Orange Markers**: Route waypoints
                    -   **Red Line**: Optimal route
                    -   **Gray Lines**: Alternative routes
                    """)

                    with st.expander("  Advanced Route Analysis"):
                        st.write("**Alternative Routes Analysis:**")
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

                    with st.expander("  Export Route Data"):
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
                        csv = route_df.to_csv(index=False)
                        st.download_button(label="  Download Route Summary (CSV)", data=csv, file_name=f"route_{source}_to_{target}.csv", mime="text/csv")

                else:
                    st.markdown(f'''
                    <div class="error-message">
                        <strong>  No Route Found!</strong><br>
                        There is no path between <strong>{source}</strong> and <strong>{target}</strong>.
                        Check if the network is properly connected.
                    </div>
                    ''', unsafe_allow_html=True)
                    st.subheader("   Network Overview")
                    folium_map = create_folium_map(G, [], source, target, map_theme)
                    st_folium(folium_map, width=700, height=500)




       # ...existing code...

        # ☁️ Weather-based alternate route
        if st.sidebar.button("☁️  Reroute by Weather", type="secondary"):
            st.info("  Checking for weather-related blockages...")
            if 'shortest_path' not in st.session_state:
                st.warning("Please calculate a route first (press 'Calculate Optimal Route').")
            else:
                current_path = st.session_state['shortest_path']
                current_source = st.session_state['source']
                current_target = st.session_state['target']
                current_map_theme = st.session_state.get('map_theme', map_theme)

                path_edges = convert_path_to_edges(current_path)

                # Use get_weather_affected_edges to fetch affected edges and severities
                affected = get_weather_affected_edges(path_edges, severity_threshold=2, use_simulation=True)

                if affected:
                # Show each blocked edge and its reason
                    blocked_msgs = [
                        f"{edge} ({info.get('label', 'Unknown')})"
                        for edge, info in affected.items()
                    ]
                    # Persist blocked messages so they remain after reruns
                    st.session_state['blocked_msgs'] = blocked_msgs

                    st.warning(
                        f"⚠️ Blocked segments due to weather:\n\n- " +
                        "\n- ".join(blocked_msgs)
                    )   
    

                    adj = networkx_to_adj_list(G)

                    alternate_path, alt_distance = calculate_alternate_route(
                        adj,
                        current_source,
                        current_target,
                        affected.keys()
                    )

                    if alternate_path and alt_distance != float('inf'):
                        severities = [v['severity'] for v in affected.values()]
                        adj_hours, adj_minutes = adjust_eta_for_weather(alt_distance, speed, severities)

                        # Persist alternate result so it survives re-runs
                        st.session_state['alternate_path'] = alternate_path
                        st.session_state['alternate_distance'] = alt_distance
                        st.session_state['alternate_severities'] = severities
                        st.session_state['alternate_eta'] = (adj_hours, adj_minutes)
                        st.session_state['alternate_source'] = current_source
                        st.session_state['alternate_target'] = current_target
                        st.session_state['alternate_map_theme'] = current_map_theme

                        st.success("✅  Alternate route found avoiding blocked roads!")
                        st.write(f"**New Route:** {' ➜ '.join(alternate_path)}")
                        st.write(f"**New Distance:** {alt_distance:.2f} km")
                        st.write(f"**Adjusted ETA:** {adj_hours}h {adj_minutes}m (based on weather)")

                        folium_map = create_folium_map(G, alternate_path, current_source, current_target, current_map_theme)
                        st_folium(folium_map, width=700, height=500)

                    else:
                        st.error("❌  No alternate path available at the moment.")
                else:
                    st.success("✅  All roads are clear. Proceed with your current route.")




        
        if st.session_state.get('blocked_msgs'):
            st.markdown("<hr>", unsafe_allow_html=True)
            st.warning(
                "⚠️ Blocked segments due to weather:\n\n- " +
                "\n- ".join(st.session_state['blocked_msgs'])
            )
            if st.button("Clear blocked messages"):
                st.session_state.pop('blocked_msgs', None)



        # Persistently show last computed alternate route (so it doesn't disappear after a rerun)
        if 'alternate_path' in st.session_state:
            try:
                ap = st.session_state['alternate_path']
                ad = st.session_state.get('alternate_distance', None)
                aeta = st.session_state.get('alternate_eta', None)
                a_src = st.session_state.get('alternate_source', source)
                a_tgt = st.session_state.get('alternate_target', target)
                a_theme = st.session_state.get('alternate_map_theme', map_theme)

                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader("  Last Alternate Route (persisted)")
                st.write(f"**Route:** {' ➜ '.join(ap)}")
                if ad is not None:
                    st.write(f"**Distance:** {ad:.2f} km")
                if aeta:
                    st.write(f"**Adjusted ETA:** {aeta[0]}h {aeta[1]}m")

                folium_map = create_folium_map(G, ap, a_src, a_tgt, a_theme)
                st_folium(folium_map, width=700, height=500)
            except Exception:
                # don't break the app on unexpected session state contents
                pass

    else:
        st.error("  **No valid data loaded**")
        st.info("  Please check the data source and try again.")
        st.subheader("  Expected Data Format")
        display_validation_help()

    # 💰 Cost Estimation Section
    st.markdown("---")
    st.subheader("💰 Cost Estimation")

    with st.expander("Estimate Delivery Cost"):
        st.write("Compare solo and pooled delivery costs based on route distance.")
        
        base_rate = st.number_input("Enter base rate (₹ per km):", min_value=1, value=10)
        num_farmers = st.number_input("Number of farmers sharing vehicle:", min_value=1, value=1)
        
        if st.button("Calculate Delivery Cost"):
            solo_cost, pooled_cost = calculate_cost(distance, base_rate, num_farmers)
            st.success("✅ Cost estimation complete!")
            st.write(f"**Distance:** {distance:.2f} km")
            st.write(f"**Base Rate:** ₹{base_rate}/km")
            st.write(f"**Solo Delivery Cost:** ₹{solo_cost:.2f}")
            st.write(f"**Pooled Delivery Cost (per farmer):** ₹{pooled_cost:.2f}")
            st.write(f"**Savings per farmer:** ₹{solo_cost - pooled_cost:.2f}")
 


import pandas as pd
import math
import os
from sklearn.cluster import KMeans

# ----------------------------
# CONFIGURATION
# ----------------------------
st.set_page_config(page_title="Farm-to-Market Optimizer", layout="wide")
st.title("🌾 Farm-to-Market Management System")

# ----------------------------
# COMMON SETTINGS
# ----------------------------
FILE_NAME = "farms.csv"
MARKET = {"lat": 26.92, "lon": 81.05}

# ==========================
# SECTION 1 — FARM POOLING
# ==========================

def load_farm_data():
    if os.path.exists(FILE_NAME):
        return pd.read_csv(FILE_NAME)
    else:
        df = pd.DataFrame(columns=["farm_id", "lat", "lon", "produce_kg"])
        df.to_csv(FILE_NAME, index=False)
        return df

def save_farm_data(df):
    df.to_csv(FILE_NAME, index=False)

def distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # km approx

def assign_vehicles(cluster_farms, vehicle_capacity=3):
    vehicles = []
    farms_list = list(cluster_farms.index)
    for i in range(0, len(farms_list), vehicle_capacity):
        vehicles.append(farms_list[i:i+vehicle_capacity])
    return vehicles

def run_pooling(df, vehicle_capacity=3):
    if df.empty:
        return None, "No farm data available!"
    
    n_clusters = 2 if len(df) >= 2 else 1
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    coords = df[['lat', 'lon']].values
    df['cluster'] = kmeans.fit_predict(coords)
    
    results = []
    base_cost_per_km = 10
    fixed_cost = 50
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_farms = df[df['cluster'] == cluster_id]
        vehicles = assign_vehicles(cluster_farms, vehicle_capacity)
        for v_num, vehicle_farms in enumerate(vehicles, start=1):
            total_distance = 0
            for idx in vehicle_farms:
                row = df.loc[idx]
                total_distance += distance(row['lat'], row['lon'], MARKET['lat'], MARKET['lon'])
            avg_distance = total_distance / len(vehicle_farms)
            total_cost = fixed_cost + avg_distance * 2 * base_cost_per_km
            per_farmer_cost = total_cost / len(vehicle_farms)
            farm_ids = [df.loc[idx, 'farm_id'] for idx in vehicle_farms]
            results.append({
                "cluster": cluster_id,
                "vehicle": f"C{cluster_id}_V{v_num}",
                "farms": farm_ids,
                "avg_distance": round(avg_distance, 2),
                "total_cost": round(total_cost, 2),
                "per_farmer_cost": round(per_farmer_cost, 2)
            })
    return results, None

# ==========================
# SECTION 2 — ROUTE OPTIMIZATION
# ==========================

def create_graph(data):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    return G

def create_route_map(G, path):
    m = folium.Map(location=[26.9, 81.05], zoom_start=7)
    for (u, v, d) in G.edges(data=True):
        u_loc = (26.9 + hash(u)%20*0.01, 81.05 + hash(u)%20*0.01)
        v_loc = (26.9 + hash(v)%20*0.01, 81.05 + hash(v)%20*0.01)
        folium.PolyLine([u_loc, v_loc], color="gray", weight=2).add_to(m)
    
    # Draw shortest path
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        u_loc = (26.9 + hash(u)%20*0.01, 81.05 + hash(u)%20*0.01)
        v_loc = (26.9 + hash(v)%20*0.01, 81.05 + hash(v)%20*0.01)
        folium.PolyLine([u_loc, v_loc], color="red", weight=4).add_to(m)
        folium.Marker(u_loc, tooltip=u).add_to(m)
    folium.Marker(v_loc, tooltip=path[-1], icon=folium.Icon(color="green")).add_to(m)
    return m

# ==========================
# SIDEBAR — NAVIGATION
# ==========================
st.sidebar.header("🧭 Select a Section")
section = st.sidebar.radio("Choose one:", ["Farm Pooling & Vehicle Scheduling", "Route Optimization"])

# ==========================
# FARM POOLING SECTION
# ==========================
if section == "Farm Pooling & Vehicle Scheduling":
    st.header("🚜 Farm Pooling and Vehicle Scheduling")
    df = load_farm_data()

    st.subheader("📋 Current Farms")
    st.dataframe(df)

    with st.expander("➕ Add a New Farm"):
        farm_id = st.text_input("Farm ID (e.g., F6)")
        lat = st.number_input("Latitude")
        lon = st.number_input("Longitude")
        produce_kg = st.number_input("Produce (kg)")
        if st.button("Add Farm"):
            new_row = {"farm_id": farm_id, "lat": lat, "lon": lon, "produce_kg": produce_kg}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_farm_data(df)
            st.success(f"Farm {farm_id} added successfully!")

    with st.expander("❌ Delete a Farm"):
        farm_id_del = st.text_input("Enter Farm ID to delete")
        if st.button("Delete Farm"):
            if farm_id_del in df['farm_id'].values:
                df = df[df['farm_id'] != farm_id_del]
                save_farm_data(df)
                st.warning(f"Farm {farm_id_del} deleted successfully!")
            else:
                st.error("Farm not found!")

    if st.button("🚚 Run Pooling & Vehicle Scheduling"):
        results, error = run_pooling(df)
        if error:
            st.error(error)
        else:
            st.subheader("📦 Pooling & Vehicle Results")
            for r in results:
                st.markdown(f"""
                **Vehicle:** {r['vehicle']}  
                **Farms:** {', '.join(r['farms'])}  
                **Average Distance:** {r['avg_distance']} km  
                **Total Cost:** ₹{r['total_cost']}  
                **Per Farmer Cost:** ₹{r['per_farmer_cost']}
                """)
            st.success("✅ Pooling and cost calculation complete!")

# ==========================
# ROUTE OPTIMIZATION SECTION
# ==========================
elif section == "Route Optimization":
    st.header("🗺️ Route Optimization (Shortest Path)")

    st.info("Upload a CSV with columns: source, target, weight")

    uploaded = st.file_uploader("Upload Road Network CSV", type=['csv'])
    if uploaded:
        data = pd.read_csv(uploaded)
        st.dataframe(data.head())

        G = create_graph(data)
        nodes = list(G.nodes)

        source = st.selectbox("Select Start Point", nodes)
        target = st.selectbox("Select Destination", nodes)

        if st.button("🔍 Find Shortest Route"):
            try:
                path = nx.dijkstra_path(G, source, target, weight='weight')
                distance = nx.dijkstra_path_length(G, source, target, weight='weight')
                st.success(f"Shortest Path: {' ➜ '.join(path)}")
                st.info(f"Total Distance: {round(distance, 2)} km")

                route_map = create_route_map(G, path)
                st_folium(route_map, width=700, height=500)
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
# ...existing code...