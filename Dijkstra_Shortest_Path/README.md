# Farm-to-Market Route Optimization

A comprehensive Python application implementing Dijkstra's Algorithm with an interactive Streamlit web interface, Folium map visualization, and robust data validation for finding optimal routes from farms to markets.

## Overview

This project demonstrates practical applications of graph algorithms in logistics and transportation. It provides both a command-line interface and a modern web application for route optimization, featuring real-time interactive maps, comprehensive route analysis, and professional-grade data validation.

### What is Dijkstra's Algorithm?

Dijkstra's Algorithm is a graph search algorithm that finds the shortest path between nodes in a weighted graph. It works by:

1. **Initialization**: Set the distance to the source node as 0 and all other nodes as infinity
2. **Selection**: Select the unvisited node with the smallest known distance
3. **Relaxation**: Update the distances to all neighbors of the current node
4. **Repetition**: Repeat until all nodes are visited or the destination is reached

### Farm-to-Market Application

In our context:
- **Nodes** represent locations (Farm, Villages, Market)
- **Edges** represent roads with distances as weights
- **Algorithm** finds the shortest route from Farm to Market

## Project Structure

```
Dijkstra_Shortest_Path/
├── main.py                      # Command-line implementation
├── app.py                       # Streamlit web application
├── validation.py                # Data validation module
├── sample_data.csv              # Sample network data
├── requirements.txt             # Python dependencies
├── README.md                    # This documentation
└── dijkstra_shortest_path.png   # Generated visualization
```

## Features

### Command-Line Application (`main.py`)
- **Graph Loading**: Reads network data from CSV files
- **Shortest Path Calculation**: Uses NetworkX's Dijkstra implementation
- **Static Visualization**: Creates matplotlib plots showing the network and optimal route
- **Comprehensive Analysis**: Provides graph statistics and alternative route information
- **Error Handling**: Robust error handling for missing files or invalid data

### Web Application (`app.py`)
- **Interactive UI**: Modern Streamlit interface with sidebar controls
- **File Upload**: Support for custom CSV data upload
- **Real-time Calculation**: Live route optimization with progress indicators
- **Interactive Maps**: Folium-powered maps with clickable markers and route highlighting
- **ETA Calculation**: Travel time estimation with configurable speed settings
- **Multiple Map Themes**: Choice of map visual styles
- **Export Functionality**: Download route summaries as CSV files
- **Responsive Design**: Works on desktop and mobile devices

### Data Validation System (`validation.py`)
- **CSV Structure Validation**: Ensures required columns and proper format
- **Data Type Checking**: Validates numeric weights and coordinate ranges
- **Geographic Validation**: Verifies latitude/longitude bounds when present
- **Graph Consistency**: Checks connectivity and identifies isolated nodes
- **User Input Safety**: Validates route selections and prevents invalid computations
- **Real-time Feedback**: Streamlit integration with color-coded messages

## Data Validation

The application includes a comprehensive data validation system that ensures data quality and prevents errors before computation.

### Validation Rules

#### 🔍 **CSV Structure Validation**
- **Required Columns**: `source`, `target`, `weight`
- **Optional Columns**: `lat`, `lon` (coordinates) or `source_lat`, `source_lon`, `target_lat`, `target_lon`
- **No Duplicate Columns**: Column names must be unique
- **Non-Empty Data**: CSV must contain actual data rows

#### 🔢 **Data Type Validation**
- **Weights**: Must be positive numeric values (> 0)
- **Coordinates**: Must be valid decimal numbers within geographic bounds
  - **Latitude**: -90 to +90 degrees
  - **Longitude**: -180 to +180 degrees
- **Location Names**: Non-empty text strings without leading/trailing whitespace

#### 🌐 **Graph Consistency Checks**
- **Self-loops**: Routes from a location to itself are detected and ignored
- **Connectivity**: Warns if graph has disconnected components
- **Isolated Nodes**: Identifies nodes with no connections
- **Component Analysis**: Reports the number and size of connected components

#### 🎯 **User Input Validation**
- **Node Existence**: Selected start/destination must exist in the dataset
- **Route Possibility**: Validates that a path exists between selected locations
- **Input Differences**: Prevents identical start and destination selections

### Validation Feedback

The system provides real-time feedback using color-coded Streamlit messages:

- **✅ Success Messages (Green)**: Validation passed, ready to proceed
- **⚠️ Warning Messages (Yellow)**: Non-critical issues that don't block computation
- **❌ Error Messages (Red)**: Critical issues that prevent route calculation
- **ℹ️ Info Messages (Blue)**: Helpful guidance and tips

### Common Validation Issues & Solutions

#### ❌ **Missing Required Columns**
- **Issue**: CSV doesn't have `source`, `target`, or `weight` columns
- **Solution**: Ensure your CSV includes all three required columns with exact names

#### ❌ **Non-Positive Weights**
- **Issue**: Weight values are zero, negative, or non-numeric
- **Solution**: All distance values must be positive numbers greater than 0

#### ❌ **Invalid Coordinates**
- **Issue**: Latitude/longitude values are outside valid ranges
- **Solution**: Check coordinate bounds (lat: -90 to 90, lon: -180 to 180)

#### ⚠️ **Disconnected Graph**
- **Issue**: Some locations are not reachable from others
- **Solution**: Add connecting routes to ensure network connectivity

#### ❌ **Node Not Found**
- **Issue**: Selected start/destination doesn't exist in the dataset
- **Solution**: Choose locations from the provided dropdown menus

### Sample Valid Data Formats

#### **Basic Format (Required Columns Only)**
```csv
source,target,weight
Farm_A,Village_X,5.2
Farm_A,Village_Y,8.1
Village_X,Market_1,2.4
Village_Y,Market_1,4.9
```

#### **Extended Format (With Coordinates)**
```csv
source,target,weight,lat,lon
Farm_A,Village_X,5.2,40.7128,-74.0060
Farm_A,Village_Y,8.1,40.7200,-74.0100
Village_X,Market_1,2.4,40.7300,-74.0150
Village_Y,Market_1,4.9,40.7180,-74.0080
```

#### **Separate Coordinates Format**
```csv
source,target,weight,source_lat,source_lon,target_lat,target_lon
Farm_A,Village_X,5.2,40.7128,-74.0060,40.7200,-74.0100
Village_X,Market_1,2.4,40.7200,-74.0100,40.7300,-74.0150
```

### Data Summary Features

After successful validation, the system displays:

- **📊 Data Summary**: Total edges, unique nodes, distance statistics
- **🔗 Network Metrics**: Connectivity status and component analysis
- **⚠️ Warning Review**: Non-critical issues that were detected
- **📋 Validation Report**: Comprehensive validation results

## Installation

1. **Clone or download** this project to your local machine

2. **Install dependencies** using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation** by checking if packages are installed:
   ```bash
   python -c "import streamlit, networkx, folium; print('All packages installed successfully!')"
   ```

## Usage

### Web Application (Recommended)

**Start the Streamlit app:**
```bash
streamlit run app.py
```

**Features:**
- 🗺️ **Interactive route planning** with dropdown menus
- 📁 **File upload** for custom network data
- 🎨 **Multiple map themes** (OpenStreetMap, Terrain, etc.)
- ⚡ **Real-time ETA calculation** with adjustable speed
- 📊 **Detailed route analysis** with step-by-step directions
- 💾 **Export capabilities** for route data

### Command-Line Application

**Run the traditional script:**
```bash
python main.py
```

**Features:**
- 📈 **Static visualization** saved as PNG
- 📋 **Comprehensive text output** with route details
- 🔍 **Network analysis** and connectivity information

## Data Format

The CSV file should contain three columns:
- `source`: Starting location name
- `target`: Ending location name  
- `weight`: Distance between locations (in km)

### Sample Data Structure

```csv
source,target,weight
Farm,VillageA,4
Farm,VillageB,2
VillageA,Market,8
VillageB,VillageC,1
VillageC,Market,6
```

## Web Application Guide

### 1. Getting Started
1. Launch the app with `streamlit run app.py`
2. Open your browser to the displayed URL (usually `http://localhost:8501`)
3. Use the sidebar to configure your route

### 2. Data Sources
- **Default Data**: Uses the included `sample_data.csv`
- **Custom Upload**: Upload your own CSV file with network data

### 3. Route Configuration
- **Start Location**: Choose from available nodes (auto-detects farms)
- **Destination**: Select target location (auto-detects markets)
- **Speed Setting**: Adjust average travel speed for ETA calculation

### 4. Map Features
- **Interactive Markers**: Click on locations for details
- **Route Highlighting**: Optimal path shown in red
- **Multiple Themes**: Choose from 4 different map styles
- **Zoom and Pan**: Full map navigation capabilities

### 5. Results Analysis
- **Distance Metrics**: Total distance and estimated travel time
- **Step-by-Step Directions**: Detailed route breakdown
- **Alternative Analysis**: Distances to all locations
- **Export Options**: Download route data as CSV

## Customization

### Modify the Dataset

1. **Edit `sample_data.csv`** to add/remove locations or change distances
2. **Ensure connectivity**: Make sure there's a path from source to destination
3. **Use realistic distances**: Keep weights positive and reasonable

### Change Default Locations

**In `app.py`, modify the auto-detection logic:**
```python
# Source selection
default_source = next((node for node in nodes if 'your_term' in node.lower()), nodes[0])

# Target selection  
default_target = next((node for node in nodes if 'your_term' in node.lower()), nodes[-1])
```

### Add Custom Map Themes

**Extend the tile options in `create_folium_map()`:**
```python
tile_options = {
    "OpenStreetMap": "OpenStreetMap",
    "Your Custom Theme": "your_tile_url",
    # ... existing themes
}
```

## Technical Details

### Dependencies

- **Streamlit**: Web application framework
- **NetworkX**: Graph creation and algorithm implementation
- **Folium**: Interactive map visualization
- **streamlit-folium**: Streamlit-Folium integration
- **Pandas**: CSV data loading and manipulation
- **NumPy**: Numerical operations support

### Algorithm Complexity

- **Time Complexity**: O((V + E) log V) where V is vertices and E is edges
- **Space Complexity**: O(V) for storing distances and paths

### Map Visualization

- **Coordinate Generation**: Automatic coordinate assignment for node positioning
- **Interactive Elements**: Clickable markers with popup information
- **Route Highlighting**: Visual path differentiation with colors
- **Responsive Design**: Adapts to different screen sizes

## Screenshots

*Note: Screenshots would be placed here showing:*
- Main application interface
- Interactive map with route highlighting
- Sidebar configuration options
- Results display with metrics

## Extensions

### 1. Real-World Integration
- **GPS Coordinates**: Use actual latitude/longitude data
- **Road Network APIs**: Integrate with OpenStreetMap or Google Maps
- **Traffic Data**: Include real-time traffic conditions

### 2. Advanced Features
- **Multi-Objective Optimization**: Consider cost, time, and distance
- **Vehicle Constraints**: Account for vehicle capacity and fuel efficiency
- **Dynamic Routing**: Real-time route updates based on conditions

### 3. Business Applications
- **Fleet Management**: Optimize multiple vehicle routes
- **Supply Chain**: Multi-depot and multi-destination routing
- **Cost Analysis**: Include fuel costs and tolls in optimization

## Troubleshooting

### Common Issues

1. **"No path found"**: Ensure your graph is connected
2. **"Node not found"**: Check spelling in location names
3. **Import errors**: Run `pip install -r requirements.txt`
4. **Streamlit won't start**: Check if port 8501 is available
5. **Map not displaying**: Verify internet connection for map tiles

### Performance Tips

- **Large Networks**: Use data caching for better performance
- **Slow Loading**: Reduce map complexity or use simpler themes
- **Memory Issues**: Limit the number of nodes for complex visualizations

## Contributing

Feel free to extend this project by:
- Adding new optimization algorithms (A*, Bellman-Ford)
- Implementing real-world constraints and data sources
- Creating additional visualization options
- Adding comprehensive testing suites

## License

This project is open source and available for educational and commercial use.

---

**🚀 Ready to optimize your routes?** Start with `streamlit run app.py` and explore the interactive farm-to-market route optimization system!