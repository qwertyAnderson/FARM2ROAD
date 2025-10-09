# Farm-to-Market Route Optimization

A Python implementation of Dijkstra's Algorithm using NetworkX to find the shortest path from a farm to a market through a network of villages and roads.

## Overview

This project demonstrates how Dijkstra's Algorithm can be applied to real-world logistics problems, specifically optimizing routes for agricultural transportation. The algorithm finds the most efficient path from a farm to a market, considering road distances and intermediate villages.

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
├── main.py           # Main implementation script
├── sample_data.csv   # Sample network data
├── requirements.txt  # Python dependencies
└── README.md        # This documentation
```

## Features

- **Graph Loading**: Reads network data from CSV files
- **Shortest Path Calculation**: Uses NetworkX's Dijkstra implementation
- **Visualization**: Creates interactive plots showing the network and optimal route
- **Comprehensive Analysis**: Provides graph statistics and alternative route information
- **Error Handling**: Robust error handling for missing files or invalid data

## Installation

1. **Clone or download** this project to your local machine

2. **Install dependencies** using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation** by checking if packages are installed:
   ```bash
   python -c "import networkx, matplotlib, pandas; print('All packages installed successfully!')"
   ```

## Usage

### Basic Usage

Run the main script:
```bash
python main.py
```

### Expected Output

The script will:
1. Load the network from `sample_data.csv`
2. Analyze the graph structure
3. Calculate the shortest path from Farm to Market
4. Display detailed results
5. Generate a visualization plot
6. Save the plot as `dijkstra_shortest_path.png`

### Sample Output

```
Farm-to-Market Route Optimization using Dijkstra's Algorithm
=================================================================
Loaded 13 edges from sample_data.csv
Graph created with 7 nodes and 13 edges

GRAPH ANALYSIS
------------------------------
Number of locations (nodes): 7
Number of routes (edges): 13
Graph is connected: True
Locations: Farm, Market, VillageA, VillageB, VillageC, VillageD, VillageE, VillageF

============================================================
DIJKSTRA'S ALGORITHM RESULTS
============================================================
Start Location: Farm
End Location: Market
Shortest Path: Farm → VillageB → VillageC → VillageF → Market
Total Distance: 6.0 km
Number of Stops: 3

Detailed Route:
  Step 1: Farm → VillageB
  Step 2: VillageB → VillageC
  Step 3: VillageC → VillageF
  Step 4: VillageF → Market
============================================================
```

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

## Customization

### Modify the Dataset

1. **Edit `sample_data.csv`** to add/remove locations or change distances
2. **Ensure connectivity**: Make sure there's a path from Farm to Market
3. **Use realistic distances**: Keep weights positive and reasonable

### Change Start/End Points

Modify the `main()` function in `main.py`:
```python
start_location = "YourStartLocation"
end_location = "YourEndLocation"
```

### Add Multiple Markets

You can extend the algorithm to find paths to multiple markets:
```python
markets = ["Market1", "Market2", "Market3"]
for market in markets:
    path, distance = find_shortest_path(graph, "Farm", market)
    print(f"Shortest path to {market}: {distance} km")
```

## Technical Details

### Dependencies

- **NetworkX**: Graph creation and algorithm implementation
- **Matplotlib**: Graph visualization and plotting
- **Pandas**: CSV data loading and manipulation
- **NumPy**: Numerical operations support

### Algorithm Complexity

- **Time Complexity**: O((V + E) log V) where V is vertices and E is edges
- **Space Complexity**: O(V) for storing distances and paths

### Visualization Features

- **Color Coding**: 
  - Green: Start location (Farm)
  - Red: End location (Market)
  - Orange: Shortest path nodes
  - Blue: Other locations
- **Edge Highlighting**: Shortest path edges shown in red
- **Weight Labels**: All edge weights displayed
- **Legend**: Clear explanation of colors and symbols

## Extensions

### 1. Multiple Destination Optimization
Find the shortest paths to multiple markets and choose the best one.

### 2. Time-Based Routing
Add time constraints or varying road conditions based on time of day.

### 3. Capacity Constraints
Consider vehicle capacity and multiple trips.

### 4. Real-World Integration
Use actual GPS coordinates and road network data.

### 5. Interactive Interface
Create a web interface for dynamic route planning.

## Troubleshooting

### Common Issues

1. **"No path found"**: Ensure your graph is connected
2. **"Node not found"**: Check spelling in start/end location names
3. **Import errors**: Run `pip install -r requirements.txt`
4. **Empty visualization**: Check if matplotlib backend is properly configured

### Debugging Tips

- Use `print(graph.nodes())` to see all available locations
- Check `nx.is_connected(graph)` to verify graph connectivity
- Validate CSV format and data types

## Contributing

Feel free to extend this project by:
- Adding new optimization algorithms (A*, Bellman-Ford)
- Implementing real-world constraints
- Creating web interfaces
- Adding more comprehensive testing

## License

This project is open source and available for educational and commercial use.

---

**Created with ❤️ for demonstrating practical applications of graph algorithms in logistics and transportation optimization.**