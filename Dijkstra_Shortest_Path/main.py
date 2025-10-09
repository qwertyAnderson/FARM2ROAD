#!/usr/bin/env python3
"""
Farm-to-Market Route Optimization using Dijkstra's Algorithm

This script implements Dijkstra's algorithm using NetworkX to find the shortest
path from a farm to a market through a network of villages and roads.

Author: AI Assistant
Date: October 2025
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any


def load_graph(csv_file: str) -> nx.Graph:
    """
    Load graph data from CSV file and create a NetworkX graph.
    
    Args:
        csv_file (str): Path to CSV file containing edge data
        
    Returns:
        nx.Graph: NetworkX graph object with weighted edges
    """
    try:
        # Load the CSV data
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} edges from {csv_file}")
        
        # Create an undirected graph
        G = nx.Graph()
        
        # Add edges with weights to the graph
        for _, row in df.iterrows():
            G.add_edge(row['source'], row['target'], weight=row['weight'])
        
        print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
        
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found!")
        return None
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None


def find_shortest_path(graph: nx.Graph, start: str, end: str) -> Tuple[List[str], float]:
    """
    Find the shortest path between two nodes using Dijkstra's algorithm.
    
    Args:
        graph (nx.Graph): NetworkX graph object
        start (str): Starting node
        end (str): Ending node
        
    Returns:
        Tuple[List[str], float]: Shortest path as list of nodes and total distance
    """
    try:
        # Use NetworkX's implementation of Dijkstra's algorithm
        shortest_path = nx.dijkstra_path(graph, start, end, weight='weight')
        shortest_distance = nx.dijkstra_path_length(graph, start, end, weight='weight')
        
        return shortest_path, shortest_distance
        
    except nx.NetworkXNoPath:
        print(f"No path found between {start} and {end}")
        return [], float('inf')
    except nx.NodeNotFound as e:
        print(f"Node not found: {e}")
        return [], float('inf')
    except Exception as e:
        print(f"Error finding shortest path: {e}")
        return [], float('inf')


def visualize_graph(graph: nx.Graph, shortest_path: List[str], start: str, end: str, 
                   save_plot: bool = True) -> None:
    """
    Visualize the graph with the shortest path highlighted.
    
    Args:
        graph (nx.Graph): NetworkX graph object
        shortest_path (List[str]): List of nodes in the shortest path
        start (str): Starting node
        end (str): Ending node
        save_plot (bool): Whether to save the plot as an image
    """
    plt.figure(figsize=(12, 8))
    
    # Create a layout for the graph
    pos = nx.spring_layout(graph, seed=42, k=3, iterations=50)
    
    # Draw all edges in light gray
    nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1, alpha=0.7)
    
    # Highlight the shortest path edges in red
    if len(shortest_path) > 1:
        path_edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, 
                              edge_color='red', width=3, alpha=0.8)
    
    # Draw nodes
    node_colors = []
    for node in graph.nodes():
        if node == start:
            node_colors.append('green')  # Start node in green
        elif node == end:
            node_colors.append('red')    # End node in red
        elif node in shortest_path:
            node_colors.append('orange') # Path nodes in orange
        else:
            node_colors.append('lightblue') # Other nodes in light blue
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                          node_size=1000, alpha=0.9)
    
    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
    
    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=8)
    
    # Set title and styling
    plt.title('Farm-to-Market Route Optimization\nDijkstra\'s Algorithm Shortest Path', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label=f'Start ({start})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label=f'End ({end})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=10, label='Shortest Path'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Shortest Route'),
        plt.Line2D([0], [0], color='lightgray', linewidth=1, label='Other Routes')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('dijkstra_shortest_path.png', dpi=300, bbox_inches='tight')
        print("Graph visualization saved as 'dijkstra_shortest_path.png'")
    
    plt.show()


def print_results(shortest_path: List[str], shortest_distance: float, 
                 start: str, end: str) -> None:
    """
    Print the results of the shortest path calculation.
    
    Args:
        shortest_path (List[str]): List of nodes in the shortest path
        shortest_distance (float): Total distance of the shortest path
        start (str): Starting node
        end (str): Ending node
    """
    print("\n" + "="*60)
    print("DIJKSTRA'S ALGORITHM RESULTS")
    print("="*60)
    
    if shortest_path:
        print(f"Start Location: {start}")
        print(f"End Location: {end}")
        print(f"Shortest Path: {' → '.join(shortest_path)}")
        print(f"Total Distance: {shortest_distance} km")
        print(f"Number of Stops: {len(shortest_path) - 1}")
        
        # Show step-by-step route
        print("\nDetailed Route:")
        for i in range(len(shortest_path) - 1):
            current = shortest_path[i]
            next_node = shortest_path[i + 1]
            print(f"  Step {i+1}: {current} → {next_node}")
    else:
        print(f"No path found from {start} to {end}")
    
    print("="*60)


def analyze_graph(graph: nx.Graph) -> None:
    """
    Provide basic analysis of the graph structure.
    
    Args:
        graph (nx.Graph): NetworkX graph object
    """
    print("\nGRAPH ANALYSIS")
    print("-" * 30)
    print(f"Number of locations (nodes): {graph.number_of_nodes()}")
    print(f"Number of routes (edges): {graph.number_of_edges()}")
    print(f"Graph is connected: {nx.is_connected(graph)}")
    
    # Show all locations
    print(f"Locations: {', '.join(sorted(graph.nodes()))}")
    
    # Show degree of each node (number of connections)
    degrees = dict(graph.degree())
    print("\nLocation Connectivity:")
    for location, degree in sorted(degrees.items()):
        print(f"  {location}: {degree} connection(s)")


def main():
    """
    Main function to execute the farm-to-market route optimization.
    """
    print("Farm-to-Market Route Optimization using Dijkstra's Algorithm")
    print("=" * 65)
    
    # Configuration
    csv_file = "sample_data.csv"
    start_location = "Farm"
    end_location = "Market"
    
    # Load the graph from CSV
    graph = load_graph(csv_file)
    if graph is None:
        return
    
    # Analyze the graph
    analyze_graph(graph)
    
    # Find the shortest path
    print(f"\nFinding shortest path from {start_location} to {end_location}...")
    shortest_path, shortest_distance = find_shortest_path(graph, start_location, end_location)
    
    # Print results
    print_results(shortest_path, shortest_distance, start_location, end_location)
    
    # Visualize the graph
    if shortest_path:
        print("\nGenerating visualization...")
        visualize_graph(graph, shortest_path, start_location, end_location)
    
    # Calculate all shortest paths from farm to all other locations
    print("\nSHORTEST DISTANCES FROM FARM TO ALL LOCATIONS")
    print("-" * 50)
    try:
        distances = nx.single_source_dijkstra_path_length(graph, start_location, weight='weight')
        for location, distance in sorted(distances.items()):
            if location != start_location:
                print(f"Farm → {location}: {distance} km")
    except Exception as e:
        print(f"Error calculating all distances: {e}")


if __name__ == "__main__":
    main()