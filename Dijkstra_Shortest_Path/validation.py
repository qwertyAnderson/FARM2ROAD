#!/usr/bin/env python3
"""
Data Validation Module for Farm-to-Market Route Optimizer

Comprehensive validation system for route data, coordinates, and graph consistency.
Provides user-friendly Streamlit feedback and ensures data integrity before
Dijkstra's algorithm computation.

Author: AI Assistant
Date: October 2025
"""

import pandas as pd
import numpy as np
import networkx as nx
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
import re


class RouteDataValidator:
    """
    Comprehensive validator for route optimization data.
    
    Validates CSV structure, data types, geographic coordinates,
    and graph consistency with detailed Streamlit feedback.
    """
    
    def __init__(self):
        self.validation_results = {
            'csv_structure': False,
            'data_types': False,
            'coordinates': False,
            'graph_consistency': False,
            'user_inputs': False
        }
        self.warnings = []
        self.errors = []
        self.summary_stats = {}
    
    def validate_route_data(self, df: pd.DataFrame) -> bool:
        """
        Main validation function that runs all checks on route data.
        
        Args:
            df (pd.DataFrame): Route data DataFrame
            
        Returns:
            bool: True if all validations pass, False otherwise
        """
        st.write("🔍 **Data Validation in Progress...**")
        
        # Clear previous results
        self.warnings.clear()
        self.errors.clear()
        self.validation_results = {key: False for key in self.validation_results}
        
        # Run validation checks
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.write("**Validation Steps:**")
            
            with col2:
                # Step 1: CSV Structure
                if self._validate_csv_structure(df):
                    st.success("✅ CSV structure valid")
                    self.validation_results['csv_structure'] = True
                else:
                    st.error("❌ CSV structure invalid")
                    return False
                
                # Step 2: Data Types
                if self._validate_data_types(df):
                    st.success("✅ Data types valid")
                    self.validation_results['data_types'] = True
                else:
                    st.error("❌ Data types invalid")
                    return False
                
                # Step 3: Coordinates (if present)
                coord_result = self._validate_coordinates(df)
                if coord_result is True:
                    st.success("✅ Coordinates valid")
                    self.validation_results['coordinates'] = True
                elif coord_result is None:
                    st.info("ℹ️ No coordinates provided")
                    self.validation_results['coordinates'] = True  # Not required
                else:
                    st.error("❌ Coordinates invalid")
                    return False
                
                # Step 4: Graph Consistency
                if self._validate_graph_consistency(df):
                    st.success("✅ Graph consistency valid")
                    self.validation_results['graph_consistency'] = True
                else:
                    st.warning("⚠️ Graph consistency issues detected")
                    # Allow continuation with warnings
                    self.validation_results['graph_consistency'] = True
        
        # Generate summary
        self._generate_data_summary(df)
        
        # Display warnings if any
        if self.warnings:
            st.warning("⚠️ **Warnings detected (computation will continue):**")
            for warning in self.warnings:
                st.write(f"• {warning}")
        
        # Final validation status
        all_passed = all(self.validation_results.values())
        
        if all_passed:
            st.success("🎉 **All validations passed successfully!** Ready for route optimization.")
        
        return all_passed
    
    def validate_data_only(self, df: pd.DataFrame) -> bool:
        """
        Validate only the data structure and content, not user inputs.
        Used for initial data validation before user selects source/target.
        """
        st.write("🔍 **Data Validation in Progress...**")
        
        # Clear previous results
        self.warnings.clear()
        self.errors.clear()
        data_validation_results = {
            'csv_structure': False,
            'data_types': False,
            'coordinates': False,
            'graph_consistency': False
        }
        
        # Run validation checks
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.write("**Validation Steps:**")
            
            with col2:
                # Step 1: CSV Structure
                try:
                    if self._validate_csv_structure(df):
                        st.success("✅ CSV structure valid")
                        data_validation_results['csv_structure'] = True
                    else:
                        st.error("❌ CSV structure invalid")
                        return False
                except Exception as e:
                    st.error(f"❌ CSV structure check failed: {e}")
                    return False
                
                # Step 2: Data Types
                try:
                    if self._validate_data_types(df):
                        st.success("✅ Data types valid")
                        data_validation_results['data_types'] = True
                    else:
                        st.error("❌ Data types invalid")
                        return False
                except Exception as e:
                    st.error(f"❌ Data types check failed: {e}")
                    return False
                
                # Step 3: Coordinates (if present)
                try:
                    coord_result = self._validate_coordinates(df)
                    if coord_result is True:
                        st.success("✅ Coordinates valid")
                        data_validation_results['coordinates'] = True
                    elif coord_result is None:
                        st.info("ℹ️ No coordinates provided")
                        data_validation_results['coordinates'] = True  # Not required
                    else:
                        st.error("❌ Coordinates invalid")
                        return False
                except Exception as e:
                    st.error(f"❌ Coordinate check failed: {e}")
                    # Coordinates are optional, so don't fail validation
                    data_validation_results['coordinates'] = True
                
                # Step 4: Graph Consistency
                try:
                    if self._validate_graph_consistency(df):
                        st.success("✅ Graph consistency valid")
                        data_validation_results['graph_consistency'] = True
                    else:
                        st.warning("⚠️ Graph consistency issues detected")
                        # Allow continuation with warnings
                        data_validation_results['graph_consistency'] = True
                except Exception as e:
                    st.error(f"❌ Graph consistency check failed: {e}")
                    # Allow continuation even if this fails
                    data_validation_results['graph_consistency'] = True
        
        # Generate summary
        try:
            self._generate_data_summary(df)
        except Exception as e:
            st.warning(f"⚠️ Could not generate data summary: {e}")
        
        # Display warnings if any
        if self.warnings:
            st.warning("⚠️ **Warnings detected (computation will continue):**")
            for warning in self.warnings:
                st.write(f"• {warning}")
        
        # Final validation status (only data-related checks)
        all_passed = all(data_validation_results.values())
        
        if all_passed:
            st.success("🎉 **Data validation passed successfully!** Ready for route configuration.")
        else:
            failed_checks = [k for k, v in data_validation_results.items() if not v]
            st.error(f"❌ **Data validation failed:** {', '.join(failed_checks)}")
        
        return all_passed
    
    def _validate_csv_structure(self, df: pd.DataFrame) -> bool:
        """Validate CSV has required columns and proper structure."""
        required_cols = {'source', 'target', 'weight'}
        optional_cols = {'lat', 'lon', 'source_lat', 'source_lon', 'target_lat', 'target_lon'}
        
        # Check if DataFrame is empty
        if df.empty:
            self.errors.append("CSV file is empty")
            st.error("❌ CSV file contains no data")
            return False
        
        # Check required columns
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            self.errors.append(f"Missing required columns: {', '.join(missing_cols)}")
            st.error(f"❌ Missing required columns: **{', '.join(missing_cols)}**")
            st.info("💡 **Required columns:** source, target, weight")
            return False
        
        # Check for extra columns
        extra_cols = set(df.columns) - required_cols - optional_cols
        if extra_cols:
            self.warnings.append(f"Extra columns detected (will be ignored): {', '.join(extra_cols)}")
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicates = [col for col in df.columns if list(df.columns).count(col) > 1]
            self.errors.append(f"Duplicate column names: {', '.join(set(duplicates))}")
            st.error(f"❌ Duplicate column names: **{', '.join(set(duplicates))}**")
            return False
        
        return True
    
    def _validate_data_types(self, df: pd.DataFrame) -> bool:
        """Validate data types and value ranges."""
        
        # Check for null values in required columns
        required_cols = ['source', 'target', 'weight']
        for col in required_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                self.errors.append(f"Column '{col}' contains {null_count} null values")
                st.error(f"❌ Column **{col}** contains {null_count} null value(s)")
                return False
        
        # Validate weight column
        if not pd.api.types.is_numeric_dtype(df['weight']):
            # Try to convert to numeric
            try:
                df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
                if df['weight'].isnull().any():
                    self.errors.append("Weight column contains non-numeric values")
                    st.error("❌ Weight column contains non-numeric values")
                    return False
                else:
                    self.warnings.append("Weight column converted to numeric type")
            except:
                self.errors.append("Weight column cannot be converted to numeric")
                st.error("❌ Weight column cannot be converted to numeric")
                return False
        
        # Check for positive weights
        negative_weights = (df['weight'] <= 0).sum()
        if negative_weights > 0:
            self.errors.append(f"{negative_weights} non-positive weight values detected")
            st.error(f"❌ Found {negative_weights} non-positive weight value(s)")
            st.info("💡 All weights must be positive numbers (> 0)")
            return False
        
        # Check for extremely large weights (potential data entry errors)
        max_weight = df['weight'].max()
        if max_weight > 10000:  # Arbitrary threshold for route distances
            self.warnings.append(f"Very large weight detected: {max_weight:.1f} km")
        
        # Validate source and target columns
        for col in ['source', 'target']:
            # Check for empty strings
            empty_values = df[col].astype(str).str.strip().eq('').sum()
            if empty_values > 0:
                self.errors.append(f"Column '{col}' contains {empty_values} empty values")
                st.error(f"❌ Column **{col}** contains {empty_values} empty value(s)")
                return False
            
            # Clean whitespace
            df[col] = df[col].astype(str).str.strip()
        
        return True
    
    def _validate_coordinates(self, df: pd.DataFrame) -> Optional[bool]:
        """Validate geographic coordinates if present."""
        
        # Check for coordinate columns
        coord_patterns = [
            ['lat', 'lon'],
            ['source_lat', 'source_lon', 'target_lat', 'target_lon'],
            ['latitude', 'longitude']
        ]
        
        coord_cols = []
        for pattern in coord_patterns:
            if all(col in df.columns for col in pattern):
                coord_cols = pattern
                break
        
        if not coord_cols:
            return None  # No coordinates provided
        
        # Validate coordinate data types
        for col in coord_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        self.errors.append(f"Coordinate column '{col}' contains non-numeric values")
                        st.error(f"❌ Coordinate column **{col}** contains non-numeric values")
                        return False
                except:
                    self.errors.append(f"Coordinate column '{col}' cannot be converted to numeric")
                    st.error(f"❌ Coordinate column **{col}** cannot be converted to numeric")
                    return False
        
        # Validate coordinate ranges
        if 'lat' in coord_cols and 'lon' in coord_cols:
            # Standard lat/lon validation
            invalid_lat = ((df['lat'] < -90) | (df['lat'] > 90)).sum()
            invalid_lon = ((df['lon'] < -180) | (df['lon'] > 180)).sum()
            
            if invalid_lat > 0:
                self.errors.append(f"{invalid_lat} invalid latitude values (must be between -90 and 90)")
                st.error(f"❌ {invalid_lat} invalid latitude value(s)")
                return False
            
            if invalid_lon > 0:
                self.errors.append(f"{invalid_lon} invalid longitude values (must be between -180 and 180)")
                st.error(f"❌ {invalid_lon} invalid longitude value(s)")
                return False
        
        elif all(col in coord_cols for col in ['source_lat', 'source_lon', 'target_lat', 'target_lon']):
            # Source/target coordinate validation
            for lat_col, lon_col in [('source_lat', 'source_lon'), ('target_lat', 'target_lon')]:
                invalid_lat = ((df[lat_col] < -90) | (df[lat_col] > 90)).sum()
                invalid_lon = ((df[lon_col] < -180) | (df[lon_col] > 180)).sum()
                
                if invalid_lat > 0:
                    self.errors.append(f"{invalid_lat} invalid {lat_col} values")
                    st.error(f"❌ {invalid_lat} invalid **{lat_col}** value(s)")
                    return False
                
                if invalid_lon > 0:
                    self.errors.append(f"{invalid_lon} invalid {lon_col} values")
                    st.error(f"❌ {invalid_lon} invalid **{lon_col}** value(s)")
                    return False
        
        return True
    
    def _validate_graph_consistency(self, df: pd.DataFrame) -> bool:
        """Validate graph structure and connectivity."""
        
        # Check for self-loops
        self_loops = (df['source'] == df['target']).sum()
        if self_loops > 0:
            self.warnings.append(f"{self_loops} self-loop(s) detected and will be ignored")
        
        # Create graph for connectivity analysis
        try:
            G = nx.Graph()
            for _, row in df.iterrows():
                if row['source'] != row['target']:  # Skip self-loops
                    G.add_edge(row['source'], row['target'], weight=row['weight'])
            
            # Check if graph is connected
            if not nx.is_connected(G):
                components = list(nx.connected_components(G))
                self.warnings.append(f"Graph has {len(components)} disconnected components")
                
                # Identify isolated nodes
                isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
                if isolated_nodes:
                    self.warnings.append(f"Isolated nodes detected: {', '.join(isolated_nodes[:5])}")
                
                # Show component sizes
                component_sizes = [len(comp) for comp in components]
                self.warnings.append(f"Component sizes: {component_sizes}")
        
        except Exception as e:
            self.errors.append(f"Graph construction failed: {str(e)}")
            st.error(f"❌ Graph construction failed: {str(e)}")
            return False
        
        return True
    
    def _generate_data_summary(self, df: pd.DataFrame) -> None:
        """Generate and display data summary statistics."""
        
        # Calculate summary statistics
        self.summary_stats = {
            'total_edges': len(df),
            'unique_nodes': len(set(df['source'].tolist() + df['target'].tolist())),
            'avg_weight': df['weight'].mean(),
            'min_weight': df['weight'].min(),
            'max_weight': df['weight'].max(),
            'weight_std': df['weight'].std()
        }
        
        # Display summary
        st.subheader("📊 Data Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🔗 Total Edges",
                value=self.summary_stats['total_edges']
            )
        
        with col2:
            st.metric(
                label="📍 Unique Nodes",
                value=self.summary_stats['unique_nodes']
            )
        
        with col3:
            st.metric(
                label="📏 Average Distance",
                value=f"{self.summary_stats['avg_weight']:.1f} km"
            )
        
        with col4:
            st.metric(
                label="📈 Distance Range",
                value=f"{self.summary_stats['min_weight']:.1f} - {self.summary_stats['max_weight']:.1f} km"
            )
    
    def validate_user_inputs(self, graph: nx.Graph, source: str, target: str) -> bool:
        """
        Validate user-selected source and target nodes.
        
        Args:
            graph (nx.Graph): NetworkX graph
            source (str): Selected source node
            target (str): Selected target node
            
        Returns:
            bool: True if inputs are valid, False otherwise
        """
        
        # Check if nodes exist in graph
        if source not in graph.nodes():
            st.error(f"❌ Start location **'{source}'** not found in the dataset")
            st.info("💡 Please select a valid start location from the dropdown")
            return False
        
        if target not in graph.nodes():
            st.error(f"❌ Destination **'{target}'** not found in the dataset")
            st.info("💡 Please select a valid destination from the dropdown")
            return False
        
        # Check if source and target are the same
        if source == target:
            st.warning("⚠️ Start and destination are the same location")
            st.info("💡 Please select different start and destination locations")
            return False
        
        # Check if path exists (basic connectivity)
        try:
            if not nx.has_path(graph, source, target):
                st.error(f"❌ No path exists between **'{source}'** and **'{target}'**")
                st.info("💡 These locations are in disconnected parts of the network")
                return False
        except Exception as e:
            st.error(f"❌ Error checking path connectivity: {str(e)}")
            return False
        
        self.validation_results['user_inputs'] = True
        st.success("✅ Start and destination locations are valid")
        return True
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get comprehensive validation report.
        
        Returns:
            Dict[str, Any]: Validation results and statistics
        """
        return {
            'validation_results': self.validation_results,
            'warnings': self.warnings,
            'errors': self.errors,
            'summary_stats': self.summary_stats,
            'all_passed': all(self.validation_results.values())
        }


def validate_route_data(df: pd.DataFrame) -> bool:
    """
    Convenience function for basic route data validation.
    
    Args:
        df (pd.DataFrame): Route data DataFrame
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    validator = RouteDataValidator()
    return validator.validate_route_data(df)


def create_sample_data_template() -> pd.DataFrame:
    """
    Create a sample data template for users.
    
    Returns:
        pd.DataFrame: Sample route data with proper format
    """
    return pd.DataFrame({
        'source': ['Farm_A', 'Farm_A', 'Farm_B', 'Village_X', 'Village_Y'],
        'target': ['Village_X', 'Village_Y', 'Village_X', 'Market_1', 'Market_1'],
        'weight': [5.2, 8.1, 3.7, 2.4, 4.9],
        'lat': [40.7128, 40.7200, 40.7050, 40.7300, 40.7180],
        'lon': [-74.0060, -74.0100, -74.0030, -74.0150, -74.0080]
    })


def display_validation_help() -> None:
    """Display helpful information about data validation requirements."""
    
    st.subheader("📋 Data Validation Requirements")
    
    with st.expander("📄 **CSV Format Requirements**"):
        st.write("""
        **Required Columns:**
        - `source`: Starting location name (text)
        - `target`: Ending location name (text)
        - `weight`: Distance/cost between locations (positive number)
        
        **Optional Columns:**
        - `lat`, `lon`: Coordinates for locations
        - `source_lat`, `source_lon`, `target_lat`, `target_lon`: Separate coordinates for each endpoint
        """)
    
    with st.expander("🔢 **Data Type Requirements**"):
        st.write("""
        - **Weights**: Must be positive numbers (> 0)
        - **Coordinates**: Must be valid decimal numbers
          - Latitude: -90 to +90 degrees
          - Longitude: -180 to +180 degrees
        - **Location Names**: Non-empty text strings
        """)
    
    with st.expander("🌐 **Graph Requirements**"):
        st.write("""
        - **Connectivity**: All locations should be reachable from each other
        - **Self-loops**: Routes from a location to itself are ignored
        - **Duplicates**: Multiple routes between same locations are allowed
        """)
    
    with st.expander("❌ **Common Issues & Solutions**"):
        st.write("""
        **Issue**: Missing required columns
        - **Solution**: Ensure CSV has 'source', 'target', 'weight' columns
        
        **Issue**: Non-positive weights
        - **Solution**: All distance/cost values must be greater than 0
        
        **Issue**: Invalid coordinates
        - **Solution**: Check latitude (-90 to 90) and longitude (-180 to 180) ranges
        
        **Issue**: Disconnected graph
        - **Solution**: Add routes to connect all locations in the network
        """)


def display_validation_help_simple() -> None:
    """Display simplified validation help without expanders (for sidebar use)."""
    
    st.write("**📋 Data Validation Requirements**")
    
    st.write("""
    **Required Columns:**
    - `source`: Starting location name
    - `target`: Ending location name  
    - `weight`: Distance (positive number)
    
    **Data Rules:**
    - Weights must be > 0
    - Coordinates: lat (-90 to 90), lon (-180 to 180)
    - All locations should be connected
    
    **Common Issues:**
    - Missing columns → Check CSV format
    - Negative weights → Use positive numbers
    - Invalid coordinates → Check lat/lon ranges
    """)
    
    st.info("💡 **Tip**: Upload sample data to see the expected format")


if __name__ == "__main__":
    # Test the validation module
    print("Testing Route Data Validator...")
    
    # Create test data
    test_data = create_sample_data_template()
    
    # Test validation
    validator = RouteDataValidator()
    result = validator.validate_route_data(test_data)
    
    print(f"Validation result: {result}")
    print(f"Validation report: {validator.get_validation_report()}")