"""
OpenSim Time-Series Graph Builder

This module provides specialized graph construction for OpenSim data with joint angles as node features
and muscle properties/geometries as edge features for time-series analysis.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from pathlib import Path

from ..utils.opensim_parser import OpenSimParser, OpenSimModelParser, OpenSimMotionParser


class OpenSimTimeSeriesGraphBuilder:
    """
    Enhanced graph builder for OpenSim time-series data with joint angles as node features
    and muscle properties/joint distances as edge features.
    """
    
    def __init__(self, model_parser: Optional[OpenSimModelParser] = None):
        """
        Initialize the OpenSim time-series graph builder.
        
        Args:
            model_parser: Optional OpenSim model parser for extracting muscle properties
        """
        self.model_parser = model_parser
        self._joint_to_node_mapping = {}
        self._muscle_edge_cache = {}
        
    def create_joint_angle_graphs(
        self, 
        motion_parser: OpenSimMotionParser,
        time_window: Optional[Tuple[float, float]] = None,
        frame_step: int = 1
    ) -> List[Data]:
        """
        Create time-series graphs with joint angles as node features.
        
        Args:
            motion_parser: OpenSim motion parser containing joint angle data
            time_window: Optional time window (start_time, end_time) to extract
            frame_step: Step size for frame sampling
            
        Returns:
            List of PyTorch Geometric Data objects, one per time frame
        """
        # Get motion data
        motion_data = motion_parser.data
        coordinate_names = motion_parser.coordinate_names
        
        # Filter by time window if specified
        if time_window is not None and 'time' in motion_data.columns:
            start_time, end_time = time_window
            mask = (motion_data['time'] >= start_time) & (motion_data['time'] <= end_time)
            motion_data = motion_data[mask]
        
        # Sample frames
        motion_data = motion_data.iloc[::frame_step]
        
        if len(motion_data) == 0:
            raise ValueError("No data available after filtering and sampling")
        
        # Create node mapping (each coordinate becomes a node)
        self._joint_to_node_mapping = {coord: i for i, coord in enumerate(coordinate_names)}
        
        # Build edge index based on anatomical connections
        edge_index = self._build_joint_edge_index(coordinate_names)
        
        # Compute edge features from model (muscle properties, joint distances)
        edge_attr = self._compute_edge_features(coordinate_names, edge_index)
        
        # Create graphs for each time frame
        graphs = []
        for idx, (_, row) in enumerate(motion_data.iterrows()):
            # Extract joint angles as node features
            node_features = []
            for coord in coordinate_names:
                if coord in motion_data.columns and coord != 'time':
                    angle_value = row[coord]
                    # Add angle, velocity, and acceleration if available
                    node_features.append([angle_value])
                else:
                    node_features.append([0.0])  # Missing data
            
            node_features = torch.tensor(node_features, dtype=torch.float)
            
            # Create graph data
            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                time=row.get('time', idx * (1/60)),  # Default to 60 Hz if no time
                frame_idx=idx,
                coordinate_names=coordinate_names
            )
            
            graphs.append(graph)
        
        return graphs
    
    def _build_joint_edge_index(self, coordinate_names: List[str]) -> torch.Tensor:
        """
        Build edge index based on joint hierarchy and anatomical connections.
        
        Args:
            coordinate_names: List of coordinate/joint names
            
        Returns:
            Edge index tensor [2, num_edges]
        """
        edges = []
        
        if self.model_parser:
            # Use model hierarchy to create edges
            joint_hierarchy = self.model_parser.get_joint_hierarchy()
            coord_to_joint = getattr(self.model_parser, 'coordinate_to_joint', {})
            
            # Create edges based on joint connections
            for coord1 in coordinate_names:
                if coord1 == 'time':
                    continue
                    
                coord1_idx = self._joint_to_node_mapping.get(coord1)
                if coord1_idx is None:
                    continue
                
                joint1 = coord_to_joint.get(coord1)
                if not joint1:
                    continue
                
                joint1_info = joint_hierarchy.get(joint1, {})
                parent_body = joint1_info.get('parent_body')
                child_body = joint1_info.get('child_body')
                
                # Connect to coordinates in the same joint
                for coord2 in coordinate_names:
                    if coord2 == 'time' or coord2 == coord1:
                        continue
                        
                    coord2_idx = self._joint_to_node_mapping.get(coord2)
                    if coord2_idx is None:
                        continue
                    
                    joint2 = coord_to_joint.get(coord2)
                    if joint2 == joint1:
                        # Same joint - strong connection
                        edges.append([coord1_idx, coord2_idx])
                        edges.append([coord2_idx, coord1_idx])
                    else:
                        # Check for hierarchical connection
                        joint2_info = joint_hierarchy.get(joint2, {})
                        if (joint2_info.get('parent_body') == child_body or 
                            joint2_info.get('child_body') == parent_body):
                            edges.append([coord1_idx, coord2_idx])
                            edges.append([coord2_idx, coord1_idx])
        else:
            # Fallback: create connections based on naming patterns
            edges = self._create_anatomical_edges_from_names(coordinate_names)
        
        # Ensure we have some edges
        if not edges:
            # Create chain connections as fallback
            for i in range(len(coordinate_names) - 1):
                if coordinate_names[i] != 'time' and coordinate_names[i+1] != 'time':
                    edges.append([i, i+1])
                    edges.append([i+1, i])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def _create_anatomical_edges_from_names(self, coordinate_names: List[str]) -> List[List[int]]:
        """
        Create anatomical edges based on coordinate naming patterns.
        """
        edges = []
        coord_to_idx = {coord: i for i, coord in enumerate(coordinate_names) if coord != 'time'}
        
        # Define anatomical groupings based on common OpenSim naming
        anatomical_groups = {
            'spine': ['lumbar', 'thorax', 'neck', 'head'],
            'pelvis': ['pelvis', 'hip'],
            'right_leg': ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                         'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r'],
            'left_leg': ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
                        'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l'],
            'right_arm': ['arm_flex_r', 'arm_add_r', 'arm_rot_r',
                         'elbow_flex_r', 'pro_sup_r'],
            'left_arm': ['arm_flex_l', 'arm_add_l', 'arm_rot_l',
                        'elbow_flex_l', 'pro_sup_l']
        }
        
        # Connect coordinates within each anatomical group
        for group_name, group_coords in anatomical_groups.items():
            group_indices = []
            for coord in group_coords:
                for actual_coord in coordinate_names:
                    if coord in actual_coord.lower() and actual_coord in coord_to_idx:
                        group_indices.append(coord_to_idx[actual_coord])
            
            # Create full connections within group
            for i in range(len(group_indices)):
                for j in range(i+1, len(group_indices)):
                    edges.append([group_indices[i], group_indices[j]])
                    edges.append([group_indices[j], group_indices[i]])
        
        # Connect bilateral pairs (left-right symmetry)
        bilateral_pairs = []
        for coord in coordinate_names:
            if coord == 'time':
                continue
            if '_r' in coord.lower():
                left_coord = coord.lower().replace('_r', '_l')
                for potential_left in coordinate_names:
                    if potential_left.lower() == left_coord and potential_left in coord_to_idx:
                        bilateral_pairs.append((coord_to_idx[coord], coord_to_idx[potential_left]))
        
        for r_idx, l_idx in bilateral_pairs:
            edges.append([r_idx, l_idx])
            edges.append([l_idx, r_idx])
        
        return edges
    
    def _compute_edge_features(self, coordinate_names: List[str], edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute edge features including muscle properties and joint distances.
        
        Args:
            coordinate_names: List of coordinate names
            edge_index: Edge connectivity tensor
            
        Returns:
            Edge feature tensor [num_edges, num_features]
        """
        num_edges = edge_index.shape[1]
        
        if self.model_parser:
            # Use model information for rich edge features
            edge_features = []
            
            muscle_summary = self.model_parser.get_muscle_summary()
            joint_hierarchy = self.model_parser.get_joint_hierarchy()
            coord_to_joint = getattr(self.model_parser, 'coordinate_to_joint', {})
            
            for i in range(num_edges):
                source_idx = edge_index[0, i].item()
                target_idx = edge_index[1, i].item()
                
                source_coord = coordinate_names[source_idx] if source_idx < len(coordinate_names) else None
                target_coord = coordinate_names[target_idx] if target_idx < len(coordinate_names) else None
                
                if not source_coord or not target_coord or source_coord == 'time' or target_coord == 'time':
                    edge_features.append([0.0, 0.0, 0.0, 0.0, 0.0])
                    continue
                
                # Get joint information
                source_joint = coord_to_joint.get(source_coord)
                target_joint = coord_to_joint.get(target_coord)
                
                # Feature 1: Joint distance (anatomical hierarchy distance)
                joint_distance = self._compute_joint_distance(source_joint, target_joint, joint_hierarchy)
                
                # Feature 2-3: Muscle properties affecting this connection
                muscle_force_sum, muscle_count = self._compute_muscle_connection_properties(
                    source_joint, target_joint, muscle_summary, joint_hierarchy)
                
                # Feature 4: Same body connection indicator
                same_body = 1.0 if source_joint == target_joint else 0.0
                
                # Feature 5: Bilateral symmetry indicator
                bilateral_symmetry = self._compute_bilateral_symmetry(source_coord, target_coord)
                
                edge_features.append([
                    joint_distance,
                    muscle_force_sum,
                    muscle_count,
                    same_body,
                    bilateral_symmetry
                ])
            
            return torch.tensor(edge_features, dtype=torch.float)
        else:
            # Fallback: simple edge features
            return torch.ones(num_edges, 3)  # uniform features
    
    def _compute_joint_distance(self, joint1: Optional[str], joint2: Optional[str], 
                               joint_hierarchy: Dict) -> float:
        """Compute anatomical distance between joints in hierarchy."""
        if not joint1 or not joint2 or joint1 == joint2:
            return 0.0
        
        # Simple heuristic: use body hierarchy
        joint1_info = joint_hierarchy.get(joint1, {})
        joint2_info = joint_hierarchy.get(joint2, {})
        
        # Same parent body = close distance
        if joint1_info.get('parent_body') == joint2_info.get('parent_body'):
            return 1.0
        
        # Child-parent relationship
        if (joint1_info.get('child_body') == joint2_info.get('parent_body') or
            joint2_info.get('child_body') == joint1_info.get('parent_body')):
            return 2.0
        
        # Default distance
        return 3.0
    
    def _compute_muscle_connection_properties(self, joint1: Optional[str], joint2: Optional[str],
                                            muscle_summary: Dict, joint_hierarchy: Dict) -> Tuple[float, float]:
        """Compute muscle properties affecting connection between joints."""
        if not joint1 or not joint2:
            return 0.0, 0.0
        
        total_force = 0.0
        muscle_count = 0.0
        
        # Find muscles that cross both joints
        joint1_info = joint_hierarchy.get(joint1, {})
        joint2_info = joint_hierarchy.get(joint2, {})
        
        joint1_body = joint1_info.get('child_body', '')
        joint2_body = joint2_info.get('child_body', '')
        
        for muscle_name, muscle_info in muscle_summary.items():
            # Simplified: assume muscle affects connection if it has path points
            # on both bodies (real implementation would analyze path points)
            if muscle_info.get('path_points', 0) > 1:
                # Heuristic: muscles with more path points likely span more joints
                max_force = muscle_info.get('max_force', 0)
                if max_force > 0:
                    total_force += max_force * 0.1  # Scale factor
                    muscle_count += 1
        
        return total_force, muscle_count
    
    def _compute_bilateral_symmetry(self, coord1: str, coord2: str) -> float:
        """Compute bilateral symmetry indicator."""
        if '_r' in coord1.lower() and '_l' in coord2.lower():
            base1 = coord1.lower().replace('_r', '')
            base2 = coord2.lower().replace('_l', '')
            return 1.0 if base1 == base2 else 0.0
        elif '_l' in coord1.lower() and '_r' in coord2.lower():
            base1 = coord1.lower().replace('_l', '')
            base2 = coord2.lower().replace('_r', '')
            return 1.0 if base1 == base2 else 0.0
        return 0.0
    
    def create_sequence_graphs(
        self,
        motion_parser: OpenSimMotionParser,
        sequence_length: int = 10,
        overlap: int = 5,
        time_window: Optional[Tuple[float, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create sequences of graphs for time-series learning.
        
        Args:
            motion_parser: OpenSim motion parser
            sequence_length: Length of each sequence
            overlap: Overlap between sequences
            time_window: Optional time window to extract
            
        Returns:
            List of sequence dictionaries containing graph sequences
        """
        # Create individual frame graphs
        frame_graphs = self.create_joint_angle_graphs(motion_parser, time_window)
        
        if len(frame_graphs) < sequence_length:
            raise ValueError(f"Not enough frames ({len(frame_graphs)}) for sequence length {sequence_length}")
        
        sequences = []
        step = sequence_length - overlap
        
        for start_idx in range(0, len(frame_graphs) - sequence_length + 1, step):
            end_idx = start_idx + sequence_length
            sequence_graphs = frame_graphs[start_idx:end_idx]
            
            # Create sequence data
            sequence_data = {
                'graphs': sequence_graphs,
                'start_frame': start_idx,
                'end_frame': end_idx - 1,
                'start_time': sequence_graphs[0].time,
                'end_time': sequence_graphs[-1].time,
                'sequence_length': sequence_length
            }
            
            sequences.append(sequence_data)
        
        return sequences
    
    def enhance_graphs_with_derivatives(
        self,
        graphs: List[Data],
        dt: float = 1/60
    ) -> List[Data]:
        """
        Enhance graphs by adding velocity and acceleration as additional node features.
        
        Args:
            graphs: List of graph data objects
            dt: Time step for derivative computation
            
        Returns:
            Enhanced graphs with velocity and acceleration features
        """
        if len(graphs) < 3:
            warnings.warn("Need at least 3 frames for acceleration computation")
            return graphs
        
        enhanced_graphs = []
        
        for i, graph in enumerate(graphs):
            # Get position (joint angles)
            positions = graph.x  # [num_nodes, 1]
            
            # Compute velocity
            if i == 0:
                # Forward difference for first frame
                velocity = (graphs[i+1].x - positions) / dt
            elif i == len(graphs) - 1:
                # Backward difference for last frame
                velocity = (positions - graphs[i-1].x) / dt
            else:
                # Central difference for middle frames
                velocity = (graphs[i+1].x - graphs[i-1].x) / (2 * dt)
            
            # Compute acceleration
            if i == 0:
                # Forward difference
                acceleration = (graphs[i+2].x - 2*graphs[i+1].x + positions) / (dt**2)
            elif i == len(graphs) - 1:
                # Backward difference
                acceleration = (positions - 2*graphs[i-1].x + graphs[i-2].x) / (dt**2)
            elif i == 1:
                # Forward difference (need 3 points ahead)
                if i + 1 < len(graphs):
                    acceleration = (graphs[i+1].x - 2*positions + graphs[i-1].x) / (dt**2)
                else:
                    acceleration = torch.zeros_like(positions)
            else:
                # Central difference
                acceleration = (graphs[i+1].x - 2*positions + graphs[i-1].x) / (dt**2)
            
            # Combine features: [position, velocity, acceleration]
            enhanced_features = torch.cat([positions, velocity, acceleration], dim=1)
            
            # Create enhanced graph
            enhanced_graph = Data(
                x=enhanced_features,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                time=graph.time,
                frame_idx=graph.frame_idx,
                coordinate_names=graph.coordinate_names
            )
            
            enhanced_graphs.append(enhanced_graph)
        
        return enhanced_graphs
