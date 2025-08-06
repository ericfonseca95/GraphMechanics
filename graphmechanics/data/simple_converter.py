"""
Simple motion graph converter for testing.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import List

class MotionGraphConverter:
    """Simple converter for TRC to PyG data."""
    
    def __init__(self):
        pass
    
    def compute_kinematic_features(self, positions: np.ndarray, dt: float = 1/120) -> np.ndarray:
        """Compute kinematic features from position data."""
        n_frames, n_joints, _ = positions.shape
        
        # Initialize feature array: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        features = np.zeros((n_frames, n_joints, 6))
        
        # Position features
        features[:, :, :3] = positions
        
        # Velocity features (simple difference)
        if n_frames > 1:
            features[1:, :, 3:6] = (positions[1:] - positions[:-1]) / dt
            features[0, :, 3:6] = features[1, :, 3:6]  # Copy first velocity
            
        return features
    
    def create_simple_edge_index(self, num_nodes: int) -> torch.Tensor:
        """Create simple chain connectivity."""
        edges = []
        for i in range(num_nodes - 1):
            edges.extend([[i, i+1], [i+1, i]])  # Bidirectional edges
        
        if not edges:
            edges = [[0, 0]]  # Self-loop for single node
            
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def trc_to_pyg_data(self, trc_data: dict, frame_window: int = 10) -> List[Data]:
        """Convert TRC data to PyG Data objects."""
        joint_names = trc_data['joint_names']
        positions = trc_data['positions']
        
        # Compute kinematic features
        features = self.compute_kinematic_features(positions)
        
        # Create simple edge index
        edge_index = self.create_simple_edge_index(len(joint_names))
        
        # Create Data objects for sliding windows
        data_objects = []
        n_frames = len(positions)
        
        for start_idx in range(0, n_frames - frame_window + 1, frame_window // 2):
            end_idx = start_idx + frame_window
            
            # Extract window features and reshape
            window_features = features[start_idx:end_idx]  # (window, joints, features)
            
            # Flatten temporal dimension into node features
            node_features = window_features.transpose(1, 0, 2).reshape(len(joint_names), -1)
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            
            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                num_nodes=len(joint_names),
                frame_start=start_idx,
                frame_end=end_idx
            )
            
            data_objects.append(data)
        
        return data_objects

class KinematicGraphBuilder:
    """Simple kinematic graph builder."""
    
    def __init__(self):
        pass
    
    def build_edge_index(self, marker_names: List[str]) -> torch.Tensor:
        """Build simple chain edge index."""
        edges = []
        for i in range(len(marker_names) - 1):
            edges.extend([[i, i+1], [i+1, i]])
        
        if not edges:
            edges = [[0, 0]]
            
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
