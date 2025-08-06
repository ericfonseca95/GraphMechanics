"""
Dataset class for motion capture data represented as graphs.

This module provides PyTorch Dataset classes for handling motion capture data
converted to graph format, with support for batching, data augmentation,
and various biomechanical analysis tasks.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Union, Callable
from ..utils.trc_parser import TRCParser
from .graph_builder import create_motion_graph, KinematicGraphBuilder


class MotionGraphDataset(Dataset):
    """
    PyTorch Dataset for motion capture data represented as graphs.
    
    This dataset converts motion capture sequences into graph representations
    suitable for training graph neural networks. Each sample can represent:
    - A single frame as a spatial graph
    - A temporal window as a spatio-temporal graph
    - An entire motion sequence
    """
    
    def __init__(
        self,
        data_source: Union[str, pd.DataFrame, List[str]],
        marker_names: Optional[List[str]] = None,
        window_size: Optional[int] = None,
        stride: int = 1,
        connectivity_type: str = 'skeletal',
        include_temporal_edges: bool = False,
        temporal_connections: int = 1,
        labels: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None
    ):
        """
        Initialize the motion graph dataset.
        
        Args:
            data_source: Either a file path to TRC file, DataFrame, or list of file paths
            marker_names: List of marker names (extracted automatically if None)
            window_size: Size of temporal windows (None for full sequences)
            stride: Stride for sliding window (only used if window_size is not None)
            connectivity_type: Graph connectivity type ('skeletal', 'distance', 'custom')
            include_temporal_edges: Whether to include temporal connections
            temporal_connections: Number of temporal connections per node
            labels: Labels for classification tasks
            transform: Transform applied to each sample
            pre_transform: Transform applied during preprocessing
        """
        super().__init__()
        
        self.window_size = window_size
        self.stride = stride
        self.connectivity_type = connectivity_type
        self.include_temporal_edges = include_temporal_edges
        self.temporal_connections = temporal_connections
        self.transform = transform
        self.pre_transform = pre_transform
        
        # Load and process data
        self.motion_data, self.marker_names = self._load_data(data_source, marker_names)
        
        # Create sample indices
        self.sample_indices = self._create_sample_indices()
        
        # Set labels
        self.labels = self._process_labels(labels)
        
        # Preprocess data if transform is provided
        if self.pre_transform is not None:
            self._apply_pre_transform()
    
    def _load_data(
        self, 
        data_source: Union[str, pd.DataFrame, List[str]], 
        marker_names: Optional[List[str]]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Load motion capture data from various sources."""
        
        if isinstance(data_source, str):
            # Single TRC file
            parser = TRCParser(data_source)
            motion_data = parser.data
            marker_names = marker_names or parser.marker_names
            
        elif isinstance(data_source, pd.DataFrame):
            # DataFrame directly provided
            motion_data = data_source.copy()
            if marker_names is None:
                # Infer marker names from column names
                marker_names = []
                for col in motion_data.columns:
                    if col.endswith('_X'):
                        marker_names.append(col[:-2])
            
        elif isinstance(data_source, list):
            # Multiple TRC files - concatenate them
            all_data = []
            all_marker_names = []
            
            for file_path in data_source:
                parser = TRCParser(file_path)
                all_data.append(parser.data)
                all_marker_names.append(parser.marker_names)
            
            # Use marker names from first file
            marker_names = marker_names or all_marker_names[0]
            
            # Concatenate data
            motion_data = pd.concat(all_data, ignore_index=True)
            
        else:
            raise ValueError("data_source must be a file path, DataFrame, or list of file paths")
        
        return motion_data, marker_names
    
    def _create_sample_indices(self) -> List[Tuple[int, int]]:
        """Create indices for samples based on windowing strategy."""
        
        if self.window_size is None:
            # Use entire sequence as one sample
            return [(0, len(self.motion_data))]
        
        else:
            # Create sliding windows
            indices = []
            for start in range(0, len(self.motion_data) - self.window_size + 1, self.stride):
                end = start + self.window_size
                indices.append((start, end))
            
            return indices
    
    def _process_labels(self, labels) -> Optional[torch.Tensor]:
        """Process labels for the dataset."""
        
        if labels is None:
            return None
        
        if isinstance(labels, (list, np.ndarray)):
            labels = torch.tensor(labels)
        elif not isinstance(labels, torch.Tensor):
            raise ValueError("Labels must be a list, numpy array, or torch tensor")
        
        # Check label dimensions
        if len(labels) != len(self.sample_indices):
            if len(labels) == len(self.motion_data):
                # Labels per frame - need to aggregate for windows
                if self.window_size is not None:
                    # Use label from middle of window or most frequent label
                    window_labels = []
                    for start, end in self.sample_indices:
                        mid_idx = start + (end - start) // 2
                        window_labels.append(labels[mid_idx])
                    labels = torch.stack(window_labels)
            else:
                raise ValueError(f"Number of labels ({len(labels)}) must match number of samples ({len(self.sample_indices)})")
        
        return labels
    
    def _apply_pre_transform(self):
        """Apply pre-transform to all samples."""
        # This could be used for data augmentation or preprocessing
        # Implementation depends on specific requirements
        pass
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Data: PyTorch Geometric Data object
        """
        start_idx, end_idx = self.sample_indices[idx]
        
        # Extract motion data for this sample
        sample_data = self.motion_data.iloc[start_idx:end_idx].copy()
        
        # Create graph
        graph_data = create_motion_graph(
            motion_data=sample_data,
            marker_names=self.marker_names,
            connectivity_type=self.connectivity_type,
            include_temporal_edges=self.include_temporal_edges,
            temporal_connections=self.temporal_connections
        )
        
        # Add label if available
        if self.labels is not None:
            graph_data.y = self.labels[idx]
        
        # Add sample metadata
        graph_data.sample_idx = idx
        graph_data.time_range = (
            sample_data['Time'].iloc[0], 
            sample_data['Time'].iloc[-1]
        )
        
        # Apply transform if provided
        if self.transform is not None:
            graph_data = self.transform(graph_data)
        
        return graph_data
    
    def get_marker_names(self) -> List[str]:
        """Get the list of marker names."""
        return self.marker_names.copy()
    
    def get_sample_info(self, idx: int) -> Dict:
        """
        Get information about a specific sample.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Dict: Sample information
        """
        start_idx, end_idx = self.sample_indices[idx]
        sample_data = self.motion_data.iloc[start_idx:end_idx]
        
        info = {
            'sample_idx': idx,
            'start_frame': sample_data['Frame'].iloc[0],
            'end_frame': sample_data['Frame'].iloc[-1],
            'start_time': sample_data['Time'].iloc[0],
            'end_time': sample_data['Time'].iloc[-1],
            'num_frames': len(sample_data),
            'duration': sample_data['Time'].iloc[-1] - sample_data['Time'].iloc[0]
        }
        
        if self.labels is not None:
            info['label'] = self.labels[idx].item()
        
        return info
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dict: Dataset statistics
        """
        stats = {
            'num_samples': len(self),
            'num_markers': len(self.marker_names),
            'total_frames': len(self.motion_data),
            'sampling_rate': self.motion_data['Time'].iloc[1] - self.motion_data['Time'].iloc[0] if len(self.motion_data) > 1 else 0,
            'duration': self.motion_data['Time'].iloc[-1] - self.motion_data['Time'].iloc[0],
            'window_size': self.window_size,
            'stride': self.stride
        }
        
        # Add label statistics if available
        if self.labels is not None:
            unique_labels, counts = torch.unique(self.labels, return_counts=True)
            stats['num_classes'] = len(unique_labels)
            stats['class_distribution'] = {
                label.item(): count.item() 
                for label, count in zip(unique_labels, counts)
            }
        
        return stats


def collate_motion_graphs(batch: List[Data]) -> Batch:
    """
    Custom collate function for batching motion graphs.
    
    Args:
        batch (List[Data]): List of Data objects
        
    Returns:
        Batch: Batched Data object
    """
    return Batch.from_data_list(batch)
