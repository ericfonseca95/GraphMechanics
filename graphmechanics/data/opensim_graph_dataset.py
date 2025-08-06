"""
OpenSim Graph Time-Series Dataset - Comprehensive Class

This module provides a comprehensive, unified class for creating, managing, and analyzing 
OpenSim time-series graph datasets with joint angles as node features and muscle 
properties/geometries as edge features.

Author: AI Assistant
Created: August 4, 2025
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Add networkx import with fallback
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available - some visualization features will be limited")


class OpenSimGraphTimeSeriesDataset:
    """
    Comprehensive class for creating, managing, and analyzing OpenSim time-series graph datasets.
    
    This class consolidates all functionality for:
    - Parsing OpenSim data
    - Creating joint-angle graphs with muscle edge features
    - Enhancing with derivatives (velocity, acceleration)
    - Sequence generation with flexible parameters
    - Advanced visualizations and analysis
    - Export/import in multiple formats
    - Custom sequencing and reloading
    
    Example:
        >>> # Create new dataset
        >>> dataset = OpenSimGraphTimeSeriesDataset(
        ...     model_path="model.osim",
        ...     motion_path="motion.mot"
        ... )
        >>> 
        >>> # Create graphs with derivatives
        >>> dataset.create_frame_graphs(add_derivatives=True)
        >>> 
        >>> # Create sequences
        >>> sequences = dataset.create_custom_sequences(
        ...     sequence_length=10, overlap=5
        ... )
        >>> 
        >>> # Export and analyze
        >>> dataset.export_numpy()
        >>> dataset.visualize_graph_structure()
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 motion_path: Optional[str] = None,
                 output_dir: str = "opensim_graph_dataset"):
        """
        Initialize the OpenSim Graph Time-Series Dataset.
        
        Args:
            model_path: Path to OpenSim model file (.osim)
            motion_path: Path to OpenSim motion file (.mot)
            output_dir: Directory for saving outputs
        """
        self.model_path = model_path
        self.motion_path = motion_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize parsers
        self.model_parser = None
        self.motion_parser = None
        self.graph_builder = None
        
        # Data storage
        self.frame_graphs = []
        self.sequences = []
        self.metadata = {}
        
        # Load parsers if paths provided
        if model_path:
            self.load_model(model_path)
        if motion_path:
            self.load_motion(motion_path)
    
    def load_model(self, model_path: str) -> None:
        """Load OpenSim model and initialize graph builder."""
        try:
            from graphmechanics.utils.opensim_parser import OpenSimModelParser
            from graphmechanics.data.opensim_time_series_graph_builder import OpenSimTimeSeriesGraphBuilder
        except ImportError:
            raise ImportError("GraphMechanics package not found. Please ensure it's installed and in your path.")
        
        print(f"🔄 Loading OpenSim model: {Path(model_path).name}")
        
        self.model_path = model_path
        self.model_parser = OpenSimModelParser(model_path)
        self.graph_builder = OpenSimTimeSeriesGraphBuilder(self.model_parser)
        
        print(f"✅ Model loaded: {self.model_parser.model_name}")
        print(f"   🦴 Bodies: {len(self.model_parser.bodies)}")
        print(f"   🔗 Joints: {len(self.model_parser.joints)}")
        print(f"   📐 Coordinates: {len(self.model_parser.coordinates)}")
        print(f"   💪 Muscles: {len(self.model_parser.muscles)}")
    
    def load_motion(self, motion_path: str) -> None:
        """Load OpenSim motion data."""
        try:
            from graphmechanics.utils.opensim_parser import OpenSimMotionParser
        except ImportError:
            raise ImportError("GraphMechanics package not found. Please ensure it's installed and in your path.")
        
        print(f"🔄 Loading OpenSim motion: {Path(motion_path).name}")
        
        self.motion_path = motion_path
        self.motion_parser = OpenSimMotionParser(motion_path)
        
        print(f"✅ Motion loaded:")
        print(f"   📊 Frames: {len(self.motion_parser.data)}")
        print(f"   📐 Coordinates: {len(self.motion_parser.coordinate_names)}")
        if 'time' in self.motion_parser.data.columns:
            time_span = self.motion_parser.data['time'].max() - self.motion_parser.data['time'].min()
            print(f"   ⏰ Duration: {time_span:.3f}s")
    
    def create_frame_graphs(self, 
                          time_window: Optional[Tuple[float, float]] = None,
                          frame_step: int = 1,
                          add_derivatives: bool = True) -> List[Data]:
        """
        Create individual frame graphs with joint angles as node features.
        
        Args:
            time_window: Optional (start_time, end_time) window
            frame_step: Step size for frame sampling
            add_derivatives: Whether to add velocity and acceleration features
            
        Returns:
            List of PyTorch Geometric graph objects
        """
        if not self.motion_parser:
            raise ValueError("Motion data not loaded. Use load_motion() first.")
        
        if not self.graph_builder:
            # Create builder without model if not available
            try:
                from graphmechanics.data.opensim_time_series_graph_builder import OpenSimTimeSeriesGraphBuilder
            except ImportError:
                raise ImportError("GraphMechanics package not found. Please ensure it's installed and in your path.")
            self.graph_builder = OpenSimTimeSeriesGraphBuilder(self.model_parser)
        
        print(f"🔄 Creating frame graphs...")
        
        # Create base graphs
        self.frame_graphs = self.graph_builder.create_joint_angle_graphs(
            self.motion_parser, time_window, frame_step
        )
        
        # Add derivatives if requested
        if add_derivatives and len(self.frame_graphs) >= 3:
            print(f"🔄 Adding velocity and acceleration features...")
            self.frame_graphs = self.graph_builder.enhance_graphs_with_derivatives(
                self.frame_graphs
            )
        
        # Update metadata
        self._update_metadata()
        
        print(f"✅ Created {len(self.frame_graphs)} frame graphs")
        return self.frame_graphs
    
    def create_sequences(self,
                        sequence_length: int = 10,
                        overlap: int = 5,
                        time_window: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        """
        Create sequences of graphs for temporal learning.
        
        Args:
            sequence_length: Number of frames per sequence
            overlap: Number of overlapping frames between sequences
            time_window: Optional time window
            
        Returns:
            List of sequence dictionaries
        """
        if not self.motion_parser:
            raise ValueError("Motion data not loaded. Use load_motion() first.")
        
        if not self.graph_builder:
            try:
                from graphmechanics.data.opensim_time_series_graph_builder import OpenSimTimeSeriesGraphBuilder
            except ImportError:
                raise ImportError("GraphMechanics package not found. Please ensure it's installed and in your path.")
            self.graph_builder = OpenSimTimeSeriesGraphBuilder(self.model_parser)
        
        print(f"🔄 Creating sequences (length={sequence_length}, overlap={overlap})...")
        
        self.sequences = self.graph_builder.create_sequence_graphs(
            self.motion_parser, sequence_length, overlap, time_window
        )
        
        print(f"✅ Created {len(self.sequences)} sequences")
        return self.sequences
    
    def create_custom_sequences(self, 
                              graphs: Optional[List[Data]] = None,
                              sequence_length: int = 10, 
                              overlap: int = 0, 
                              stride: int = 1) -> List[Dict[str, Any]]:
        """
        Create custom sequences from graphs with flexible parameters.
        
        Args:
            graphs: List of graphs (uses self.frame_graphs if None)
            sequence_length: Number of frames per sequence
            overlap: Number of overlapping frames between sequences
            stride: Step size between sequence starts
            
        Returns:
            List of sequence dictionaries
        """
        if graphs is None:
            graphs = self.frame_graphs
            
        if not graphs:
            raise ValueError("No graphs available. Create frame graphs first.")
        
        print(f"🔄 Creating custom sequences (len={sequence_length}, overlap={overlap}, stride={stride})...")
        
        sequences = []
        
        if overlap > 0:
            step_size = sequence_length - overlap
        else:
            step_size = stride
        
        for start_idx in range(0, len(graphs) - sequence_length + 1, step_size):
            end_idx = start_idx + sequence_length - 1
            
            if end_idx < len(graphs):
                sequence_graphs = graphs[start_idx:start_idx + sequence_length]
                
                sequence = {
                    'start_frame': sequence_graphs[0].frame_idx,
                    'end_frame': sequence_graphs[-1].frame_idx,
                    'start_time': float(sequence_graphs[0].time),
                    'end_time': float(sequence_graphs[-1].time),
                    'sequence_length': sequence_length,
                    'graphs': sequence_graphs,
                    'overlap': overlap,
                    'stride': step_size
                }
                sequences.append(sequence)
        
        print(f"✅ Created {len(sequences)} custom sequences")
        return sequences
    
    def _update_metadata(self) -> None:
        """Update dataset metadata."""
        if not self.frame_graphs:
            return
        
        sample_graph = self.frame_graphs[0]
        
        self.metadata = {
            'creation_info': {
                'timestamp': datetime.now().isoformat(),
                'model_file': str(self.model_path) if self.model_path else None,
                'motion_file': str(self.motion_path) if self.motion_path else None,
            },
            'dataset_info': {
                'total_frames': len(self.frame_graphs),
                'coordinates': sample_graph.coordinate_names,
                'num_nodes': sample_graph.x.shape[0],
                'num_edges': sample_graph.edge_index.shape[1],
                'node_features': sample_graph.x.shape[1],
                'edge_features': sample_graph.edge_attr.shape[1] if sample_graph.edge_attr is not None else 0,
                'time_span': {
                    'start': float(self.frame_graphs[0].time),
                    'end': float(self.frame_graphs[-1].time),
                    'duration': float(self.frame_graphs[-1].time - self.frame_graphs[0].time)
                }
            }
        }
    
    def export_numpy(self, filename: Optional[str] = None) -> str:
        """Export dataset to NumPy format."""
        if not self.frame_graphs:
            raise ValueError("No frame graphs to export. Create graphs first.")
        
        if filename is None:
            filename = "frame_graphs.npz"
        
        export_path = self.output_dir / filename
        
        print(f"💾 Exporting to NumPy format: {export_path}")
        
        # Convert graphs to numpy-serializable format
        graphs_data = []
        for i, graph in enumerate(self.frame_graphs):
            graph_dict = {
                'node_features': graph.x.numpy(),
                'edge_index': graph.edge_index.numpy(),
                'edge_attr': graph.edge_attr.numpy() if graph.edge_attr is not None else None,
                'time': float(graph.time),
                'frame_idx': int(graph.frame_idx),
                'coordinate_names': graph.coordinate_names
            }
            graphs_data.append(graph_dict)
        
        # Save data
        np.savez_compressed(export_path, graphs=graphs_data)
        
        # Save metadata
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Exported {len(graphs_data)} graphs to {export_path}")
        return str(export_path)
    
    def export_pytorch_geometric(self, filename: Optional[str] = None) -> str:
        """Export dataset as PyTorch Geometric dataset."""
        if not self.frame_graphs:
            raise ValueError("No frame graphs to export. Create graphs first.")
        
        if filename is None:
            filename = "pytorch_geometric_dataset.pt"
        
        export_path = self.output_dir / filename
        
        print(f"💾 Exporting to PyTorch Geometric format: {export_path}")
        
        # Save using PyTorch's save function (handling the weights_only parameter)
        try:
            torch.save(self.frame_graphs, export_path, _use_new_zipfile_serialization=False)
        except TypeError:
            # Fallback for older PyTorch versions
            torch.save(self.frame_graphs, export_path)
        
        print(f"✅ Exported {len(self.frame_graphs)} graphs to {export_path}")
        return str(export_path)
    
    @classmethod
    def load_from_numpy(cls, 
                       numpy_path: str, 
                       metadata_path: Optional[str] = None,
                       output_dir: str = "opensim_graph_dataset") -> 'OpenSimGraphTimeSeriesDataset':
        """Load dataset from NumPy export."""
        print(f"📂 Loading dataset from NumPy: {numpy_path}")
        
        dataset = cls(output_dir=output_dir)
        
        # Load graphs
        graphs_data = np.load(numpy_path, allow_pickle=True)
        frame_graphs_data = graphs_data['graphs']
        
        # Convert back to PyTorch Geometric format
        dataset.frame_graphs = []
        for graph_dict in frame_graphs_data:
            data = Data(
                x=torch.tensor(graph_dict['node_features'], dtype=torch.float),
                edge_index=torch.tensor(graph_dict['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(graph_dict['edge_attr'], dtype=torch.float) if graph_dict['edge_attr'] is not None else None,
                time=graph_dict['time'],
                frame_idx=graph_dict['frame_idx'],
                coordinate_names=graph_dict['coordinate_names']
            )
            dataset.frame_graphs.append(data)
        
        # Load metadata
        if metadata_path is None:
            metadata_path = Path(numpy_path).parent / "dataset_metadata.json"
        
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                dataset.metadata = json.load(f)
        
        print(f"✅ Loaded {len(dataset.frame_graphs)} graphs from NumPy export")
        return dataset
    
    @classmethod
    def load_from_pytorch_geometric(cls, 
                                  pytorch_path: str,
                                  output_dir: str = "opensim_graph_dataset") -> 'OpenSimGraphTimeSeriesDataset':
        """Load dataset from PyTorch Geometric export."""
        print(f"📂 Loading dataset from PyTorch Geometric: {pytorch_path}")
        
        dataset = cls(output_dir=output_dir)
        
        # Load graphs (handling the weights_only parameter)
        try:
            dataset.frame_graphs = torch.load(pytorch_path, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            dataset.frame_graphs = torch.load(pytorch_path)
        
        print(f"✅ Loaded {len(dataset.frame_graphs)} graphs from PyTorch Geometric export")
        return dataset
    
    def get_dataloader(self, 
                      sequences: Optional[List[Dict]] = None,
                      batch_size: int = 16, 
                      shuffle: bool = False) -> DataLoader:
        """Create PyTorch Geometric DataLoader from sequences."""
        if sequences is None:
            sequences = self.sequences
        
        if not sequences:
            raise ValueError("No sequences available. Create sequences first.")
        
        # Flatten sequences into individual graphs
        all_graphs = []
        for seq_idx, sequence in enumerate(sequences):
            for graph_idx, graph in enumerate(sequence['graphs']):
                # Add sequence metadata to each graph
                graph.sequence_id = seq_idx
                graph.position_in_sequence = graph_idx
                graph.sequence_length = sequence['sequence_length']
                all_graphs.append(graph)
        
        return DataLoader(all_graphs, batch_size=batch_size, shuffle=shuffle)
    
    def analyze_sequences(self, sequences: Optional[List[Dict]] = None, name: str = "Sequences") -> None:
        """Analyze properties of sequence configuration."""
        if sequences is None:
            sequences = self.sequences
        
        if not sequences:
            print("❌ No sequences to analyze")
            return
        
        durations = [seq['end_time'] - seq['start_time'] for seq in sequences]
        time_coverage = sum(durations)
        total_duration = self.metadata.get('dataset_info', {}).get('time_span', {}).get('duration', 1.0)
        
        print(f"\n📈 {name} Analysis:")
        print(f"   📏 Sequences: {len(sequences)}")
        print(f"   ⏱️  Avg duration: {np.mean(durations):.3f}s")
        print(f"   🔄 Duration range: [{min(durations):.3f}s, {max(durations):.3f}s]")
        print(f"   📊 Total time coverage: {time_coverage:.3f}s")
        print(f"   🎯 Coverage efficiency: {time_coverage / total_duration:.1f}x")
        
        # Show first few sequences
        print(f"   📋 First 3 sequences:")
        for i, seq in enumerate(sequences[:3]):
            print(f"      {i+1}. Frames {seq['start_frame']}-{seq['end_frame']} "
                  f"({seq['start_time']:.3f}s-{seq['end_time']:.3f}s)")
    
    def visualize_graph_structure(self, graph_idx: int = 0) -> None:
        """Create comprehensive graph structure visualizations."""
        if not self.frame_graphs:
            raise ValueError("No frame graphs available. Create graphs first.")
        
        sample_graph = self.frame_graphs[graph_idx]
        node_features = sample_graph.x
        edge_features = sample_graph.edge_attr
        coordinate_names = sample_graph.coordinate_names
        
        print(f"🎨 Creating graph structure visualizations for frame {graph_idx}...")
        
        # Create 3x2 subplot layout for comprehensive analysis
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: Node Feature Distribution (Joint Angles)
        axes[0, 0].hist(node_features[:, 0].numpy(), bins=20, alpha=0.7, color='skyblue', 
                       edgecolor='black', linewidth=1)
        axes[0, 0].axvline(node_features[:, 0].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {node_features[:, 0].mean():.3f}')
        axes[0, 0].set_xlabel('Joint Angle Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Node Features Distribution (Joint Angles)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Additional Node Features (if available)
        if node_features.shape[1] > 1:
            feature_names = ['Joint Angle', 'Angular Velocity', 'Angular Acceleration']
            for feat_idx in range(min(3, node_features.shape[1])):
                feature_data = node_features[:, feat_idx].numpy()
                axes[0, 1].hist(feature_data, bins=15, alpha=0.6, 
                               label=feature_names[feat_idx], density=True)
            
            axes[0, 1].set_xlabel('Feature Value')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Multiple Node Features Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Only Joint Angles\nAvailable', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=14,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[0, 1].set_title('Additional Node Features')
        
        # Plot 3: Edge Feature Analysis
        if edge_features is not None:
            axes[1, 0].hist(edge_features[:, 0].numpy(), bins=20, alpha=0.7, color='lightcoral',
                           edgecolor='black', linewidth=1)
            axes[1, 0].axvline(edge_features[:, 0].mean(), color='darkred', linestyle='--',
                              linewidth=2, label=f'Mean: {edge_features[:, 0].mean():.3f}')
            axes[1, 0].set_xlabel('Joint Distance')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Edge Features Distribution (Joint Distances)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Edge Features\nAvailable', ha='center', va='center',
                           transform=axes[1, 0].transAxes, fontsize=14,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 0].set_title('Edge Features Distribution')
        
        # Plot 4: Feature Correlation (if multiple edge features)
        if edge_features is not None and edge_features.shape[1] > 1:
            distances = edge_features[:, 0].numpy()
            forces = edge_features[:, 1].numpy()
            
            scatter = axes[1, 1].scatter(distances, forces, alpha=0.6, c=range(len(distances)), 
                                       cmap='plasma', s=50, edgecolors='black', linewidth=0.5)
            
            # Add correlation coefficient
            edge_correlation = np.corrcoef(distances, forces)[0, 1]
            axes[1, 1].text(0.05, 0.95, f'Correlation: {edge_correlation:.3f}', 
                           transform=axes[1, 1].transAxes, fontsize=12, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            axes[1, 1].set_xlabel('Joint Distance')
            axes[1, 1].set_ylabel('Muscle Force Sum')
            axes[1, 1].set_title('Distance vs Muscle Force Correlation')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=axes[1, 1], shrink=0.8, label='Edge Index')
        else:
            axes[1, 1].text(0.5, 0.5, 'Need Multiple Edge Features\nfor Correlation Analysis', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 1].set_title('Feature Correlation Analysis')
        
        # Plot 5 & 6: Network Visualization (if NetworkX available)
        if NETWORKX_AVAILABLE:
            self._create_network_visualization(sample_graph, axes[2, 0], axes[2, 1])
        else:
            for ax in [axes[2, 0], axes[2, 1]]:
                ax.text(0.5, 0.5, 'NetworkX Required\nfor Network Visualization\npip install networkx', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                ax.set_title('Network Visualization')
        
        plt.suptitle(f'Comprehensive Graph Analysis\nFrame {sample_graph.frame_idx} at t={sample_graph.time:.3f}s', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        self._print_graph_statistics(sample_graph)
    
    def _create_network_visualization(self, graph: Data, ax1, ax2) -> None:
        """Create network visualizations using NetworkX."""
        edge_index = graph.edge_index
        coordinate_names = graph.coordinate_names  
        node_features = graph.x
        edge_features = graph.edge_attr
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes with features
        for i, coord_name in enumerate(coordinate_names):
            if coord_name != 'time' and i < node_features.shape[0]:
                node_value = node_features[i, 0].item()
                velocity = node_features[i, 1].item() if node_features.shape[1] > 1 else 0
                G.add_node(i, name=coord_name, angle=node_value, velocity=velocity)
        
        # Add edges
        edge_list = edge_index.t().numpy()
        for idx, edge in enumerate(edge_list):
            if (edge[0] < len(coordinate_names) and edge[1] < len(coordinate_names) and 
                coordinate_names[edge[0]] != 'time' and coordinate_names[edge[1]] != 'time' and
                edge[0] < node_features.shape[0] and edge[1] < node_features.shape[0]):
                
                edge_weight = 1.0
                if edge_features is not None and idx < edge_features.shape[0]:
                    edge_weight = edge_features[idx, 0].item()
                
                G.add_edge(edge[0], edge[1], weight=edge_weight)
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Plot 1: Colored by joint angles
        node_angles = [G.nodes[node]['angle'] for node in G.nodes()]
        node_colors = plt.cm.RdYlBu_r(plt.Normalize()(node_angles))
        node_sizes = [200 + abs(angle) * 10 for angle in node_angles]
        
        ax1.set_aspect('equal')
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.4, width=1, edge_color='gray')
        ax1.set_title('Graph Network (Colored by Joint Angles)')
        ax1.axis('off')
        
        # Plot 2: Colored by body regions
        node_colors_region = []
        for node in G.nodes():
            coord_name = coordinate_names[node].lower()
            if any(keyword in coord_name for keyword in ['hip', 'pelvis']):
                node_colors_region.append('red')
            elif any(keyword in coord_name for keyword in ['knee', 'ankle', 'subtalar']):
                node_colors_region.append('blue')
            elif any(keyword in coord_name for keyword in ['lumbar', 'thorax', 'neck']):
                node_colors_region.append('green')
            elif any(keyword in coord_name for keyword in ['shoulder', 'elbow', 'wrist']):
                node_colors_region.append('orange')
            else:
                node_colors_region.append('gray')
        
        ax2.set_aspect('equal')
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=node_colors_region, node_size=200, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.6, width=1, edge_color='gray')
        ax2.set_title('Graph Network (Colored by Body Regions)\nRed=Hip, Blue=Leg, Green=Spine, Orange=Arm')
        ax2.axis('off')
    
    def _print_graph_statistics(self, graph: Data) -> None:
        """Print comprehensive graph statistics."""
        node_features = graph.x
        edge_features = graph.edge_attr
        coordinate_names = graph.coordinate_names
        
        print("\n📊 Enhanced Graph Statistics:")
        print(f"   🎯 Time: {graph.time:.3f}s (frame {graph.frame_idx})")
        print(f"   🔸 Nodes: {node_features.shape[0]} joints")
        print(f"   📈 Node features: {node_features.shape[1]} per joint")
        if edge_features is not None:
            print(f"   🔗 Edges: {edge_features.shape[0]} connections")
            print(f"   📊 Edge features: {edge_features.shape[1]} per connection")
        
        print(f"\n📈 Node Feature Ranges:")
        print(f"   🎯 Joint angles: [{node_features[:, 0].min().item():.3f}, {node_features[:, 0].max().item():.3f}]")
        if node_features.shape[1] > 1:
            print(f"   ⚡ Angular velocities: [{node_features[:, 1].min().item():.3f}, {node_features[:, 1].max().item():.3f}]")
        if node_features.shape[1] > 2:
            print(f"   🚀 Angular accelerations: [{node_features[:, 2].min().item():.3f}, {node_features[:, 2].max().item():.3f}]")
        
        if edge_features is not None:
            print(f"\n🔗 Edge Feature Ranges:")
            print(f"   📏 Joint distances: [{edge_features[:, 0].min().item():.3f}, {edge_features[:, 0].max().item():.3f}]")
            if edge_features.shape[1] > 1:
                print(f"   💪 Muscle forces: [{edge_features[:, 1].min().item():.1f}, {edge_features[:, 1].max().item():.1f}]")
    
    def save_sequences_config(self, sequences: List[Dict], config_name: str) -> str:
        """Save sequence configuration to JSON file."""
        # Convert sequences to serializable format
        serializable_sequences = []
        for seq in sequences:
            seq_dict = {
                'start_frame': seq['start_frame'],
                'end_frame': seq['end_frame'],
                'start_time': seq['start_time'],
                'end_time': seq['end_time'],
                'sequence_length': seq['sequence_length'],
                'overlap': seq.get('overlap', 0),
                'stride': seq.get('stride', seq['sequence_length'])
            }
            serializable_sequences.append(seq_dict)
        
        # Save to file
        config_file = self.output_dir / f"{config_name}.json"
        with open(config_file, 'w') as f:
            json.dump(serializable_sequences, f, indent=2)
        
        print(f"✅ Saved {len(serializable_sequences)} sequences → {config_file.name}")
        return str(config_file)
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        info = f"OpenSimGraphTimeSeriesDataset("
        if self.model_path:
            info += f"\n  Model: {Path(self.model_path).name}"
        if self.motion_path:
            info += f"\n  Motion: {Path(self.motion_path).name}"
        if self.frame_graphs:
            info += f"\n  Frames: {len(self.frame_graphs)}"
        if self.sequences:
            info += f"\n  Sequences: {len(self.sequences)}"
        info += f"\n  Output: {self.output_dir}"
        info += "\n)"
        return info


# Convenience function for quick dataset creation
def create_opensim_graph_dataset(model_path: str, 
                                motion_path: str,
                                output_dir: str = "opensim_graph_dataset",
                                sequence_length: int = 10,
                                overlap: int = 5,
                                add_derivatives: bool = True) -> OpenSimGraphTimeSeriesDataset:
    """
    Convenience function to quickly create an OpenSim graph dataset.
    
    Args:
        model_path: Path to OpenSim model file
        motion_path: Path to OpenSim motion file
        output_dir: Output directory
        sequence_length: Length of sequences to create
        overlap: Overlap between sequences
        add_derivatives: Whether to add velocity/acceleration features
        
    Returns:
        Configured OpenSimGraphTimeSeriesDataset instance
    """
    dataset = OpenSimGraphTimeSeriesDataset(model_path, motion_path, output_dir)
    dataset.create_frame_graphs(add_derivatives=add_derivatives)
    dataset.create_sequences(sequence_length=sequence_length, overlap=overlap)
    return dataset


if __name__ == "__main__":
    # Example usage
    print("OpenSim Graph Time-Series Dataset - Comprehensive Class")
    print("Usage:")
    print("  from opensim_graph_dataset import OpenSimGraphTimeSeriesDataset")
    print("  dataset = OpenSimGraphTimeSeriesDataset('model.osim', 'motion.mot')")
    print("  dataset.create_frame_graphs(add_derivatives=True)")
    print("  sequences = dataset.create_custom_sequences(sequence_length=10, overlap=5)")
    print("  dataset.export_numpy()")
    print("  dataset.visualize_graph_structure()")
