"""
Utility functions for training GraphMechanics models.
"""

import os
import torch
from typing import Dict, List, Tuple
from collections import Counter


def load_multiple_movements(movement_files: Dict[str, str], frame_window: int = 20) -> Tuple[List, Dict[str, int]]:
    """
    Load multiple TRC files and create labeled dataset.
    
    Args:
        movement_files: Dictionary mapping movement names to file paths
        frame_window: Number of frames per graph window
        
    Returns:
        Tuple of (labeled_data_list, label_mapping)
    """
    from graphmechanics.utils.trc_parser import TRCParser
    from graphmechanics.data.graph_builder import MotionGraphConverter
    
    movement_graphs = {}
    converter = MotionGraphConverter()
    
    for movement_name, file_path in movement_files.items():
        if os.path.exists(file_path):
            print(f"Processing {movement_name}...")
            
            # Parse TRC file
            parser = TRCParser(file_path)
            trc_data = parser.to_graph_format()
            
            # Convert to PyG graphs
            graphs = converter.trc_to_pyg_data(trc_data, frame_window=frame_window)
            
            movement_graphs[movement_name] = {
                'graphs': graphs,
                'summary': parser.get_summary()
            }
            
            print(f"  Created {len(graphs)} graph windows")
    
    return create_movement_dataset(movement_graphs)


def create_movement_dataset(movement_graphs: Dict) -> Tuple[List, Dict[str, int]]:
    """
    Create labeled dataset from movement graphs.
    
    Args:
        movement_graphs: Dictionary of movement data
        
    Returns:
        Tuple of (labeled_data_list, label_mapping)
    """
    labeled_data = []
    label_map = {name: idx for idx, name in enumerate(movement_graphs.keys())}
    
    for movement_name, movement_info in movement_graphs.items():
        label = label_map[movement_name]
        for graph in movement_info['graphs']:
            graph.y = torch.tensor([label], dtype=torch.long)
            labeled_data.append(graph)
    
    print(f"Created dataset with {len(labeled_data)} graphs")
    print(f"Movement types: {list(label_map.keys())}")
    
    # Show data distribution
    label_counts = Counter([graph.y.item() for graph in labeled_data])
    for movement_name, label in label_map.items():
        count = label_counts[label]
        print(f"  {movement_name}: {count} graphs")
    
    return labeled_data, label_map
