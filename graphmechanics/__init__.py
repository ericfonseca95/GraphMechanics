"""
GraphMechanics: Graph Neural Networks for Biomechanical Motion Analysis

A comprehensive package for analyzing human motion capture data using graph neural networks.
Provides tools for parsing TRC files, constructing anatomically-aware graphs, and training
graph transformer models for motion classification and analysis.
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "team@graphmechanics.org"

# Core utilities
try:
    from .utils import (
        TRCParser, OpenSimParser, OpenSimModelParser, OpenSimMotionParser,
        GraphMechanicsValidator, BiomechanicalLossFunctions, 
        GraphMechanicsPerformanceAnalyzer, BiomechanicalValidator
    )
    from .data.graph_builder import (
        MotionGraphConverter, KinematicGraphBuilder, 
        BiomechanicalConstraints, JointLimits
    )
    from .data.motion_graph_dataset import MotionGraphDataset
    from .data.opensim_time_series_graph_builder import OpenSimTimeSeriesGraphBuilder
    from .data.opensim_graph_dataset import OpenSimGraphTimeSeriesDataset, create_opensim_graph_dataset
    
    __all__ = [
        # Parsers
        'TRCParser', 'OpenSimParser', 'OpenSimModelParser', 'OpenSimMotionParser',
        # Graph builders and constraints
        'MotionGraphConverter', 'KinematicGraphBuilder', 'BiomechanicalConstraints', 'JointLimits',
        # Datasets
        'MotionGraphDataset', 'OpenSimTimeSeriesGraphBuilder', 'OpenSimGraphTimeSeriesDataset', 'create_opensim_graph_dataset',
        # Validation and analysis
        'GraphMechanicsValidator', 'BiomechanicalLossFunctions', 'GraphMechanicsPerformanceAnalyzer', 'BiomechanicalValidator'
    ]
    
except ImportError as e:
    print(f"Warning: Could not import some GraphMechanics components: {e}")
    __all__ = []

# Models (optional, requires PyTorch Geometric)
try:
    from .models.graph_transformer import GraphTransformer
    __all__.append('GraphTransformer')
except ImportError:
    pass

# Training utilities (optional, requires PyTorch Geometric)
try:
    from .training.motion_classifier import MotionClassificationTask
    from .training.utils import load_multiple_movements, create_movement_dataset
    __all__.extend(['MotionClassificationTask', 'load_multiple_movements', 'create_movement_dataset'])
except ImportError:
    pass
