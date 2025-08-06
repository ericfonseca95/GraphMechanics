# GraphMechanics

A comprehensive PyTorch Geometric-based library for applying graph neural networks to biomechanical motion analysis, with specialized support for OpenSim and OpenCap data processing.

## 🚀 Quick Start

```bash
# Install the package
pip install -e .

# Process OpenCap data into graph datasets
cd scripts
python process_opencap_data.py --data-dir ../Data --output-dir opencap_dataset

# Explore the generated dataset
python example_usage.py
```

## Overview

GraphMechanics provides a complete pipeline for biomechanical data analysis using graph neural networks:

- **🔧 OpenSim Integration**: Native support for OpenSim models and motion files (.osim, .mot)
- **📊 OpenCap Processing**: Batch processing of OpenCap data into unified graph datasets
- **🕸️ Graph Construction**: Convert biomechanical data into graph structures with joint angles as nodes and muscle properties as edges
- **🧠 Graph Neural Networks**: PyTorch Geometric-based models for motion analysis, prediction, and classification
- **📈 Comprehensive Analysis**: Tools for extracting meaningful biomechanical features from graph representations

## Key Features

### 🎯 OpenSim & OpenCap Integration
- **OpenSim Model Parser**: Extract muscle properties, joint definitions, and anatomical constraints
- **Motion Data Processing**: Handle .mot files with kinematic and kinetic data
- **OpenCap Batch Processing**: Process multiple sessions and subjects into unified datasets
- **Metadata Preservation**: Maintain subject information, session data, and motion classifications

### 🔗 Advanced Graph Construction
- **Biomechanical Graphs**: Joint angles as nodes, muscle properties and joint distances as edges
- **Temporal Sequences**: Create time-series graph sequences for motion analysis
- **Flexible Features**: Customizable node and edge features based on biomechanical requirements
- **Multiple Export Formats**: NumPy, PyTorch Geometric, and JSON export options

### 🧠 Graph Neural Networks
- **GNN-Ready Datasets**: PyTorch Geometric InMemoryDataset classes with proper batching
- **Sequence Generation**: Create overlapping sequences for temporal pattern recognition
- **DataLoader Integration**: Seamless integration with PyTorch training pipelines
- **Batch Processing**: Efficient handling of large-scale biomechanical datasets

### 📊 Comprehensive Analysis Tools
- **Motion Classification**: Distinguish between different movement types (jumping, running, squatting)
- **Subject Analysis**: Cross-subject and subject-specific model development
- **Temporal Patterns**: Analyze movement progression and temporal dependencies
- **Statistical Insights**: Comprehensive dataset statistics and visualizations

## Installation

### Quick Installation
```bash
git clone <repository-url>
cd GraphMechanics
pip install -e .
```

### Full Setup with Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Install with OpenSim (recommended)
conda install -c opensim-org opensim
pip install -e .
```

**📖 See [SETUP.md](SETUP.md) for detailed installation instructions.**

## Quick Start

```python
from graphmechanics.utils import TRCParser
## 📖 Documentation & Examples

### Core Documentation
- **[SETUP.md](SETUP.md)**: Comprehensive installation and setup guide
- **[scripts/README.md](scripts/README.md)**: Batch processing documentation and troubleshooting

### Interactive Notebooks
- **[examples/graphmechanics_demo.ipynb](examples/graphmechanics_demo.ipynb)**: Basic introduction and tutorial
- **[notebooks/opensim_time_series_graph_dataset.ipynb](notebooks/opensim_time_series_graph_dataset.ipynb)**: Complete dataset creation workflow
- **[notebooks/test_package_integration.ipynb](notebooks/test_package_integration.ipynb)**: Package testing and validation

### Scripts & Tools
- **[scripts/process_opencap_data.py](scripts/process_opencap_data.py)**: Batch processing for OpenCap data
- **[scripts/example_usage.py](scripts/example_usage.py)**: Dataset usage examples and patterns

## 🔧 Usage Examples

### Basic OpenSim Graph Dataset Creation

```python
from graphmechanics import OpenSimGraphTimeSeriesDataset
from graphmechanics import OpenSimModelParser, OpenSimTimeSeriesGraphBuilder

# Parse OpenSim model
parser = OpenSimModelParser('LaiUhlrich2022_scaled.osim')
parser.parse_model()

# Create graph builder
builder = OpenSimTimeSeriesGraphBuilder()
builder.set_model(parser.model_data)

# Load motion data and create graphs
motion_data = builder.load_motion_data('walking_trial.mot')
graphs = builder.create_time_series_graphs(motion_data)

# Create dataset
dataset = OpenSimGraphTimeSeriesDataset(graphs)
sequences = dataset.create_custom_sequences(sequence_length=10)
dataloader = dataset.get_dataloader(sequences, batch_size=32)
```

### Batch Processing OpenCap Data

```python
from scripts.process_opencap_data import OpenCapDataProcessor

# Initialize processor
processor = OpenCapDataProcessor('Data/', 'unified_dataset/')

# Discover and process all sessions
sessions = processor.discover_sessions()
processor.process_all_sessions(sessions)

# Create unified dataset
unified_dataset = processor.create_unified_dataset()
```

### Training Graph Neural Networks

```python
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class MotionGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.classifier = torch.nn.Linear(32, num_classes)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)

# Use with your dataset
model = MotionGNN(num_features=graph.x.shape[1], num_classes=5)
```

## 🎯 Applications

- **🏥 Clinical Analysis**: Analyze movement patterns in patients with musculoskeletal disorders
- **🏃 Sports Performance**: Optimize athletic movement techniques and identify performance bottlenecks
- **🦴 Rehabilitation**: Monitor recovery progress and movement quality improvements
- **🤖 Motion Prediction**: Forecast future poses and movements for prosthetics and robotics
- **📊 Biomechanical Research**: Large-scale analysis of human movement patterns and variations  
- **Rehabilitation**: Track recovery progress and movement quality
- **Ergonomics**: Assess workplace movement patterns
- **Human-Computer Interaction**: Gesture recognition and motion interfaces

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GraphMechanics in your research, please cite:

```bibtex
@software{graphmechanics2025,
  title={GraphMechanics: Graph Neural Networks for Biomechanical Motion Analysis},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/GraphMechanics}
}
```
