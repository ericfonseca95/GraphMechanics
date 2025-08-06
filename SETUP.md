# GraphMechanics Setup Guide

This guide will help you set up the GraphMechanics package for processing OpenCap biomechanical data into graph datasets suitable for machine learning.

## Quick Start

1. **Clone and Install**
   ```bash
   cd GraphMechanics
   pip install -e .
   ```

2. **Process OpenCap Data**
   ```bash
   cd scripts
   python process_opencap_data.py --data-dir ../Data --output-dir opencap_unified_dataset
   ```

3. **Explore the Dataset**
   ```bash
   python example_usage.py
   ```

## Detailed Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Environment Setup

We recommend using a virtual environment:

```bash
# Create virtual environment
python -m venv graphmechanics_env

# Activate environment
# On Linux/Mac:
source graphmechanics_env/bin/activate
# On Windows:
graphmechanics_env\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Install from requirements file
pip install -r requirements.txt

# Or install package in development mode (recommended)
pip install -e .
```

### Step 3: OpenSim Installation (Optional but Recommended)

If you need the full OpenSim functionality:

```bash
# Using conda (recommended)
conda install -c opensim-org opensim

# Or using pip (may have limitations)
pip install opensim
```

## Data Preparation

### Expected OpenCap Data Structure

Your OpenCap data should be organized as follows:

```
Data/
├── OpenCapData_session1_id/
│   ├── sessionMetadata.yaml
│   └── OpenSimData/
│       ├── Model/
│       │   └── LaiUhlrich2022_scaled.osim
│       └── Kinematics/
│           ├── motion1.mot
│           ├── motion2.mot
│           └── ...
├── OpenCapData_session2_id/
│   └── ...
└── ...
```

### Downloading OpenCap Data

If you're starting from scratch, you can download OpenCap data:

1. Visit [OpenCap.ai](https://opencap.ai)
2. Download session data in OpenSim format
3. Extract to your Data directory
4. Ensure each session has both `.osim` model files and `.mot` motion files

## Usage Workflows

### Workflow 1: Basic Dataset Creation

```bash
# Process all OpenCap data into a unified dataset
python scripts/process_opencap_data.py \
    --data-dir Data \
    --output-dir unified_dataset \
    --sequence-length 8 \
    --export-formats numpy,pytorch,json
```

### Workflow 2: Subject-Specific Processing

```bash
# Process specific sessions
python scripts/process_opencap_data.py \
    --data-dir Data \
    --sessions session1_id,session2_id \
    --output-dir subject_specific_dataset
```

### Workflow 3: Motion-Type Filtering

```bash
# Process only jumping and running motions
python scripts/process_opencap_data.py \
    --data-dir Data \
    --motion-types jumping,running \
    --output-dir motion_specific_dataset
```

### Workflow 4: Memory-Constrained Processing

```bash
# Limit frames for large datasets
python scripts/process_opencap_data.py \
    --data-dir Data \
    --max-frames 1000 \
    --output-dir limited_dataset
```

## Jupyter Notebook Usage

### Starting Jupyter

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

### Key Notebooks

1. **`examples/graphmechanics_demo.ipynb`**: Basic introduction to GraphMechanics
2. **`notebooks/opensim_time_series_graph_dataset.ipynb`**: Comprehensive dataset creation tutorial
3. **`notebooks/test_package_integration.ipynb`**: Package testing and validation

## Programmatic Usage

### Basic Dataset Loading

```python
from graphmechanics import OpenSimGraphTimeSeriesDataset

# Load processed dataset
dataset = OpenSimGraphTimeSeriesDataset.load_from_numpy('unified_dataset/opencap_unified_graphs.npz')

# Explore the data
print(f"Dataset contains {len(dataset.frame_graphs)} graphs")
print(f"Each graph has {dataset.frame_graphs[0].x.shape[0]} nodes")
```

### Custom Processing

```python
from graphmechanics import OpenSimModelParser, OpenSimTimeSeriesGraphBuilder

# Parse OpenSim model
parser = OpenSimModelParser('path/to/model.osim')
parser.parse_model()

# Create graph builder
builder = OpenSimTimeSeriesGraphBuilder()
builder.set_model(parser.model_data)

# Process motion data
motion_data = builder.load_motion_data('path/to/motion.mot')
graphs = builder.create_time_series_graphs(motion_data)
```

### Machine Learning Integration

```python
from torch_geometric.loader import DataLoader

# Create data loaders
sequences = dataset.create_custom_sequences(sequence_length=10, overlap=5)
train_loader = dataset.get_dataloader(sequences, batch_size=32, shuffle=True)

# Use in training loop
for batch in train_loader:
    # batch.x: node features
    # batch.edge_index: graph connectivity
    # batch.edge_attr: edge features
    # batch.batch: batch assignment
    pass
```

## Advanced Configuration

### Customizing Graph Construction

```python
# Custom joint angle selection
builder.set_joint_angles(['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r'])

# Custom muscle properties
builder.set_muscle_properties(['max_isometric_force', 'optimal_fiber_length'])

# Custom sequence parameters
sequences = dataset.create_custom_sequences(
    sequence_length=15,
    overlap=10,
    stride=2
)
```

### Performance Optimization

```python
# Use PyTorch DataLoader with multiple workers
dataloader = dataset.get_dataloader(
    sequences=sequences,
    batch_size=64,
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # GPU optimization
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure package is installed in development mode
   pip install -e .
   ```

2. **OpenSim Import Issues**
   ```bash
   # Install OpenSim via conda
   conda install -c opensim-org opensim
   ```

3. **Memory Issues**
   ```bash
   # Use max-frames limitation
   python scripts/process_opencap_data.py --max-frames 500
   ```

4. **CUDA/GPU Issues**
   ```bash
   # Check PyTorch CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Getting Help

- **Documentation**: Check notebook examples and inline documentation
- **Issues**: Report bugs and issues on the project repository
- **Community**: Join discussions on relevant biomechanics and ML forums

## Next Steps

1. **Explore Examples**: Run the provided notebooks to understand the workflow
2. **Process Your Data**: Use the batch processing script on your OpenCap data
3. **Build Models**: Create graph neural networks for your specific use case
4. **Contribute**: Help improve the package by contributing code or reporting issues

## Development Setup

If you want to contribute to GraphMechanics:

```bash
# Clone in development mode
git clone <repository-url>
cd GraphMechanics

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Run linting
flake8 graphmechanics/
black graphmechanics/
```
