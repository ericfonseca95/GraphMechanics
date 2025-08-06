# GraphMechanics Scripts 📜

This folder contains utility scripts for processing and analyzing biomechanical motion data using the GraphMechanics package.

## Scripts Overview

### 🔧 `process_opencap_data.py`

**Purpose**: Batch processes all OpenCap data folders and converts them into a single comprehensive graph dataset.

**Features**:
- ✅ Automatically discovers all OpenCap session folders
- ✅ Processes multiple subjects with multiple motion types
- ✅ Combines all sessions into a unified dataset
- ✅ Adds comprehensive metadata (subject info, motion types, session details)
- ✅ Supports flexible sequence generation
- ✅ Exports in multiple formats (NumPy, PyTorch Geometric, JSON)
- ✅ Generates detailed reports and statistics
- ✅ Motion type classification (jumping, running, squatting, etc.)

## Usage Examples

### Basic Usage
```bash
# Process all OpenCap data with default settings
python scripts/process_opencap_data.py

# This will:
# - Look for OpenCap folders in ./Data/
# - Create sequences of length 10 with overlap 5
# - Add velocity and acceleration features
# - Export to ./opencap_unified_dataset/
```

### Advanced Usage
```bash
# Custom configuration
python scripts/process_opencap_data.py \
    --data-dir /path/to/opencap/data \
    --output-dir my_custom_dataset \
    --sequence-length 8 \
    --overlap 4 \
    --stride 2

# Skip certain processing steps
python scripts/process_opencap_data.py \
    --no-derivatives \
    --no-sequences \
    --no-export
```

### Command-Line Options

```bash
python process_opencap_data.py --help
```

Available options:
- `--data-dir`: Path to directory containing OpenCap session folders
- `--output-dir`: Output directory for unified dataset
- `--sessions`: Process specific sessions (comma-separated IDs)
- `--motion-types`: Process specific motion types (comma-separated)
- `--sequence-length`: Default sequence length for graph sequences
- `--max-frames`: Maximum frames to process per motion (for memory management)
- `--export-formats`: Export formats (numpy, pytorch, json)
- `--verbose`: Enable detailed logging

## Usage Example

After running the batch processing script, you can use the generated dataset with the provided example script:

```bash
# Run the usage example
cd scripts
python example_usage.py
```

The example script (`example_usage.py`) demonstrates:
- Loading the unified dataset
- Exploring comprehensive metadata and statistics
- Filtering data by subject, motion type, and time
- Creating custom sequences for different use cases
- Setting up DataLoaders for machine learning training
- Common usage patterns for research and development

### Key Features Demonstrated

1. **Dataset Exploration**: Comprehensive analysis of subjects, motion types, and temporal patterns
2. **Flexible Filtering**: Filter by subject, motion type, session, or time range
3. **Custom Sequences**: Create sequences of different lengths and overlaps for various use cases
4. **Training Setup**: Configure DataLoaders for graph neural network training
5. **Usage Patterns**: Common patterns for motion classification, subject analysis, and temporal studies

## Expected Data Structure

## Expected Data Structure

The script expects OpenCap data to be organized as follows:

```
Data/
├── OpenCapData_session1_id/
│   ├── sessionMetadata.yaml           # Subject info, model settings
│   └── OpenSimData/
│       ├── Model/
│       │   └── LaiUhlrich2022_scaled.osim    # OpenSim model
│       └── Kinematics/
│           ├── jump.mot               # Motion files
│           ├── run.mot
│           ├── squat.mot
│           └── ...
├── OpenCapData_session2_id/
│   └── ...
└── ...
```

## Output Structure

The script generates a comprehensive dataset with the following structure:

```
opencap_unified_dataset/
├── opencap_unified_graphs.npz         # NumPy format graphs
├── opencap_unified_graphs.pt          # PyTorch Geometric format
├── opencap_dataset_metadata.json      # Comprehensive dataset metadata
├── motion_metadata.json               # Per-motion metadata
├── opencap_sequences.json             # Sequence configurations
├── dataset_report.txt                 # Human-readable report
└── dataset_metadata.json              # GraphMechanics metadata
```

## Motion Type Classification

The script automatically classifies motions based on filename patterns:

| Motion Type | Keywords | Examples |
|-------------|----------|----------|
| `vertical_jump` | "jump" + "vertical" | `1legverticaljump.mot` |
| `drop_landing` | "jump" + "drop" | `2legjumpdroplanding.mot` |
| `jump` | "jump" | `jump.mot` |
| `running` | "run" | `run2.mot` |
| `walking` | "walk" | `walk.mot` |
| `squatting` | "squat" | `1legsquat.mot` |
| `cutting` | "cut" | `cut.mot` |
| `balance` | "balance" | `YBALANCE.mot` |
| `alignment` | "alignment" | `alignment.mot` |
| `other` | *none of above* | Custom motions |

## Sample Output

```
🚀 OpenCap Data Batch Processor
========================================
📁 Data directory: Data
📁 Output directory: opencap_unified_dataset

🔍 Discovering OpenCap sessions...
Found 2 OpenCap session folders

📂 Processing folder: OpenCapData_7272a71a-e70a-4794-a253-39e11cb7542c
   ✅ Loaded session metadata
   ✅ Model file: LaiUhlrich2022_scaled.osim
   ✅ Motion files: 4
   ✅ Subject: afalisse
      - cut.mot
      - run2.mot
      - jump.mot
      - walk.mot

📊 Total valid sessions found: 2

🔄 Processing session: OpenCapData_7272a71a-e70a-4794-a253-39e11cb7542c
   👤 Subject: afalisse
   📏 Height: 1.96m, Mass: 80.0kg
   ✅ Loaded model: LaiUhlrich2022

   🏃 Processing motion 1/4: cut
      📊 Frames: 157
      📐 Coordinates: 34
      ⏰ Duration: 2.600s
      ✅ Created 157 frame graphs

📊 OpenCap Unified Dataset Report
================================================

📈 Dataset Summary:
   🎬 Total sessions: 2
   🏃 Total motions: 11
   🔢 Total frame graphs: 1,847
   ⏰ Total duration: 30.6s
   👥 Unique subjects: 2

🎯 Motion Type Distribution:
   📊 jumping: 4 motions (36.4%)
   📊 running: 2 motions (18.2%)
   📊 cutting: 1 motion (9.1%)
   📊 squatting: 2 motions (18.2%)
   📊 balance: 1 motion (9.1%)
   📊 alignment: 1 motion (9.1%)

✅ Created 369 sequences
💾 Exporting unified dataset...
✅ Export complete!

🎉 Processing completed successfully!
```

## Integration with GraphMechanics

The processed dataset is fully compatible with the GraphMechanics package:

```python
from graphmechanics import OpenSimGraphTimeSeriesDataset

# Load the unified dataset
dataset = OpenSimGraphTimeSeriesDataset.load_from_numpy(
    "opencap_unified_dataset/opencap_unified_graphs.npz"
)

# Access metadata
print(f"Total subjects: {len(dataset.metadata['dataset_summary']['subject_ids'])}")

# Filter by motion type or subject
subject_graphs = [g for g in dataset.frame_graphs if g.subject_id == 'afalisse']
jump_graphs = [g for g in dataset.frame_graphs if g.motion_type == 'jumping']

# Create custom sequences
custom_sequences = dataset.create_custom_sequences(
    sequence_length=15, 
    overlap=10
)

# Get DataLoader for training
train_loader = dataset.get_dataloader(custom_sequences, batch_size=32, shuffle=True)
```

## Requirements

- Python 3.7+
- GraphMechanics package (with all dependencies)
- PyYAML for metadata parsing
- Standard libraries: pathlib, json, datetime, argparse

## Troubleshooting

### Common Issues

1. **"No valid OpenCap sessions found"**
   - Check that your data directory contains folders starting with `OpenCapData_`
   - Verify each folder has the required `OpenSimData/Model/*.osim` and `OpenSimData/Kinematics/*.mot` files

2. **Import errors**
   - Ensure GraphMechanics package is properly installed
   - Check that you're running from the correct directory

3. **Memory issues with large datasets**
   - Use `--no-derivatives` to reduce memory usage
   - Process sessions individually if needed
   - Consider using smaller sequence lengths

### Getting Help

For issues specific to this script, check:
1. The console output for detailed error messages
2. The generated report files for dataset statistics
3. The GraphMechanics documentation for underlying functionality

---

**Ready to process your OpenCap data! 🚀**
