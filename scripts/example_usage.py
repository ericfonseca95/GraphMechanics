#!/usr/bin/env python3
"""
OpenCap Dataset Usage Example

This script demonstrates how to load and use the unified OpenCap dataset
created by the process_opencap_data.py script.

Features demonstrated:
- Loading the unified dataset
- Exploring dataset metadata and statistics
- Filtering data by subject, motion type, and other criteria
- Creating custom sequences for different use cases
- Setting up DataLoaders for machine learning training
"""

import sys
from pathlib import Path
import json
import numpy as np
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import GraphMechanics components
from graphmechanics import OpenSimGraphTimeSeriesDataset

def load_unified_dataset(dataset_dir: str = "opencap_unified_dataset"):
    """
    Load the unified OpenCap dataset.
    
    Args:
        dataset_dir: Directory containing the unified dataset
        
    Returns:
        Loaded dataset instance
    """
    dataset_path = Path(dataset_dir)
    
    print(f"📂 Loading unified OpenCap dataset from: {dataset_path}")
    
    # Check if dataset exists
    numpy_file = dataset_path / "opencap_unified_graphs.npz"
    if not numpy_file.exists():
        print(f"❌ Dataset not found at {numpy_file}")
        print("💡 Run process_opencap_data.py first to create the unified dataset")
        return None
    
    # Load dataset
    dataset = OpenSimGraphTimeSeriesDataset.load_from_numpy(str(numpy_file))
    
    print(f"✅ Loaded {len(dataset.frame_graphs)} frame graphs")
    return dataset

def explore_dataset_metadata(dataset_dir: str = "opencap_unified_dataset"):
    """
    Explore the comprehensive metadata of the unified dataset.
    
    Args:
        dataset_dir: Directory containing the unified dataset
    """
    print(f"\n📊 Exploring Dataset Metadata")
    print("-" * 40)
    
    # Load comprehensive metadata
    metadata_file = Path(dataset_dir) / "opencap_dataset_metadata.json"
    motion_metadata_file = Path(dataset_dir) / "motion_metadata.json"
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"📈 Dataset Summary:")
        summary = metadata['dataset_summary']
        print(f"   🎬 Total sessions: {summary['total_sessions']}")
        print(f"   🏃 Total motions: {summary['total_motions']}")
        print(f"   🔢 Total frames: {summary['total_frames']}")
        print(f"   ⏰ Total duration: {summary['total_duration_seconds']:.1f}s")
        print(f"   👥 Unique subjects: {summary['unique_subjects']}")
        print(f"   📋 Subject IDs: {', '.join(summary['subject_ids'])}")
        
        print(f"\n🎯 Motion Type Distribution:")
        for motion_type, count in metadata['motion_type_distribution'].items():
            percentage = (count / summary['total_motions']) * 100
            print(f"   📊 {motion_type}: {count} motions ({percentage:.1f}%)")
        
        return metadata
    else:
        print(f"❌ Metadata file not found: {metadata_file}")
        return None

def analyze_dataset_graphs(dataset):
    """
    Analyze the loaded graph dataset in detail.
    
    Args:
        dataset: Loaded OpenSimGraphTimeSeriesDataset instance
    """
    print(f"\n🔍 Analyzing Graph Dataset")
    print("-" * 40)
    
    if not dataset or not dataset.frame_graphs:
        print("❌ No dataset or graphs available")
        return
    
    graphs = dataset.frame_graphs
    
    # Basic graph statistics
    print(f"📊 Graph Statistics:")
    print(f"   🔢 Total graphs: {len(graphs)}")
    
    sample_graph = graphs[0]
    print(f"   🔸 Nodes per graph: {sample_graph.x.shape[0]}")
    print(f"   📈 Node features: {sample_graph.x.shape[1]}")
    print(f"   🔗 Edges per graph: {sample_graph.edge_index.shape[1]}")
    if sample_graph.edge_attr is not None:
        print(f"   📊 Edge features: {sample_graph.edge_attr.shape[1]}")
    
    # Analyze by subject
    if hasattr(sample_graph, 'subject_id'):
        subjects = [g.subject_id for g in graphs]
        subject_counts = Counter(subjects)
        print(f"\n👥 Graphs by Subject:")
        for subject, count in subject_counts.items():
            percentage = (count / len(graphs)) * 100
            print(f"   👤 {subject}: {count} graphs ({percentage:.1f}%)")
    
    # Analyze by motion type
    if hasattr(sample_graph, 'motion_type'):
        motion_types = [g.motion_type for g in graphs]
        motion_counts = Counter(motion_types)
        print(f"\n🏃 Graphs by Motion Type:")
        for motion_type, count in motion_counts.items():
            percentage = (count / len(graphs)) * 100
            print(f"   🎯 {motion_type}: {count} graphs ({percentage:.1f}%)")
    
    # Time span analysis
    if hasattr(sample_graph, 'time'):
        times = [float(g.time) for g in graphs]
        print(f"\n⏰ Time Analysis:")
        print(f"   🕐 Time range: {min(times):.3f}s - {max(times):.3f}s")
        print(f"   📊 Total time span: {max(times) - min(times):.3f}s")

def filter_dataset_examples(dataset):
    """
    Demonstrate various ways to filter the dataset.
    
    Args:
        dataset: Loaded OpenSimGraphTimeSeriesDataset instance
    """
    print(f"\n🔧 Dataset Filtering Examples")
    print("-" * 40)
    
    if not dataset or not dataset.frame_graphs:
        print("❌ No dataset available")
        return
    
    graphs = dataset.frame_graphs
    
    # Filter by subject
    if hasattr(graphs[0], 'subject_id'):
        subjects = list(set(g.subject_id for g in graphs))
        if subjects:
            target_subject = subjects[0]
            subject_graphs = [g for g in graphs if g.subject_id == target_subject]
            print(f"🔍 Filter by subject '{target_subject}': {len(subject_graphs)} graphs")
    
    # Filter by motion type
    if hasattr(graphs[0], 'motion_type'):
        motion_types = list(set(g.motion_type for g in graphs))
        jumping_graphs = [g for g in graphs if 'jump' in g.motion_type]
        running_graphs = [g for g in graphs if g.motion_type == 'running']
        
        print(f"🔍 Filter by jumping motions: {len(jumping_graphs)} graphs")
        print(f"🔍 Filter by running motions: {len(running_graphs)} graphs")
    
    # Filter by time range
    if hasattr(graphs[0], 'time'):
        # Get graphs from first 1 second
        early_graphs = [g for g in graphs if float(g.time) <= 1.0]
        print(f"🔍 Filter by time (first 1s): {len(early_graphs)} graphs")
    
    # Filter by session
    if hasattr(graphs[0], 'session_id'):
        sessions = list(set(g.session_id for g in graphs))
        if sessions:
            target_session = sessions[0]
            session_graphs = [g for g in graphs if g.session_id == target_session]
            print(f"🔍 Filter by session '{target_session[:20]}...': {len(session_graphs)} graphs")

def create_custom_sequences_examples(dataset):
    """
    Demonstrate creating custom sequences for different use cases.
    
    Args:
        dataset: Loaded OpenSimGraphTimeSeriesDataset instance
    """
    print(f"\n🔄 Custom Sequence Creation Examples")
    print("-" * 40)
    
    if not dataset or not dataset.frame_graphs:
        print("❌ No dataset available")
        return
    
    # Example 1: Short sequences for real-time applications
    short_sequences = dataset.create_custom_sequences(
        sequence_length=5,
        stride=5  # No overlap
    )
    print(f"⚡ Short sequences (len=5, no overlap): {len(short_sequences)} sequences")
    
    # Example 2: Medium sequences with high overlap for training
    medium_sequences = dataset.create_custom_sequences(
        sequence_length=10,
        overlap=8  # 80% overlap
    )
    print(f"🔄 Medium sequences (len=10, 80% overlap): {len(medium_sequences)} sequences")
    
    # Example 3: Long sequences for complex pattern recognition
    long_sequences = dataset.create_custom_sequences(
        sequence_length=20,
        overlap=5  # Minimal overlap
    )
    print(f"📈 Long sequences (len=20, minimal overlap): {len(long_sequences)} sequences")
    
    # Example 4: Downsampled sequences for efficiency
    strided_sequences = dataset.create_custom_sequences(
        sequence_length=8,
        stride=3  # Skip frames
    )
    print(f"⬇️ Strided sequences (len=8, stride=3): {len(strided_sequences)} sequences")
    
    return {
        'short': short_sequences,
        'medium': medium_sequences,
        'long': long_sequences,
        'strided': strided_sequences
    }

def setup_training_dataloaders(dataset, sequence_configs):
    """
    Demonstrate setting up DataLoaders for machine learning training.
    
    Args:
        dataset: Loaded OpenSimGraphTimeSeriesDataset instance
        sequence_configs: Dictionary of sequence configurations
    """
    print(f"\n🤖 Training DataLoader Setup")
    print("-" * 40)
    
    if not dataset or not sequence_configs:
        print("❌ No dataset or sequences available")
        return
    
    dataloaders = {}
    
    # Training DataLoader with shuffling
    if 'medium' in sequence_configs:
        train_loader = dataset.get_dataloader(
            sequences=sequence_configs['medium'],
            batch_size=32,
            shuffle=True
        )
        dataloaders['train'] = train_loader
        print(f"🏋️ Training DataLoader: {len(train_loader)} batches (batch_size=32, shuffle=True)")
        
        # Show batch information
        try:
            sample_batch = next(iter(train_loader))
            print(f"   📊 Sample batch shape: {sample_batch.x.shape}")
            print(f"   🔗 Edge index shape: {sample_batch.edge_index.shape}")
            if hasattr(sample_batch, 'sequence_id'):
                unique_sequences = len(set(sample_batch.sequence_id.tolist()))
                print(f"   🔢 Sequences in batch: {unique_sequences}")
        except:
            print("   ⚠️ Could not load sample batch")
    
    # Validation DataLoader without shuffling
    if 'long' in sequence_configs:
        val_loader = dataset.get_dataloader(
            sequences=sequence_configs['long'],
            batch_size=16,
            shuffle=False
        )
        dataloaders['val'] = val_loader
        print(f"✅ Validation DataLoader: {len(val_loader)} batches (batch_size=16, shuffle=False)")
    
    # Test DataLoader for inference
    if 'short' in sequence_configs:
        test_loader = dataset.get_dataloader(
            sequences=sequence_configs['short'],
            batch_size=64,
            shuffle=False
        )
        dataloaders['test'] = test_loader
        print(f"🧪 Test DataLoader: {len(test_loader)} batches (batch_size=64, shuffle=False)")
    
    return dataloaders

def demonstrate_usage_patterns():
    """
    Show common usage patterns for the unified dataset.
    """
    print(f"\n💡 Common Usage Patterns")
    print("-" * 40)
    
    print(f"""
🎯 **Pattern 1: Motion Type Classification**
```python
# Filter by motion type
jump_graphs = [g for g in dataset.frame_graphs if 'jump' in g.motion_type]
run_graphs = [g for g in dataset.frame_graphs if g.motion_type == 'running']

# Create motion-specific sequences
jump_sequences = dataset.create_custom_sequences(jump_graphs, sequence_length=8)
train_loader = dataset.get_dataloader(jump_sequences, batch_size=32, shuffle=True)
```

🎯 **Pattern 2: Subject-Specific Analysis**
```python
# Analyze individual subjects
for subject_id in metadata['dataset_summary']['subject_ids']:
    subject_graphs = [g for g in dataset.frame_graphs if g.subject_id == subject_id]
    print(f"Subject {subject_id}: {len(subject_graphs)} graphs")
```

🎯 **Pattern 3: Cross-Subject Training**
```python
# Leave-one-subject-out validation
subjects = list(set(g.subject_id for g in dataset.frame_graphs))
for test_subject in subjects:
    train_graphs = [g for g in dataset.frame_graphs if g.subject_id != test_subject]
    test_graphs = [g for g in dataset.frame_graphs if g.subject_id == test_subject]
    # Create sequences and train model...
```

🎯 **Pattern 4: Temporal Analysis**
```python
# Analyze motion progression
for motion_name in unique_motion_names:
    motion_graphs = [g for g in dataset.frame_graphs if g.motion_name == motion_name]
    motion_graphs.sort(key=lambda x: x.time)  # Sort by time
    # Analyze temporal patterns...
```
""")

def main():
    """
    Main function demonstrating how to use the unified OpenCap dataset.
    """
    print("🚀 OpenCap Unified Dataset Usage Example")
    print("=" * 50)
    
    try:
        # Load the unified dataset
        dataset = load_unified_dataset()
        if not dataset:
            return
        
        # Explore metadata
        metadata = explore_dataset_metadata()
        
        # Analyze the graph dataset
        analyze_dataset_graphs(dataset)
        
        # Demonstrate filtering
        filter_dataset_examples(dataset)
        
        # Create custom sequences
        sequence_configs = create_custom_sequences_examples(dataset)
        
        # Setup training DataLoaders
        dataloaders = setup_training_dataloaders(dataset, sequence_configs)
        
        # Show usage patterns
        demonstrate_usage_patterns()
        
        print(f"\n🎉 Example completed successfully!")
        print(f"""
🎯 **Next Steps:**
1. Use the DataLoaders for training your graph neural networks
2. Experiment with different sequence configurations
3. Try filtering by subject, motion type, or time
4. Build models for motion classification, prediction, or analysis
5. Explore the comprehensive metadata for deeper insights

📚 **Resources:**
- Check the scripts/README.md for detailed documentation
- See the notebook examples for advanced usage
- Refer to GraphMechanics documentation for API details
""")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
