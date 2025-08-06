#!/usr/bin/env python3
"""
GraphMechanics OpenSim Graph Dataset - Complete Usage Example

This script demonstrates how to use the comprehensive OpenSimGraphTimeSeriesDataset class
that has been integrated into the GraphMechanics package.

Features demonstrated:
- Loading OpenSim data
- Creating time-series graphs with derivatives
- Flexible sequence generation
- Multiple export formats
- Data reloading and re-sequencing
- Comprehensive analysis and visualization
"""

import sys
from pathlib import Path
import numpy as np

# Import from the GraphMechanics package
from graphmechanics import (
    OpenSimGraphTimeSeriesDataset,
    create_opensim_graph_dataset,
    OpenSimModelParser,
    OpenSimMotionParser
)

def find_test_data():
    """Find available test data files."""
    project_root = Path.cwd()
    
    # Look for OpenSim files
    osim_files = list(project_root.rglob("*.osim"))
    mot_files = list(project_root.rglob("*.mot"))
    
    print(f"🔍 Found {len(osim_files)} .osim files and {len(mot_files)} .mot files")
    
    if osim_files and mot_files:
        return str(osim_files[0]), str(mot_files[0])
    else:
        print("⚠️ No test data found. This example requires .osim and .mot files.")
        return None, None

def example_basic_usage(model_path, motion_path):
    """Demonstrate basic usage of OpenSimGraphTimeSeriesDataset."""
    print("\n🚀 Example 1: Basic Usage")
    print("-" * 40)
    
    # Create dataset using the convenience function
    dataset = create_opensim_graph_dataset(
        model_path=model_path,
        motion_path=motion_path,
        output_dir="example_dataset",
        sequence_length=8,
        overlap=4,
        add_derivatives=True
    )
    
    print(f"📊 Dataset created: {len(dataset.frame_graphs)} frame graphs")
    print(f"📊 Sequences created: {len(dataset.sequences)} sequences")
    
    # Analyze the dataset
    dataset.analyze_sequences(dataset.sequences, "Basic Sequences")
    
    return dataset

def example_advanced_sequencing(dataset):
    """Demonstrate advanced sequencing capabilities."""
    print("\n🔧 Example 2: Advanced Sequencing")
    print("-" * 40)
    
    # Create different sequence configurations
    configs = [
        {"name": "Short", "params": {"sequence_length": 5, "stride": 5}},
        {"name": "Medium", "params": {"sequence_length": 10, "overlap": 7}},
        {"name": "Long", "params": {"sequence_length": 15, "overlap": 3}},
        {"name": "Strided", "params": {"sequence_length": 8, "stride": 3}}
    ]
    
    sequence_results = {}
    
    for config in configs:
        sequences = dataset.create_custom_sequences(**config["params"])
        sequence_results[config["name"]] = sequences
        dataset.analyze_sequences(sequences, f"{config['name']} Sequences")
        
        # Save configuration
        dataset.save_sequences_config(sequences, f"{config['name'].lower()}_config")
    
    return sequence_results

def example_export_import(dataset):
    """Demonstrate export and import functionality."""
    print("\n💾 Example 3: Export and Import")
    print("-" * 40)
    
    # Export in multiple formats
    numpy_path = dataset.export_numpy("example_graphs.npz")
    pytorch_path = dataset.export_pytorch_geometric("example_graphs.pt")
    
    print(f"✅ Exported to NumPy: {numpy_path}")
    print(f"✅ Exported to PyTorch: {pytorch_path}")
    
    # Test reloading from NumPy
    reloaded_dataset = OpenSimGraphTimeSeriesDataset.load_from_numpy(
        numpy_path,
        output_dir="reloaded_dataset"
    )
    
    print(f"📂 Reloaded dataset: {len(reloaded_dataset.frame_graphs)} graphs")
    
    # Create new sequences from reloaded data
    new_sequences = reloaded_dataset.create_custom_sequences(
        sequence_length=12,
        overlap=6,
        stride=4
    )
    
    print(f"🔧 Created {len(new_sequences)} new sequences from reloaded data")
    
    return reloaded_dataset

def example_dataloader_integration(dataset, sequence_results):
    """Demonstrate DataLoader integration for ML training."""
    print("\n🤖 Example 4: DataLoader Integration")
    print("-" * 40)
    
    # Create DataLoaders for different configurations
    dataloaders = {}
    
    for config_name, sequences in sequence_results.items():
        if sequences:  # Only create if sequences exist
            loader = dataset.get_dataloader(
                sequences=sequences,
                batch_size=16 if config_name in ["Short", "Medium"] else 8,
                shuffle=True if config_name in ["Short", "Medium"] else False
            )
            dataloaders[config_name] = loader
            
            print(f"✅ {config_name} DataLoader: {len(loader)} batches")
            
            # Show first batch info
            first_batch = next(iter(loader))
            print(f"   📊 Batch shape: {first_batch.x.shape}")
            print(f"   🔗 Edge index shape: {first_batch.edge_index.shape}")
            if hasattr(first_batch, 'sequence_id'):
                print(f"   🔢 Unique sequences in batch: {len(set(first_batch.sequence_id.tolist()))}")
    
    return dataloaders

def example_analysis_visualization(dataset):
    """Demonstrate analysis and visualization capabilities."""
    print("\n📊 Example 5: Analysis and Visualization")
    print("-" * 40)
    
    # Print dataset summary
    print(f"📈 Dataset Summary:")
    print(f"   Total frames: {len(dataset.frame_graphs)}")
    if dataset.metadata:
        info = dataset.metadata.get('dataset_info', {})
        print(f"   Duration: {info.get('time_span', {}).get('duration', 0):.3f}s")
        print(f"   Nodes per graph: {info.get('num_nodes', 0)}")
        print(f"   Edges per graph: {info.get('num_edges', 0)}")
        print(f"   Node features: {info.get('node_features', 0)}")
        print(f"   Edge features: {info.get('edge_features', 0)}")
    
    # Create visualizations (this will display plots)
    try:
        print("🎨 Creating comprehensive visualizations...")
        dataset.visualize_graph_structure(graph_idx=0)
        print("✅ Visualizations created successfully!")
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        print("   (This may happen in headless environments)")

def main():
    """Run the complete example."""
    print("🎯 GraphMechanics OpenSim Graph Dataset - Complete Example")
    print("=" * 60)
    
    # Find test data
    model_path, motion_path = find_test_data()
    
    if not model_path or not motion_path:
        print("❌ Cannot run example without test data.")
        print("💡 Please ensure you have .osim and .mot files in your project directory.")
        return False
    
    print(f"📄 Using model: {Path(model_path).name}")
    print(f"📈 Using motion: {Path(motion_path).name}")
    
    try:
        # Run examples
        dataset = example_basic_usage(model_path, motion_path)
        sequence_results = example_advanced_sequencing(dataset)
        reloaded_dataset = example_export_import(dataset)
        dataloaders = example_dataloader_integration(dataset, sequence_results)
        example_analysis_visualization(dataset)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("\n💡 Key takeaways:")
        print("   ✅ OpenSimGraphTimeSeriesDataset provides a unified interface")
        print("   ✅ Flexible sequencing with overlap, stride, and window parameters")
        print("   ✅ Multiple export formats (NumPy, PyTorch Geometric, JSON)")
        print("   ✅ Easy reloading and re-sequencing capabilities")
        print("   ✅ Direct integration with PyTorch DataLoaders")
        print("   ✅ Comprehensive analysis and visualization tools")
        
        print(f"\n📁 Generated files:")
        print(f"   - example_dataset/ (main dataset)")
        print(f"   - reloaded_dataset/ (reloaded dataset)")
        print(f"   - Multiple .json configuration files")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
