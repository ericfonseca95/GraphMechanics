#!/usr/bin/env python3
"""
Test script to verify GraphMechanics package imports and functionality.

This script tests all the new OpenSim graph dataset functionality added to the package.
"""

import sys
from pathlib import Path

# Add project root to path for testing
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all package imports work correctly."""
    print("🧪 Testing GraphMechanics Package Imports...")
    
    try:
        # Test core package import
        import graphmechanics
        print(f"✅ Package version: {graphmechanics.__version__}")
        
        # Test individual imports
        from graphmechanics import (
            TRCParser, 
            OpenSimParser, 
            OpenSimModelParser, 
            OpenSimMotionParser,
            MotionGraphDataset,
            KinematicGraphBuilder,
            OpenSimTimeSeriesGraphBuilder,
            OpenSimGraphTimeSeriesDataset,
            create_opensim_graph_dataset
        )
        
        print("✅ All core classes imported successfully!")
        
        # Test data module imports
        from graphmechanics.data import (
            MotionGraphDataset,
            KinematicGraphBuilder, 
            OpenSimTimeSeriesGraphBuilder,
            OpenSimGraphTimeSeriesDataset,
            create_opensim_graph_dataset
        )
        
        print("✅ All data module classes imported successfully!")
        
        # Test utils module imports
        from graphmechanics.utils import (
            TRCParser,
            OpenSimParser,
            OpenSimModelParser,
            OpenSimMotionParser
        )
        
        print("✅ All utils module classes imported successfully!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_class_instantiation():
    """Test that classes can be instantiated properly."""
    print("\n🏗️  Testing Class Instantiation...")
    
    try:
        from graphmechanics import OpenSimGraphTimeSeriesDataset
        
        # Test basic instantiation
        dataset = OpenSimGraphTimeSeriesDataset(output_dir="test_output")
        print(f"✅ OpenSimGraphTimeSeriesDataset instantiated: {type(dataset)}")
        
        # Test convenience function import
        from graphmechanics import create_opensim_graph_dataset
        print(f"✅ Convenience function imported: {create_opensim_graph_dataset}")
        
        return True
        
    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        return False

def test_package_structure():
    """Test package structure and __all__ attributes."""
    print("\n📦 Testing Package Structure...")
    
    try:
        import graphmechanics
        import graphmechanics.data
        import graphmechanics.utils
        
        print(f"✅ Main package __all__: {len(graphmechanics.__all__)} items")
        for item in graphmechanics.__all__:
            print(f"   - {item}")
            
        print(f"✅ Data module __all__: {len(graphmechanics.data.__all__)} items")
        for item in graphmechanics.data.__all__:
            print(f"   - {item}")
            
        print(f"✅ Utils module __all__: {len(graphmechanics.utils.__all__)} items")
        for item in graphmechanics.utils.__all__:
            print(f"   - {item}")
        
        return True
        
    except Exception as e:
        print(f"❌ Package structure error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 GraphMechanics Package Integration Test")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Run tests
    if test_imports():
        success_count += 1
        
    if test_class_instantiation():
        success_count += 1
        
    if test_package_structure():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! GraphMechanics package is ready to use.")
        print("\n💡 Usage Example:")
        print("   from graphmechanics import OpenSimGraphTimeSeriesDataset")
        print("   dataset = OpenSimGraphTimeSeriesDataset('model.osim', 'motion.mot')")
        print("   dataset.create_frame_graphs(add_derivatives=True)")
        print("   sequences = dataset.create_custom_sequences(sequence_length=10)")
        print("   dataset.export_numpy()")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
