# GraphMechanics Notebooks

This directory contains demonstration and analysis notebooks for the GraphMechanics package.

## 📚 Available Notebooks

### `comprehensive_graph_builders_analysis.ipynb` ⭐ **MAIN DEMO**
**Clean package demonstration notebook**
- **Purpose**: Showcase all GraphMechanics package functionality
- **Style**: Clean, focused, imports from package only
- **Content**:
  - Package import and setup
  - Biomechanical constraints demonstration
  - Graph construction with KinematicGraphBuilder
  - Motion validation with GraphMechanicsValidator
  - Physics-informed loss functions
  - Performance analysis and benchmarking
  - Final validation and summary

### `comprehensive_graph_builders_analysis_backup.ipynb`
**Legacy research notebook (BACKUP)**
- Contains the original research implementation with embedded classes
- Use for reference only - classes have been moved to the package
- ⚠️ **Do not use for new work** - this is kept for historical reference

### `graphmechanics_package_demo.ipynb`
**Identical to main demo** - clean package showcase

## 🚀 Quick Start

1. **Open the main demo notebook:**
   ```bash
   jupyter notebook comprehensive_graph_builders_analysis.ipynb
   ```

2. **Run all cells** to see the complete GraphMechanics pipeline in action

3. **Expected outputs:**
   - ✅ Package component verification
   - 📊 Synthetic motion data generation and visualization
   - 🕸️ Graph construction with anatomical constraints
   - 🔍 Motion validation results
   - ⚡ Physics-informed loss function calculations
   - 📈 Performance benchmarking across all components
   - 🏥 Clinical validation scoring

## 📋 Package Components Demonstrated

| Component | Purpose | Notebook Section |
|-----------|---------|------------------|
| `BiomechanicalConstraints` | Anatomical limit enforcement | Section 2 |
| `KinematicGraphBuilder` | Graph construction from motion | Section 3 |
| `GraphMechanicsValidator` | Motion sequence validation | Section 4 |
| `BiomechanicalLossFunctions` | Physics-informed losses | Section 5 |
| `GraphMechanicsPerformanceAnalyzer` | Benchmarking tools | Section 6 |
| `BiomechanicalValidator` | Final validation system | Section 7 |

## 🔧 Requirements

- GraphMechanics package installed: `pip install -e .`
- PyTorch with PyTorch Geometric
- NumPy, Matplotlib, SciPy
- Jupyter Notebook or JupyterLab

## 📊 Expected Performance

- **Graph Construction**: ~0.01-0.02s per frame
- **Motion Validation**: ~0.1-0.2s per sequence
- **Loss Calculation**: ~0.001-0.002s per batch
- **Overall Validity**: 80-90% for typical motion data

## 🎯 Design Philosophy

The main demo notebook follows these principles:

1. **Package-First**: All functionality imported from `graphmechanics` package
2. **Clean & Focused**: No embedded class definitions
3. **Comprehensive**: Covers all major package components
4. **Production-Ready**: Demonstrates real-world usage patterns
5. **Well-Documented**: Clear explanations and visualizations

## 🛠️ Development

For package development:
1. **Add new features** to the appropriate module in `graphmechanics/`
2. **Update `__init__.py`** to export new classes
3. **Test in notebook** by importing from package
4. **Keep notebooks clean** - no embedded implementations

---

**Note**: If you need to run research experiments with custom implementations, create a new notebook but keep the main demo clean and focused on showcasing the package capabilities.
