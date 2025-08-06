#!/usr/bin/env python3
"""
OpenCap Data Batch Processor

This script processes all OpenCap data folders and converts them into a single comprehensive
graph dataset using the GraphMechanics package. Each OpenCap session contains multiple
motion trials with the same model, and this script combines them all into one dataset.

Features:
- Automatically discovers all OpenCap data folders
- Processes each session with its own model and multiple motion files
- Combines all sessions into a single comprehensive dataset
- Adds metadata for session identification and motion type
- Supports flexible sequence generation
- Exports in multiple formats
- Provides comprehensive progress reporting

Usage:
    python process_opencap_data.py [--output-dir OUTPUT_DIR] [--sequence-length LENGTH] [--overlap OVERLAP]
"""

import sys
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import GraphMechanics components
try:
    from graphmechanics import (
        OpenSimGraphTimeSeriesDataset,
        OpenSimModelParser,
        OpenSimMotionParser
    )
    print("✅ GraphMechanics package imported successfully")
except ImportError as e:
    print(f"❌ Error importing GraphMechanics: {e}")
    print("Please ensure the GraphMechanics package is properly installed")
    sys.exit(1)

class OpenCapDataProcessor:
    """
    Processes all OpenCap data folders and creates a unified graph dataset.
    """
    
    def __init__(self, data_dir: str = "Data", output_dir: str = "opencap_unified_dataset"):
        """
        Initialize the OpenCap data processor.
        
        Args:
            data_dir: Directory containing OpenCap data folders
            output_dir: Output directory for the unified dataset
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Storage for all data
        self.sessions = []
        self.all_frame_graphs = []
        self.session_metadata = []
        self.motion_metadata = []
        
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def discover_opencap_sessions(self) -> List[Dict]:
        """
        Discover all OpenCap session folders and their contents.
        
        Returns:
            List of session dictionaries with paths and metadata
        """
        print("\n🔍 Discovering OpenCap sessions...")
        
        sessions = []
        
        # Look for OpenCapData folders
        opencap_folders = [f for f in self.data_dir.iterdir() 
                          if f.is_dir() and f.name.startswith('OpenCapData_')]
        
        print(f"Found {len(opencap_folders)} OpenCap session folders")
        
        for folder in opencap_folders:
            print(f"\n📂 Processing folder: {folder.name}")
            
            # Look for session metadata
            metadata_file = folder / "sessionMetadata.yaml"
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = yaml.safe_load(f)
                    print(f"   ✅ Loaded session metadata")
                except Exception as e:
                    print(f"   ⚠️ Error loading metadata: {e}")
            
            # Look for OpenSim model
            model_dir = folder / "OpenSimData" / "Model"
            osim_files = list(model_dir.glob("*.osim")) if model_dir.exists() else []
            
            # Look for motion files
            kinematics_dir = folder / "OpenSimData" / "Kinematics"
            mot_files = list(kinematics_dir.glob("*.mot")) if kinematics_dir.exists() else []
            
            if osim_files and mot_files:
                session_info = {
                    'session_id': folder.name,
                    'folder_path': folder,
                    'metadata': metadata,
                    'model_file': osim_files[0],  # Use first .osim file
                    'motion_files': mot_files,
                    'subject_id': metadata.get('subjectID', 'unknown'),
                    'height_m': metadata.get('height_m', 0.0),
                    'mass_kg': metadata.get('mass_kg', 0.0),
                    'gender': metadata.get('gender_mf', 'unknown')
                }
                sessions.append(session_info)
                
                print(f"   ✅ Model file: {osim_files[0].name}")
                print(f"   ✅ Motion files: {len(mot_files)}")
                print(f"   ✅ Subject: {session_info['subject_id']}")
                
                # Show motion file names
                for mot_file in mot_files:
                    print(f"      - {mot_file.name}")
            else:
                print(f"   ❌ Missing required files (model: {len(osim_files)}, motion: {len(mot_files)})")
        
        print(f"\n📊 Total valid sessions found: {len(sessions)}")
        return sessions
    
    def process_session(self, session_info: Dict, add_derivatives: bool = True) -> List[Dict]:
        """
        Process a single OpenCap session with multiple motion files.
        
        Args:
            session_info: Session information dictionary
            add_derivatives: Whether to add velocity and acceleration features
            
        Returns:
            List of motion data dictionaries
        """
        print(f"\n🔄 Processing session: {session_info['session_id']}")
        print(f"   👤 Subject: {session_info['subject_id']}")
        print(f"   📏 Height: {session_info['height_m']}m, Mass: {session_info['mass_kg']}kg")
        
        # Load model once for the session
        try:
            model_parser = OpenSimModelParser(str(session_info['model_file']))
            print(f"   ✅ Loaded model: {model_parser.model_name}")
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            return []
        
        motion_data_list = []
        
        # Process each motion file
        for i, motion_file in enumerate(session_info['motion_files']):
            motion_name = motion_file.stem
            print(f"\n   🏃 Processing motion {i+1}/{len(session_info['motion_files'])}: {motion_name}")
            
            try:
                # Load motion data
                motion_parser = OpenSimMotionParser(str(motion_file))
                print(f"      📊 Frames: {len(motion_parser.data)}")
                print(f"      📐 Coordinates: {len(motion_parser.coordinate_names)}")
                
                if 'time' in motion_parser.data.columns:
                    duration = motion_parser.data['time'].max() - motion_parser.data['time'].min()
                    print(f"      ⏰ Duration: {duration:.3f}s")
                
                # Create dataset for this motion
                dataset = OpenSimGraphTimeSeriesDataset(
                    output_dir=str(self.output_dir / "temp"),
                    model_path=str(session_info['model_file']),
                    motion_path=str(motion_file)
                )
                
                # Create frame graphs
                frame_graphs = dataset.create_frame_graphs(add_derivatives=add_derivatives)
                print(f"      ✅ Created {len(frame_graphs)} frame graphs")
                
                # Add session and motion metadata to each graph
                for graph in frame_graphs:
                    graph.session_id = session_info['session_id']
                    graph.subject_id = session_info['subject_id']
                    graph.motion_name = motion_name
                    graph.motion_type = self._classify_motion_type(motion_name)
                    graph.session_metadata = session_info['metadata']
                
                # Store motion data
                motion_data = {
                    'session_id': session_info['session_id'],
                    'subject_id': session_info['subject_id'],
                    'motion_name': motion_name,
                    'motion_type': self._classify_motion_type(motion_name),
                    'motion_file': str(motion_file),
                    'frame_graphs': frame_graphs,
                    'num_frames': len(frame_graphs),
                    'duration': duration if 'time' in motion_parser.data.columns else 0.0,
                    'coordinates': motion_parser.coordinate_names
                }
                
                motion_data_list.append(motion_data)
                self.all_frame_graphs.extend(frame_graphs)
                
            except Exception as e:
                print(f"      ❌ Error processing motion: {e}")
                continue
        
        print(f"   ✅ Session processed: {len(motion_data_list)} motions successfully")
        return motion_data_list
    
    def _classify_motion_type(self, motion_name: str) -> str:
        """
        Classify motion type based on filename.
        
        Args:
            motion_name: Motion filename (without extension)
            
        Returns:
            Motion type classification
        """
        motion_name_lower = motion_name.lower()
        
        if 'jump' in motion_name_lower:
            if 'vertical' in motion_name_lower:
                return 'vertical_jump'
            elif 'drop' in motion_name_lower:
                return 'drop_landing'
            else:
                return 'jump'
        elif 'run' in motion_name_lower:
            return 'running'
        elif 'walk' in motion_name_lower:
            return 'walking'
        elif 'squat' in motion_name_lower:
            return 'squatting'
        elif 'cut' in motion_name_lower:
            return 'cutting'
        elif 'balance' in motion_name_lower:
            return 'balance'
        elif 'alignment' in motion_name_lower:
            return 'alignment'
        else:
            return 'other'
    
    def process_all_sessions(self, add_derivatives: bool = True) -> None:
        """
        Process all discovered OpenCap sessions.
        
        Args:
            add_derivatives: Whether to add velocity and acceleration features
        """
        print("\n🚀 Processing All OpenCap Sessions")
        print("=" * 50)
        
        # Discover sessions
        sessions = self.discover_opencap_sessions()
        
        if not sessions:
            print("❌ No valid OpenCap sessions found!")
            return
        
        # Process each session
        for i, session_info in enumerate(sessions, 1):
            print(f"\n📋 Session {i}/{len(sessions)}")
            motion_data_list = self.process_session(session_info, add_derivatives)
            
            if motion_data_list:
                self.sessions.append({
                    'session_info': session_info,
                    'motion_data': motion_data_list
                })
                
                # Update metadata
                for motion_data in motion_data_list:
                    self.motion_metadata.append({
                        'session_id': motion_data['session_id'],
                        'subject_id': motion_data['subject_id'],
                        'motion_name': motion_data['motion_name'],
                        'motion_type': motion_data['motion_type'],
                        'num_frames': motion_data['num_frames'],
                        'duration': motion_data['duration']
                    })
        
        # Create comprehensive metadata
        self._create_dataset_metadata()
        
        print(f"\n✅ Processing Complete!")
        print(f"   📊 Total sessions: {len(self.sessions)}")
        print(f"   🎬 Total motions: {len(self.motion_metadata)}")
        print(f"   🔢 Total frame graphs: {len(self.all_frame_graphs)}")
    
    def _create_dataset_metadata(self) -> None:
        """Create comprehensive dataset metadata."""
        # Summary statistics
        total_duration = sum(m['duration'] for m in self.motion_metadata)
        motion_types = [m['motion_type'] for m in self.motion_metadata]
        motion_type_counts = {mt: motion_types.count(mt) for mt in set(motion_types)}
        
        subjects = list(set(m['subject_id'] for m in self.motion_metadata))
        
        self.dataset_metadata = {
            'creation_info': {
                'timestamp': datetime.now().isoformat(),
                'processor': 'OpenCapDataProcessor',
                'version': '1.0.0'
            },
            'dataset_summary': {
                'total_sessions': len(self.sessions),
                'total_motions': len(self.motion_metadata),
                'total_frames': len(self.all_frame_graphs),
                'total_duration_seconds': total_duration,
                'unique_subjects': len(subjects),
                'subject_ids': subjects
            },
            'motion_type_distribution': motion_type_counts,
            'sessions_detail': [
                {
                    'session_id': session['session_info']['session_id'],
                    'subject_id': session['session_info']['subject_id'],
                    'num_motions': len(session['motion_data']),
                    'motion_names': [m['motion_name'] for m in session['motion_data']]
                }
                for session in self.sessions
            ]
        }
    
    def create_sequences(self, 
                        sequence_length: int = 10, 
                        overlap: int = 5, 
                        stride: Optional[int] = None) -> List[Dict]:
        """
        Create sequences from all frame graphs.
        
        Args:
            sequence_length: Length of each sequence
            overlap: Overlap between sequences
            stride: Custom stride (if None, uses sequence_length - overlap)
            
        Returns:
            List of sequence dictionaries
        """
        print(f"\n🔧 Creating sequences (length={sequence_length}, overlap={overlap})...")
        
        if not self.all_frame_graphs:
            print("❌ No frame graphs available!")
            return []
        
        # Create unified dataset
        unified_dataset = OpenSimGraphTimeSeriesDataset(output_dir=str(self.output_dir))
        unified_dataset.frame_graphs = self.all_frame_graphs
        unified_dataset.metadata = self.dataset_metadata
        
        # Create sequences
        if stride is not None:
            sequences = unified_dataset.create_custom_sequences(
                sequence_length=sequence_length,
                overlap=overlap,
                stride=stride
            )
        else:
            sequences = unified_dataset.create_custom_sequences(
                sequence_length=sequence_length,
                overlap=overlap
            )
        
        print(f"✅ Created {len(sequences)} sequences")
        return sequences
    
    def export_dataset(self, sequences: Optional[List[Dict]] = None) -> Dict[str, str]:
        """
        Export the unified dataset in multiple formats.
        
        Args:
            sequences: Optional sequences to export
            
        Returns:
            Dictionary of exported file paths
        """
        print(f"\n💾 Exporting unified dataset...")
        
        if not self.all_frame_graphs:
            print("❌ No data to export!")
            return {}
        
        # Create unified dataset
        unified_dataset = OpenSimGraphTimeSeriesDataset(output_dir=str(self.output_dir))
        unified_dataset.frame_graphs = self.all_frame_graphs
        unified_dataset.metadata = self.dataset_metadata
        
        exported_files = {}
        
        try:
            # Export frame graphs
            numpy_path = unified_dataset.export_numpy("opencap_unified_graphs.npz")
            pytorch_path = unified_dataset.export_pytorch_geometric("opencap_unified_graphs.pt")
            
            exported_files['numpy'] = numpy_path
            exported_files['pytorch'] = pytorch_path
            
            # Export metadata
            metadata_path = self.output_dir / "opencap_dataset_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.dataset_metadata, f, indent=2)
            exported_files['metadata'] = str(metadata_path)
            
            # Export motion metadata
            motion_metadata_path = self.output_dir / "motion_metadata.json"
            with open(motion_metadata_path, 'w') as f:
                json.dump(self.motion_metadata, f, indent=2)
            exported_files['motion_metadata'] = str(motion_metadata_path)
            
            # Export sequences if provided
            if sequences:
                sequences_path = unified_dataset.save_sequences_config(sequences, "opencap_sequences")
                exported_files['sequences'] = sequences_path
            
            print(f"✅ Export complete!")
            for format_name, file_path in exported_files.items():
                print(f"   📄 {format_name}: {Path(file_path).name}")
            
        except Exception as e:
            print(f"❌ Export error: {e}")
        
        return exported_files
    
    def generate_report(self) -> None:
        """Generate a comprehensive dataset report."""
        print(f"\n📊 OpenCap Unified Dataset Report")
        print("=" * 50)
        
        if not self.dataset_metadata:
            print("❌ No dataset metadata available!")
            return
        
        # Dataset summary
        summary = self.dataset_metadata['dataset_summary']
        print(f"\n📈 Dataset Summary:")
        print(f"   🎬 Total sessions: {summary['total_sessions']}")
        print(f"   🏃 Total motions: {summary['total_motions']}")
        print(f"   🔢 Total frame graphs: {summary['total_frames']}")
        print(f"   ⏰ Total duration: {summary['total_duration_seconds']:.1f}s")
        print(f"   👥 Unique subjects: {summary['unique_subjects']}")
        
        # Motion type distribution
        print(f"\n🎯 Motion Type Distribution:")
        motion_dist = self.dataset_metadata['motion_type_distribution']
        for motion_type, count in sorted(motion_dist.items()):
            percentage = (count / summary['total_motions']) * 100
            print(f"   📊 {motion_type}: {count} motions ({percentage:.1f}%)")
        
        # Session details
        print(f"\n📂 Session Details:")
        for session_detail in self.dataset_metadata['sessions_detail']:
            print(f"   🗂️  {session_detail['session_id']}:")
            print(f"      👤 Subject: {session_detail['subject_id']}")
            print(f"      🎬 Motions: {session_detail['num_motions']}")
            print(f"      📋 Types: {', '.join(session_detail['motion_names'])}")
        
        # Save report to file
        report_path = self.output_dir / "dataset_report.txt"
        with open(report_path, 'w') as f:
            f.write("OpenCap Unified Dataset Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("Dataset Summary:\n")
            f.write(f"  Total sessions: {summary['total_sessions']}\n")
            f.write(f"  Total motions: {summary['total_motions']}\n")
            f.write(f"  Total frame graphs: {summary['total_frames']}\n")
            f.write(f"  Total duration: {summary['total_duration_seconds']:.1f}s\n")
            f.write(f"  Unique subjects: {summary['unique_subjects']}\n\n")
            
            f.write("Motion Type Distribution:\n")
            for motion_type, count in sorted(motion_dist.items()):
                percentage = (count / summary['total_motions']) * 100
                f.write(f"  {motion_type}: {count} motions ({percentage:.1f}%)\n")
        
        print(f"\n✅ Report saved to: {report_path.name}")


def main():
    """Main function to run the OpenCap data processor."""
    parser = argparse.ArgumentParser(
        description="Process all OpenCap data folders into a unified graph dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_opencap_data.py
  python process_opencap_data.py --output-dir my_dataset --sequence-length 8 --overlap 4
  python process_opencap_data.py --data-dir /path/to/opencap/data --no-derivatives
        """
    )
    
    parser.add_argument('--data-dir', default='Data', 
                       help='Directory containing OpenCap data folders (default: Data)')
    parser.add_argument('--output-dir', default='opencap_unified_dataset',
                       help='Output directory for unified dataset (default: opencap_unified_dataset)')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='Length of sequences to create (default: 10)')
    parser.add_argument('--overlap', type=int, default=5,
                       help='Overlap between sequences (default: 5)')
    parser.add_argument('--stride', type=int, default=None,
                       help='Custom stride for sequences (default: sequence_length - overlap)')
    parser.add_argument('--no-derivatives', action='store_true',
                       help='Skip adding velocity and acceleration features')
    parser.add_argument('--no-sequences', action='store_true',
                       help='Skip sequence creation')
    parser.add_argument('--no-export', action='store_true',
                       help='Skip dataset export')
    
    args = parser.parse_args()
    
    print("🚀 OpenCap Data Batch Processor")
    print("=" * 40)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔧 Sequence length: {args.sequence_length}")
    print(f"🔧 Overlap: {args.overlap}")
    print(f"🔧 Add derivatives: {not args.no_derivatives}")
    
    try:
        # Initialize processor
        processor = OpenCapDataProcessor(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        
        # Process all sessions
        processor.process_all_sessions(add_derivatives=not args.no_derivatives)
        
        # Create sequences if requested
        sequences = None
        if not args.no_sequences:
            sequences = processor.create_sequences(
                sequence_length=args.sequence_length,
                overlap=args.overlap,
                stride=args.stride
            )
        
        # Export dataset if requested
        if not args.no_export:
            exported_files = processor.export_dataset(sequences)
        
        # Generate report
        processor.generate_report()
        
        print("\n🎉 Processing completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
