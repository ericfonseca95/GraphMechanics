"""
TRC (Track Row Column) File Parser

This module provides a TRCParser class to parse motion capture data from .trc files.
The parser extracts header metadata as class attributes and converts the coordinate 
data into a pandas DataFrame.

The TRCParser is designed to handle real-world TRC files with robust error handling
and flexible data export capabilities for downstream biomechanical analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os


class TRCParser:
    """
    A parser for .trc (Track Row Column) motion capture files.
    
    This class reads .trc files and extracts:
    - Header metadata as class attributes
    - Coordinate data as a pandas DataFrame with proper column names
    
    Attributes:
        file_path (str): Path to the .trc file
        path_file_type (str): File format information
        data_rate (float): Data sampling rate in Hz
        camera_rate (float): Camera frame rate in Hz
        num_frames (int): Number of data frames
        num_markers (int): Number of motion capture markers
        units (str): Units of measurement (typically 'm' for meters)
        orig_data_rate (float): Original data rate before processing
        orig_data_start_frame (int): Original starting frame number
        orig_num_frames (int): Original number of frames
        marker_names (List[str]): List of marker names
        data (pd.DataFrame): DataFrame containing the motion capture data
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the TRC parser and parse the file.
        
        Args:
            file_path (str): Path to the .trc file to parse
        """
        self.file_path = file_path
        self.data = None
        self.marker_names = []
        
        # Initialize header attributes
        self.path_file_type = None
        self.data_rate = None
        self.camera_rate = None
        self.num_frames = None
        self.num_markers = None
        self.units = None
        self.orig_data_rate = None
        self.orig_data_start_frame = None
        self.orig_num_frames = None
        
        # Parse the file
        self._parse_file()
    
    def _parse_file(self) -> None:
        """Parse the .trc file and extract header information and data."""
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
        
        # Parse header information
        self._parse_header(lines)
        
        # Parse marker names and create column structure
        self._parse_marker_names(lines)
        
        # Parse data
        self._parse_data(lines)
    
    def _parse_header(self, lines: List[str]) -> None:
        """
        Parse header information from the first few lines.
        
        Args:
            lines (List[str]): All lines from the file
        """
        # Line 1: PathFileType information
        header_line = lines[0].strip()
        # Extract path file type (everything before the path)
        parts = header_line.split()
        if len(parts) >= 3:
            self.path_file_type = f"{parts[0]} {parts[1]} {parts[2]}"
        
        # Line 2: Header field names (we can store these if needed)
        header_fields = lines[1].strip().split('\t')
        
        # Line 3: Header values
        header_values = lines[2].strip().split('\t')
        
        # Map header values to attributes
        if len(header_values) >= 8:
            self.data_rate = float(header_values[0])
            self.camera_rate = float(header_values[1])
            self.num_frames = int(header_values[2])
            self.num_markers = int(header_values[3])
            self.units = header_values[4]
            self.orig_data_rate = float(header_values[5])
            self.orig_data_start_frame = int(header_values[6])
            self.orig_num_frames = int(header_values[7])
    
    def _parse_marker_names(self, lines: List[str]) -> None:
        """
        Robustly parse marker names from TRC header with comprehensive error handling.
        
        This method handles various TRC formats from different motion capture systems
        including Vicon, Qualisys, OptiTrack, and OpenCap.
        
        Args:
            lines (List[str]): All lines from the file
        """
        # Line 4: Marker names (tab-separated)
        marker_line = lines[3].strip().split('\t')
        
        # Remove Frame# and Time columns and any empty entries
        raw_markers = [marker.strip() for marker in marker_line[2:] if marker.strip()]
        
        # Handle different TRC formats:
        # 1. Standard format: one marker name per column group
        # 2. Some systems repeat marker names 3 times (X, Y, Z)
        # 3. Some systems have trailing empty columns
        
        # Detect if marker names are repeated (common in some systems)
        if len(raw_markers) >= 3 and raw_markers[0] == raw_markers[1] == raw_markers[2]:
            # Markers are repeated 3 times, take every 3rd
            self.marker_names = raw_markers[::3]
        else:
            # Standard format or unique names
            self.marker_names = raw_markers
        
        # Validate against header information
        if hasattr(self, 'num_markers') and self.num_markers:
            if len(self.marker_names) != self.num_markers:
                print(f"Info: Header says {self.num_markers} markers, found {len(self.marker_names)} names")
                # Trust the actual data structure over header
                self.num_markers = len(self.marker_names)
        else:
            self.num_markers = len(self.marker_names)
        
        # Line 5: Coordinate labels - validate structure
        coord_labels = lines[4].strip().split('\t')
        coord_labels = [label.strip() for label in coord_labels if label.strip()]
        
        # Skip Frame# and Time columns
        actual_coord_labels = coord_labels[2:] if len(coord_labels) > 2 else coord_labels
        
        # Validate coordinate structure
        expected_coords = self.num_markers * 3  # X, Y, Z per marker
        if len(actual_coord_labels) != expected_coords:
            print(f"Warning: Expected {expected_coords} coordinate columns, found {len(actual_coord_labels)}")
            # Adjust marker count based on actual coordinate columns
            self.num_markers = len(actual_coord_labels) // 3
            self.marker_names = self.marker_names[:self.num_markers]
        
        print(f"Successfully parsed {self.num_markers} markers: {self.marker_names[:5]}{'...' if len(self.marker_names) > 5 else ''}")
    
    def _parse_data(self, lines: List[str]) -> None:
        """
        Robustly parse coordinate data with comprehensive error handling and validation.
        
        This method handles various data quality issues common in motion capture:
        - Missing frames
        - Inconsistent column counts
        - Non-numeric values (NaN, empty cells)
        - Different decimal separators
        
        Args:
            lines (List[str]): All lines from the file
        """
        # Data starts from line 6 (index 5)
        data_lines = lines[5:]
        
        # Parse each line with robust error handling
        data_rows = []
        skipped_lines = 0
        
        for line_num, line in enumerate(data_lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                # Split and clean values
                values = line.split('\t')
                
                # Handle missing values and convert to float
                numeric_values = []
                for val in values:
                    val = val.strip()
                    if val == '' or val.lower() in ['nan', 'null', 'none']:
                        numeric_values.append(np.nan)
                    else:
                        try:
                            # Handle different decimal separators
                            val = val.replace(',', '.')
                            numeric_values.append(float(val))
                        except ValueError:
                            # If conversion fails, use NaN
                            numeric_values.append(np.nan)
                
                # Validate row length
                expected_cols = 2 + (self.num_markers * 3)  # Frame, Time + XYZ per marker
                
                if len(numeric_values) < expected_cols:
                    # Pad with NaN if too short
                    numeric_values.extend([np.nan] * (expected_cols - len(numeric_values)))
                elif len(numeric_values) > expected_cols:
                    # Truncate if too long
                    numeric_values = numeric_values[:expected_cols]
                
                data_rows.append(numeric_values)
                
            except Exception as e:
                skipped_lines += 1
                if skipped_lines <= 5:  # Only show first 5 errors
                    print(f"Warning: Skipped line {line_num + 6}: {str(e)[:50]}...")
                continue
        
        if skipped_lines > 5:
            print(f"... and {skipped_lines - 5} more lines with parsing errors")
        
        # Create DataFrame with robust column naming
        if data_rows:
            # Create column names based on validated structure
            columns = ['Frame', 'Time']
            
            # Add X, Y, Z columns for each marker
            for i, marker_name in enumerate(self.marker_names):
                # Clean marker names for column headers
                clean_name = self._clean_marker_name(marker_name)
                columns.extend([f'{clean_name}_X', f'{clean_name}_Y', f'{clean_name}_Z'])
            
            # Create DataFrame
            self.data = pd.DataFrame(data_rows, columns=columns)
            
            # Data quality checks and cleaning
            self._validate_and_clean_data()
            
        else:
            print("Error: No valid data rows found")
            self.data = pd.DataFrame()
    
    def _clean_marker_name(self, marker_name: str) -> str:
        """
        Clean marker names to be valid Python identifiers and consistent.
        
        Args:
            marker_name: Raw marker name from TRC file
            
        Returns:
            Cleaned marker name
        """
        # Remove problematic characters and standardize
        clean_name = marker_name.replace(' ', '_').replace('.', '_').replace('-', '_')
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
        
        # Ensure it doesn't start with a number
        if clean_name and clean_name[0].isdigit():
            clean_name = 'M_' + clean_name
        
        return clean_name or f'Marker_{id(marker_name)}'
    
    def _validate_and_clean_data(self) -> None:
        """
        Validate and clean the parsed data for biomechanical analysis.
        
        Performs quality checks and applies appropriate cleaning:
        - Frame number validation
        - Time sequence validation  
        - Coordinate range validation
        - Missing data interpolation (conservative)
        """
        if self.data.empty:
            return
        
        # Validate frame numbers
        if 'Frame' in self.data.columns:
            self.data['Frame'] = pd.to_numeric(self.data['Frame'], errors='coerce')
            self.data['Frame'] = self.data['Frame'].fillna(method='ffill').astype(int)
        
        # Validate time sequence
        if 'Time' in self.data.columns:
            self.data['Time'] = pd.to_numeric(self.data['Time'], errors='coerce')
            # Fill missing time values with interpolation
            self.data['Time'] = self.data['Time'].interpolate(method='linear')
        
        # Validate coordinate data ranges (reasonable for human motion in meters)
        coord_columns = [col for col in self.data.columns if col.endswith(('_X', '_Y', '_Z'))]
        
        for col in coord_columns:
            # Convert to numeric, coercing errors to NaN
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Flag obviously erroneous values (outside reasonable human movement range)
            if col.endswith('_Z'):  # Height coordinate
                self.data.loc[self.data[col] < -2.0, col] = np.nan  # Below ground
                self.data.loc[self.data[col] > 3.0, col] = np.nan   # Above reasonable height
            else:  # X, Y coordinates
                self.data.loc[abs(self.data[col]) > 10.0, col] = np.nan  # Beyond reasonable lab space
        
        # Calculate data quality metrics
        total_points = len(coord_columns) * len(self.data) if coord_columns else 0
        missing_points = self.data[coord_columns].isna().sum().sum() if coord_columns else 0
        
        if total_points > 0:
            completeness = 100 * (1 - missing_points / total_points)
            print(f"Data quality: {completeness:.1f}% complete ({missing_points}/{total_points} missing points)")
            
            if completeness < 90:
                print("Warning: Low data completeness. Consider data cleaning or different analysis approach.")
        
        # Store quality metrics
        self.data_quality = {
            'completeness_percent': completeness if total_points > 0 else 0,
            'total_frames': len(self.data),
            'missing_points': missing_points,
            'total_points': total_points
        }
    
    def get_marker_data(self, marker_name: str) -> Optional[pd.DataFrame]:
        """
        Get X, Y, Z coordinate data for a specific marker.
        
        Args:
            marker_name (str): Name of the marker
            
        Returns:
            pd.DataFrame: DataFrame with Time, X, Y, Z columns for the marker,
                         or None if marker not found
        """
        if marker_name not in self.marker_names:
            print(f"Marker '{marker_name}' not found. Available markers: {self.marker_names}")
            return None
        
        columns = ['Time', f'{marker_name}_X', f'{marker_name}_Y', f'{marker_name}_Z']
        marker_data = self.data[columns].copy()
        
        # Rename columns for easier use
        marker_data.columns = ['Time', 'X', 'Y', 'Z']
        
        return marker_data
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the parsed TRC data.
        
        Returns:
            Dict: Summary information about the data
        """
        summary = {
            'file_path': self.file_path,
            'path_file_type': self.path_file_type,
            'data_rate': self.data_rate,
            'camera_rate': self.camera_rate,
            'num_frames': self.num_frames,
            'num_markers': self.num_markers,
            'units': self.units,
            'orig_data_rate': self.orig_data_rate,
            'orig_data_start_frame': self.orig_data_start_frame,
            'orig_num_frames': self.orig_num_frames,
            'marker_names': self.marker_names,
            'data_shape': self.data.shape if self.data is not None else None,
            'duration': self.data['Time'].max() - self.data['Time'].min() if self.data is not None else None
        }
        return summary
    
    def export_marker_to_csv(self, marker_name: str, output_path: str = None) -> str:
        """
        Export a specific marker's data to a CSV file.
        
        Args:
            marker_name (str): Name of the marker to export
            output_path (str, optional): Path for the output CSV file. 
                                       If None, creates a filename based on marker name
        
        Returns:
            str: Path to the exported CSV file
        
        Raises:
            ValueError: If marker not found or data not available
        """
        if self.data is None:
            raise ValueError("No data available to export")
        
        if marker_name not in self.marker_names:
            raise ValueError(f"Marker '{marker_name}' not found. Available markers: {self.marker_names}")
        
        # Get marker data
        marker_data = self.get_marker_data(marker_name)
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_path = f"{base_name}_{marker_name}_data.csv"
        
        # Export to CSV
        marker_data.to_csv(output_path, index=False)
        print(f"Exported {marker_name} data to: {output_path}")
        
        return output_path
    
    def export_all_markers_to_csv(self, output_dir: str = None) -> List[str]:
        """
        Export all markers' data to individual CSV files.
        
        Args:
            output_dir (str, optional): Directory to save CSV files. 
                                      If None, saves in current directory
        
        Returns:
            List[str]: List of paths to exported CSV files
        
        Raises:
            ValueError: If no data available
        """
        if self.data is None:
            raise ValueError("No data available to export")
        
        # Create output directory if provided
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        exported_files = []
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        
        # Export each marker
        for marker_name in self.marker_names:
            if output_dir:
                output_path = os.path.join(output_dir, f"{base_name}_{marker_name}_data.csv")
            else:
                output_path = f"{base_name}_{marker_name}_data.csv"
            
            try:
                exported_path = self.export_marker_to_csv(marker_name, output_path)
                exported_files.append(exported_path)
            except Exception as e:
                print(f"Warning: Failed to export {marker_name}: {e}")
        
        print(f"Exported {len(exported_files)} marker data files")
        return exported_files
    
    def export_full_data_to_csv(self, output_path: str = None) -> str:
        """
        Export the complete motion capture data to a CSV file.
        
        Args:
            output_path (str, optional): Path for the output CSV file.
                                       If None, creates filename based on original file
        
        Returns:
            str: Path to the exported CSV file
        
        Raises:
            ValueError: If no data available
        """
        if self.data is None:
            raise ValueError("No data available to export")
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_path = f"{base_name}_full_data.csv"
        
        # Export full dataset to CSV
        self.data.to_csv(output_path, index=False)
        print(f"Exported full motion capture data to: {output_path}")
        
        return output_path
    
    def detect_marker_set(self) -> Dict[str, any]:
        """
        Intelligently detect the marker set protocol and anatomical coverage.
        
        This is crucial for creating appropriate graph topologies and
        selecting biomechanically meaningful analysis approaches.
        
        Returns:
            Dictionary with marker set information
        """
        if not self.marker_names:
            return {'protocol': 'unknown', 'confidence': 0.0}
        
        # Convert to lowercase for matching
        markers_lower = [name.lower() for name in self.marker_names]
        
        # Define marker set patterns (biomechanics protocols)
        protocols = {
            'helen_hayes': {
                'required': ['lasi', 'rasi', 'lpsi', 'rpsi', 'lkne', 'rkne', 'lank', 'rank'],
                'optional': ['lthi', 'rthi', 'lshn', 'rshn', 'ltoe', 'rtoe', 'lhee', 'rhee'],
                'description': 'Helen Hayes lower body marker set'
            },
            'plug_in_gait': {
                'required': ['lasi', 'rasi', 'lpsi', 'rpsi', 'lthi', 'lkne', 'lshn', 'lank',
                           'lhee', 'ltoe', 'rthi', 'rkne', 'rshn', 'rank', 'rhee', 'rtoe'],
                'optional': ['lfhd', 'rfhd', 'lbhd', 'rbhd', 'c7', 't10', 'clav', 'strn'],
                'description': 'Vicon Plug-in Gait full body'
            },
            'opensim_fullbody': {
                'required': ['pelvis', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle'],
                'optional': ['head', 'torso', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow'],
                'description': 'OpenSim full body marker set'
            },
            'opencap': {
                'required': ['neck', 'midHip', 'lhip', 'rhip', 'lknee', 'rknee', 'lankle', 'rankle'],
                'optional': ['lshoulder', 'rshoulder', 'lelbow', 'relbow', 'lwrist', 'rwrist'],
                'description': 'OpenCap smartphone-based motion capture'
            },
            'custom_clinical': {
                'required': [],  # Will be detected based on anatomical coverage
                'optional': [],
                'description': 'Custom clinical/research marker set'
            }
        }
        
        # Score each protocol
        best_protocol = 'unknown'
        best_score = 0.0
        
        for protocol_name, protocol_info in protocols.items():
            if protocol_name == 'custom_clinical':
                continue  # Handle separately
            
            required_matches = sum(1 for req in protocol_info['required'] 
                                 if any(req in marker.lower() for marker in markers_lower))
            optional_matches = sum(1 for opt in protocol_info['optional']
                                 if any(opt in marker.lower() for marker in markers_lower))
            
            total_required = len(protocol_info['required'])
            if total_required == 0:
                continue
                
            # Score based on required marker coverage
            required_score = required_matches / total_required
            optional_score = optional_matches / max(len(protocol_info['optional']), 1)
            
            # Weight required markers heavily
            total_score = 0.8 * required_score + 0.2 * optional_score
            
            if total_score > best_score:
                best_score = total_score
                best_protocol = protocol_name
        
        # Detect anatomical coverage
        anatomy_coverage = self._analyze_anatomical_coverage(markers_lower)
        
        # If no protocol matches well, classify as custom
        if best_score < 0.6:
            best_protocol = 'custom_clinical'
            best_score = anatomy_coverage['completeness_score']
        
        return {
            'protocol': best_protocol,
            'confidence': best_score,
            'anatomy_coverage': anatomy_coverage,
            'total_markers': len(self.marker_names),
            'recommendations': self._generate_analysis_recommendations(best_protocol, anatomy_coverage)
        }
    
    def _analyze_anatomical_coverage(self, markers_lower: List[str]) -> Dict:
        """
        Analyze what anatomical segments are covered by the marker set.
        
        Args:
            markers_lower: Lowercase marker names
            
        Returns:
            Dictionary with anatomical coverage information
        """
        # Define anatomical keywords
        segments = {
            'head': ['head', 'skull', 'frontal', 'occip'],
            'spine': ['c7', 't10', 't12', 'l5', 'spine', 'back'],
            'pelvis': ['asi', 'psi', 'iliac', 'pelv', 'hip'],
            'upper_arm': ['shoulder', 'humerus', 'arm'],
            'forearm': ['elbow', 'radius', 'ulna', 'forearm'],
            'hand': ['wrist', 'hand', 'finger'],
            'thigh': ['thigh', 'femur', 'thi'],
            'shank': ['knee', 'tibia', 'shin', 'shank'],
            'foot': ['ankle', 'heel', 'toe', 'foot', 'calc', 'meta']
        }
        
        coverage = {}
        for segment, keywords in segments.items():
            count = sum(1 for marker in markers_lower 
                       if any(keyword in marker for keyword in keywords))
            coverage[segment] = count > 0
        
        # Calculate bilateral symmetry
        left_markers = sum(1 for marker in markers_lower if 'l' in marker[:2])
        right_markers = sum(1 for marker in markers_lower if 'r' in marker[:2])
        bilateral_ratio = min(left_markers, right_markers) / max(left_markers, right_markers, 1)
        
        return {
            'segments': coverage,
            'segments_covered': sum(coverage.values()),
            'total_segments': len(segments),
            'completeness_score': sum(coverage.values()) / len(segments),
            'bilateral_symmetry': bilateral_ratio > 0.7,
            'bilateral_ratio': bilateral_ratio
        }
    
    def _generate_analysis_recommendations(self, protocol: str, anatomy: Dict) -> List[str]:
        """
        Generate biomechanically-informed analysis recommendations.
        
        Args:
            protocol: Detected marker protocol
            anatomy: Anatomical coverage information
            
        Returns:
            List of analysis recommendations
        """
        recommendations = []
        
        # Protocol-specific recommendations
        if protocol == 'helen_hayes':
            recommendations.append("Lower body gait analysis recommended")
            recommendations.append("Use sagittal plane joint angles")
        elif protocol == 'plug_in_gait':
            recommendations.append("Full body biomechanical analysis possible")
            recommendations.append("3D joint kinematics and kinetics recommended")
        elif protocol == 'opensim':
            recommendations.append("OpenSim musculoskeletal modeling recommended")
            recommendations.append("Muscle force estimation possible")
        elif protocol == 'opencap':
            recommendations.append("Clinical gait assessment appropriate")
            recommendations.append("Focus on gross movement patterns")
        
        # Anatomical coverage recommendations
        if anatomy['segments']['foot'] and anatomy['segments']['shank']:
            recommendations.append("Ankle biomechanics analysis possible")
        
        if anatomy['segments']['pelvis'] and anatomy['segments']['thigh']:
            recommendations.append("Hip joint analysis recommended")
        
        if not anatomy['bilateral_symmetry']:
            recommendations.append("Warning: Asymmetric marker placement detected")
        
        if anatomy['completeness_score'] < 0.5:
            recommendations.append("Limited anatomical coverage - focus on covered segments")
        
        return recommendations
    
    def get_position_array(self) -> np.ndarray:
        """
        Extract 3D position array from TRC data with robust handling of missing data.
        
        Returns:
            np.ndarray: Position array with shape (n_frames, n_markers, 3)
                       Missing values are preserved as NaN for downstream handling
        """
        if self.data is None or self.data.empty:
            return np.array([])
        
        positions = []
        missing_markers = []
        
        for marker in self.marker_names:
            # Use cleaned marker names for column lookup
            clean_marker = self._clean_marker_name(marker)
            marker_cols = [f'{clean_marker}_X', f'{clean_marker}_Y', f'{clean_marker}_Z']
            
            if all(col in self.data.columns for col in marker_cols):
                marker_data = self.data[marker_cols].values  # (n_frames, 3)
                positions.append(marker_data)
            else:
                missing_markers.append(marker)
                # Add NaN array for missing markers to maintain consistent indexing
                nan_data = np.full((len(self.data), 3), np.nan)
                positions.append(nan_data)
        
        if missing_markers:
            print(f"Warning: Missing data for markers: {missing_markers}")
        
        if positions:
            # Stack along marker axis: (n_frames, n_markers, 3)
            position_array = np.stack(positions, axis=1)
            return position_array
        else:
            return np.array([])
    
    def get_biomechanical_summary(self) -> Dict:
        """
        Generate a comprehensive biomechanical summary of the motion data.
        
        Returns:
            Dictionary with biomechanical analysis summary
        """
        if self.data is None or not hasattr(self, 'data_quality'):
            return {}
        
        # Get marker set information
        marker_info = self.detect_marker_set()
        
        # Calculate basic motion statistics
        positions = self.get_position_array()
        if positions.size == 0:
            return marker_info
        
        # Calculate movement characteristics
        velocities = np.diff(positions, axis=0) * self.data_rate if self.data_rate else np.array([])
        speeds = np.linalg.norm(velocities, axis=2) if velocities.size > 0 else np.array([])
        
        # Calculate workspace dimensions (movement volume)
        workspace = {}
        if positions.size > 0:
            # Remove NaN values for statistics
            valid_pos = positions[~np.isnan(positions)]
            if valid_pos.size > 0:
                workspace = {
                    'x_range': np.ptp(positions[:, :, 0][~np.isnan(positions[:, :, 0])]) if np.any(~np.isnan(positions[:, :, 0])) else 0,
                    'y_range': np.ptp(positions[:, :, 1][~np.isnan(positions[:, :, 1])]) if np.any(~np.isnan(positions[:, :, 1])) else 0,
                    'z_range': np.ptp(positions[:, :, 2][~np.isnan(positions[:, :, 2])]) if np.any(~np.isnan(positions[:, :, 2])) else 0,
                    'volume': np.prod([np.ptp(positions[:, :, i][~np.isnan(positions[:, :, i])]) for i in range(3)]) if positions.shape[2] >= 3 else 0
                }
        
        summary = {
            **marker_info,
            'data_quality': self.data_quality,
            'temporal_info': {
                'duration': self.data['Time'].iloc[-1] - self.data['Time'].iloc[0] if 'Time' in self.data.columns and len(self.data) > 0 else 0,
                'sampling_rate': self.data_rate,
                'num_frames': len(self.data)
            },
            'movement_characteristics': {
                'workspace': workspace,
                'max_speed': np.nanmax(speeds) if speeds.size > 0 else 0,
                'mean_speed': np.nanmean(speeds) if speeds.size > 0 else 0,
                'movement_variability': np.nanstd(speeds) if speeds.size > 0 else 0
            }
        }
        
        return summary
    
    def to_graph_format(self) -> dict:
        """
        Convert TRC data to format suitable for graph neural networks.
        
        Returns:
            dict: Dictionary with 'joint_names', 'positions', 'frame_rate'
        """
        return {
            'joint_names': self.marker_names,
            'positions': self.get_position_array(),
            'frame_rate': self.data_rate
        }
    
    def __repr__(self) -> str:
        """String representation of the TRCParser object."""
        return (f"TRCParser(file='{self.file_path}', "
                f"markers={self.num_markers}, "
                f"frames={self.num_frames}, "
                f"rate={self.data_rate}Hz)")
