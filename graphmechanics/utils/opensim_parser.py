"""
OpenSim File Parser

This module provides comprehensive parsing capabilities for OpenSim files, including:
- .osim model files (XML-based musculoskeletal models)
- .mot motion files (coordinate time series data)

The parser is designed with deep understanding of OpenSim's biomechanical modeling
principles and data structures, providing robust handling of the complex hierarchical
relationships between bodies, joints, coordinates, and muscles.

Author: Developed with insights from Scott Delp's OpenSim architecture
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import re
from dataclasses import dataclass, field
from collections import defaultdict
import warnings


@dataclass
class OpenSimJoint:
    """Represents an OpenSim joint with its properties and coordinates."""
    name: str
    type: str
    parent_body: str
    child_body: str
    coordinates: List[str] = field(default_factory=list)
    location_in_parent: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation_in_parent: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    location_in_child: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation_in_child: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class OpenSimBody:
    """Represents an OpenSim body with its properties."""
    name: str
    mass: float = 0.0
    mass_center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    inertia: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    attached_geometry: List[str] = field(default_factory=list)


@dataclass
class OpenSimMuscle:
    """Represents an OpenSim muscle with its properties."""
    name: str
    type: str
    max_isometric_force: float = 0.0
    optimal_fiber_length: float = 0.0
    tendon_slack_length: float = 0.0
    pennation_angle: float = 0.0
    path_points: List[Dict] = field(default_factory=list)


@dataclass
class OpenSimCoordinate:
    """Represents an OpenSim coordinate (degree of freedom)."""
    name: str
    motion_type: str  # 'rotational' or 'translational'
    default_value: float = 0.0
    range_min: float = -np.inf
    range_max: float = np.inf
    locked: bool = False
    prescribed: bool = False
    unit: str = "radians"  # or "meters"


class OpenSimModelParser:
    """
    Parser for OpenSim .osim model files.
    
    This class extracts the complete musculoskeletal model structure including
    bodies, joints, coordinates, muscles, and their hierarchical relationships.
    """
    
    def __init__(self, osim_file_path: str):
        """
        Initialize the OpenSim model parser.
        
        Args:
            osim_file_path (str): Path to the .osim model file
        """
        self.osim_file_path = osim_file_path
        self.model_name = None
        self.credits = None
        self.publications = None
        self.length_units = "meters"
        self.force_units = "N"
        
        # Model components
        self.bodies: Dict[str, OpenSimBody] = {}
        self.joints: Dict[str, OpenSimJoint] = {}
        self.coordinates: Dict[str, OpenSimCoordinate] = {}
        self.muscles: Dict[str, OpenSimMuscle] = {}
        
        # Hierarchical relationships
        self.body_hierarchy: Dict[str, List[str]] = defaultdict(list)  # parent -> children
        self.coordinate_to_joint: Dict[str, str] = {}  # coordinate -> joint
        
        # Parse the model
        self._parse_model()
    
    def _parse_model(self) -> None:
        """Parse the complete OpenSim model file."""
        try:
            tree = ET.parse(self.osim_file_path)
            root = tree.getroot()
            
            # Find the Model element
            model_elem = root.find('.//Model')
            if model_elem is None:
                raise ValueError("No Model element found in OpenSim file")
            
            self.model_name = model_elem.get('name', 'Unknown')
            
            # Parse model metadata
            self._parse_model_metadata(model_elem)
            
            # Parse bodies
            self._parse_bodies(model_elem)
            
            # Parse joints
            self._parse_joints(model_elem)
            
            # Parse coordinates (from joints)
            self._extract_coordinates()
            
            # Parse muscles
            self._parse_muscles(model_elem)
            
            # Build hierarchical relationships
            self._build_hierarchy()
            
            print(f"Successfully parsed OpenSim model: {self.model_name}")
            print(f"  Bodies: {len(self.bodies)}")
            print(f"  Joints: {len(self.joints)}")
            print(f"  Coordinates: {len(self.coordinates)}")
            print(f"  Muscles: {len(self.muscles)}")
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML in OpenSim file: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing OpenSim model: {e}")
    
    def _parse_model_metadata(self, model_elem: ET.Element) -> None:
        """Parse model metadata including credits, publications, and units."""
        self.credits = self._get_element_text(model_elem, 'credits', 'Unknown')
        self.publications = self._get_element_text(model_elem, 'publications', '')
        self.length_units = self._get_element_text(model_elem, 'length_units', 'meters')
        self.force_units = self._get_element_text(model_elem, 'force_units', 'N')
    
    def _parse_bodies(self, model_elem: ET.Element) -> None:
        """Parse all bodies in the model."""
        bodyset = model_elem.find('.//BodySet')
        if bodyset is None:
            return
        
        for body_elem in bodyset.findall('.//Body'):
            body = self._parse_single_body(body_elem)
            if body:
                self.bodies[body.name] = body
    
    def _parse_single_body(self, body_elem: ET.Element) -> Optional[OpenSimBody]:
        """Parse a single body element."""
        name = body_elem.get('name')
        if not name:
            return None
        
        body = OpenSimBody(name=name)
        
        # Parse mass properties
        body.mass = float(self._get_element_text(body_elem, 'mass', '0.0'))
        
        # Parse mass center
        mass_center_elem = body_elem.find('mass_center')
        if mass_center_elem is not None:
            body.mass_center = self._parse_vector3(mass_center_elem.text or "0 0 0")
        
        # Parse inertia
        inertia_elem = body_elem.find('inertia')
        if inertia_elem is not None:
            inertia_text = inertia_elem.text or "0 0 0 0 0 0"
            body.inertia = [float(x) for x in inertia_text.split()]
        
        # Parse attached geometry
        for geom_elem in body_elem.findall('.//Mesh'):
            mesh_file = self._get_element_text(geom_elem, 'mesh_file', '')
            if mesh_file:
                body.attached_geometry.append(mesh_file)
        
        return body
    
    def _parse_joints(self, model_elem: ET.Element) -> None:
        """Parse all joints in the model."""
        jointset = model_elem.find('.//JointSet')
        if jointset is None:
            return
        
        # Handle different joint types
        joint_types = ['PinJoint', 'SliderJoint', 'BallJoint', 'FreeJoint', 
                      'WeldJoint', 'UniversalJoint', 'CustomJoint', 'PlanarJoint']
        
        for joint_type in joint_types:
            for joint_elem in jointset.findall(f'.//{joint_type}'):
                joint = self._parse_single_joint(joint_elem, joint_type)
                if joint:
                    self.joints[joint.name] = joint
    
    def _parse_single_joint(self, joint_elem: ET.Element, joint_type: str) -> Optional[OpenSimJoint]:
        """Parse a single joint element."""
        name = joint_elem.get('name')
        if not name:
            return None
        
        # Parse parent and child frames (OpenSim uses socket_parent_frame and socket_child_frame)
        parent_body = self._get_element_text(joint_elem, 'socket_parent_frame', 'ground')
        child_body = self._get_element_text(joint_elem, 'socket_child_frame', '')
        
        # Try alternative format
        if not parent_body or parent_body == 'ground':
            parent_body = self._get_element_text(joint_elem, 'parent_body', 'ground')
        if not child_body:
            child_body = self._get_element_text(joint_elem, 'child_body', '')
        
        # Ensure we have valid body names (avoid empty strings)
        if not parent_body:
            parent_body = 'ground'
        if not child_body:
            child_body = f'child_of_{name}'
        
        joint = OpenSimJoint(name=name, type=joint_type, 
                           parent_body=parent_body, child_body=child_body)
        
        # Parse spatial transforms
        joint.location_in_parent = self._parse_vector3(
            self._get_element_text(joint_elem, 'location_in_parent', '0 0 0'))
        joint.orientation_in_parent = self._parse_vector3(
            self._get_element_text(joint_elem, 'orientation_in_parent', '0 0 0'))
        joint.location_in_child = self._parse_vector3(
            self._get_element_text(joint_elem, 'location_in_child', '0 0 0'))
        joint.orientation_in_child = self._parse_vector3(
            self._get_element_text(joint_elem, 'orientation_in_child', '0 0 0'))
        
        # Parse coordinates
        coordinates_elem = joint_elem.find('coordinates')
        if coordinates_elem is not None:
            for coord_elem in coordinates_elem.findall('Coordinate'):
                coord_name = coord_elem.get('name')
                if coord_name:
                    joint.coordinates.append(coord_name)
        
        return joint
    
    def _extract_coordinates(self) -> None:
        """Extract coordinate information from joints."""
        for joint_name, joint in self.joints.items():
            for coord_name in joint.coordinates:
                # Determine motion type based on joint type and coordinate name
                motion_type = self._determine_motion_type(joint.type, coord_name)
                unit = "radians" if motion_type == "rotational" else "meters"
                
                coordinate = OpenSimCoordinate(
                    name=coord_name,
                    motion_type=motion_type,
                    unit=unit
                )
                
                self.coordinates[coord_name] = coordinate
                self.coordinate_to_joint[coord_name] = joint_name
    
    def _parse_muscles(self, model_elem: ET.Element) -> None:
        """Parse all muscles in the model."""
        forceset = model_elem.find('.//ForceSet')
        if forceset is None:
            return
        
        # Handle different muscle types
        muscle_types = ['Millard2012EquilibriumMuscle', 'DeGrooteFregly2016Muscle',
                       'Thelen2003Muscle', 'RigidTendonMuscle']
        
        for muscle_type in muscle_types:
            for muscle_elem in forceset.findall(f'.//{muscle_type}'):
                muscle = self._parse_single_muscle(muscle_elem, muscle_type)
                if muscle:
                    self.muscles[muscle.name] = muscle
    
    def _parse_single_muscle(self, muscle_elem: ET.Element, muscle_type: str) -> Optional[OpenSimMuscle]:
        """Parse a single muscle element."""
        name = muscle_elem.get('name')
        if not name:
            return None
        
        muscle = OpenSimMuscle(name=name, type=muscle_type)
        
        # Parse muscle properties
        muscle.max_isometric_force = float(
            self._get_element_text(muscle_elem, 'max_isometric_force', '0.0'))
        muscle.optimal_fiber_length = float(
            self._get_element_text(muscle_elem, 'optimal_fiber_length', '0.0'))
        muscle.tendon_slack_length = float(
            self._get_element_text(muscle_elem, 'tendon_slack_length', '0.0'))
        muscle.pennation_angle = float(
            self._get_element_text(muscle_elem, 'pennation_angle_at_optimal', '0.0'))
        
        # Parse muscle path
        path_elem = muscle_elem.find('.//GeometryPath')
        if path_elem is not None:
            for point_elem in path_elem.findall('.//PathPoint'):
                point_name = point_elem.get('name', '')
                body = self._get_element_text(point_elem, 'body', '')
                location = self._parse_vector3(
                    self._get_element_text(point_elem, 'location', '0 0 0'))
                
                muscle.path_points.append({
                    'name': point_name,
                    'body': body,
                    'location': location
                })
        
        return muscle
    
    def _build_hierarchy(self) -> None:
        """Build the body hierarchy from joint connections."""
        for joint in self.joints.values():
            parent = joint.parent_body
            child = joint.child_body
            if parent and child:
                self.body_hierarchy[parent].append(child)
    
    def _determine_motion_type(self, joint_type: str, coord_name: str) -> str:
        """Determine if a coordinate represents rotational or translational motion."""
        # Common coordinate naming patterns
        translational_patterns = ['_tx', '_ty', '_tz', 'translation', 'displacement']
        rotational_patterns = ['_rotation', '_tilt', '_list', '_flexion', '_extension', 
                             '_adduction', '_abduction', 'angle', 'rot']
        
        coord_lower = coord_name.lower()
        
        # Check explicit patterns first
        if any(pattern in coord_lower for pattern in translational_patterns):
            return "translational"
        if any(pattern in coord_lower for pattern in rotational_patterns):
            return "rotational"
        
        # Joint-type based inference
        if joint_type in ['SliderJoint']:
            return "translational"
        elif joint_type in ['PinJoint']:
            return "rotational"
        elif joint_type in ['FreeJoint']:
            # FreeJoint typically has 6 DOF: 3 translations + 3 rotations
            if any(suffix in coord_lower for suffix in ['_tx', '_ty', '_tz']):
                return "translational"
            else:
                return "rotational"
        
        # Default assumption for biomechanical models
        return "rotational"
    
    def _get_element_text(self, parent: ET.Element, tag: str, default: str = '') -> str:
        """Safely get text content of a child element."""
        elem = parent.find(tag)
        return elem.text.strip() if elem is not None and elem.text else default
    
    def _parse_vector3(self, text: str) -> List[float]:
        """Parse a 3D vector from text."""
        try:
            values = [float(x) for x in text.split()]
            if len(values) >= 3:
                return values[:3]
            else:
                return values + [0.0] * (3 - len(values))
        except (ValueError, AttributeError):
            return [0.0, 0.0, 0.0]
    
    def get_coordinate_names(self) -> List[str]:
        """Get list of all coordinate names in the model."""
        return list(self.coordinates.keys())
    
    def get_joint_hierarchy(self) -> Dict[str, Dict]:
        """Get hierarchical representation of the joint structure."""
        hierarchy = {}
        for joint_name, joint in self.joints.items():
            hierarchy[joint_name] = {
                'type': joint.type,
                'parent_body': joint.parent_body,
                'child_body': joint.child_body,
                'coordinates': joint.coordinates,
                'dof': len(joint.coordinates)
            }
        return hierarchy
    
    def get_muscle_summary(self) -> Dict[str, Dict]:
        """Get summary of muscle properties."""
        summary = {}
        for muscle_name, muscle in self.muscles.items():
            summary[muscle_name] = {
                'type': muscle.type,
                'max_force': muscle.max_isometric_force,
                'fiber_length': muscle.optimal_fiber_length,
                'tendon_length': muscle.tendon_slack_length,
                'pennation_angle': muscle.pennation_angle,
                'path_points': len(muscle.path_points)
            }
        return summary


class OpenSimMotionParser:
    """
    Parser for OpenSim .mot motion files.
    
    This class handles the parsing of coordinate time series data, providing
    robust error handling and data validation for biomechanical analysis.
    """
    
    def __init__(self, mot_file_path: str):
        """
        Initialize the motion parser.
        
        Args:
            mot_file_path (str): Path to the .mot motion file
        """
        self.mot_file_path = mot_file_path
        self.version = None
        self.n_rows = 0
        self.n_columns = 0
        self.in_degrees = True
        self.units_description = ""
        self.coordinate_names = []
        self.data = None
        self.data_quality = {}
        
        # Parse the motion file
        self._parse_motion_file()
    
    def _parse_motion_file(self) -> None:
        """Parse the complete motion file."""
        try:
            with open(self.mot_file_path, 'r') as file:
                lines = file.readlines()
            
            # Parse header
            header_end_idx = self._parse_header(lines)
            
            # Parse coordinate names
            self._parse_coordinate_names(lines, header_end_idx)
            
            # Parse data
            self._parse_motion_data(lines, header_end_idx + 1)
            
            # Validate and clean data
            self._validate_motion_data()
            
            print(f"Successfully parsed motion file: {os.path.basename(self.mot_file_path)}")
            print(f"  Duration: {self.get_duration():.2f} seconds")
            print(f"  Frames: {len(self.data)}")
            print(f"  Coordinates: {len(self.coordinate_names)}")
            print(f"  Units: {'degrees' if self.in_degrees else 'radians'} (rotational)")
            
        except Exception as e:
            raise ValueError(f"Error parsing motion file: {e}")
    
    def _parse_header(self, lines: List[str]) -> int:
        """Parse the header section and return the index where data starts."""
        header_end_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Look for header fields
            if line.startswith('Coordinates'):
                continue
            elif line.startswith('version='):
                self.version = line.split('=')[1].strip()
            elif line.startswith('nRows='):
                self.n_rows = int(line.split('=')[1].strip())
            elif line.startswith('nColumns='):
                self.n_columns = int(line.split('=')[1].strip())
            elif line.startswith('inDegrees='):
                self.in_degrees = line.split('=')[1].strip().lower() == 'yes'
            elif line.startswith('Units are'):
                self.units_description = line
            elif line == 'endheader':
                header_end_idx = i + 1
                break
        
        return header_end_idx
    
    def _parse_coordinate_names(self, lines: List[str], start_idx: int) -> None:
        """Parse coordinate names from the header line after 'endheader'."""
        if start_idx < len(lines):
            header_line = lines[start_idx].strip()
            self.coordinate_names = [name.strip() for name in header_line.split('\t') if name.strip()]
    
    def _parse_motion_data(self, lines: List[str], start_idx: int) -> None:
        """Parse the numerical motion data."""
        data_rows = []
        skipped_lines = 0
        
        for line_num, line in enumerate(lines[start_idx:], start_idx):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Split by tab and convert to float
                values = line.split('\t')
                numeric_values = []
                
                for val in values:
                    val = val.strip()
                    if val == '' or val.lower() in ['nan', 'null', 'none']:
                        numeric_values.append(np.nan)
                    else:
                        try:
                            numeric_values.append(float(val))
                        except ValueError:
                            numeric_values.append(np.nan)
                
                # Validate row length
                expected_cols = len(self.coordinate_names)
                if len(numeric_values) != expected_cols:
                    if len(numeric_values) < expected_cols:
                        # Pad with NaN
                        numeric_values.extend([np.nan] * (expected_cols - len(numeric_values)))
                    else:
                        # Truncate
                        numeric_values = numeric_values[:expected_cols]
                
                data_rows.append(numeric_values)
                
            except Exception as e:
                skipped_lines += 1
                if skipped_lines <= 5:
                    print(f"Warning: Skipped line {line_num}: {str(e)[:50]}...")
        
        if skipped_lines > 5:
            print(f"... and {skipped_lines - 5} more lines with parsing errors")
        
        # Create DataFrame
        if data_rows and self.coordinate_names:
            self.data = pd.DataFrame(data_rows, columns=self.coordinate_names)
        else:
            self.data = pd.DataFrame()
    
    def _validate_motion_data(self) -> None:
        """Validate and assess the quality of motion data."""
        if self.data.empty:
            self.data_quality = {'completeness_percent': 0, 'total_frames': 0}
            return
        
        # Convert numeric columns
        for col in self.data.columns:
            if col not in ['time']:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Calculate data quality metrics
        total_points = self.data.size
        missing_points = self.data.isna().sum().sum()
        completeness = 100 * (1 - missing_points / total_points) if total_points > 0 else 0
        
        # Validate time sequence
        if 'time' in self.data.columns:
            time_col = self.data['time']
            time_diffs = np.diff(time_col.dropna())
            if len(time_diffs) > 0:
                sampling_rate = 1.0 / np.mean(time_diffs)
            else:
                sampling_rate = 0.0
        else:
            sampling_rate = 0.0
        
        # Validate coordinate ranges (biomechanically reasonable)
        coord_stats = {}
        for col in self.data.columns:
            if col != 'time':
                col_data = self.data[col].dropna()
                if len(col_data) > 0:
                    coord_stats[col] = {
                        'mean': float(np.mean(col_data)),
                        'std': float(np.std(col_data)),
                        'range': float(np.ptp(col_data)),
                        'min': float(np.min(col_data)),
                        'max': float(np.max(col_data))
                    }
        
        self.data_quality = {
            'completeness_percent': completeness,
            'total_frames': len(self.data),
            'missing_points': int(missing_points),
            'total_points': int(total_points),
            'sampling_rate': sampling_rate,
            'coordinate_statistics': coord_stats
        }
        
        if completeness < 95:
            print(f"Warning: Motion data completeness is {completeness:.1f}%")
    
    def get_duration(self) -> float:
        """Get the duration of the motion in seconds."""
        if self.data is None or 'time' not in self.data.columns:
            return 0.0
        time_col = self.data['time'].dropna()
        return float(time_col.iloc[-1] - time_col.iloc[0]) if len(time_col) > 1 else 0.0
    
    def get_coordinate_data(self, coordinate_name: str) -> Optional[pd.Series]:
        """Get time series data for a specific coordinate."""
        if self.data is None or coordinate_name not in self.data.columns:
            print(f"Coordinate '{coordinate_name}' not found. Available: {self.coordinate_names}")
            return None
        return self.data[coordinate_name].copy()
    
    def get_time_series(self, coordinate_names: List[str] = None) -> pd.DataFrame:
        """Get time series data for specified coordinates."""
        if self.data is None:
            return pd.DataFrame()
        
        if coordinate_names is None:
            return self.data.copy()
        
        # Validate coordinate names
        valid_coords = [coord for coord in coordinate_names if coord in self.data.columns]
        if 'time' not in valid_coords and 'time' in self.data.columns:
            valid_coords = ['time'] + valid_coords
        
        return self.data[valid_coords].copy()
    
    def convert_to_radians(self, coordinate_names: List[str] = None) -> pd.DataFrame:
        """Convert rotational coordinates from degrees to radians."""
        if not self.in_degrees or self.data is None:
            return self.data.copy() if self.data is not None else pd.DataFrame()
        
        data_copy = self.data.copy()
        coords_to_convert = coordinate_names or [col for col in self.coordinate_names if col != 'time']
        
        for coord in coords_to_convert:
            if coord in data_copy.columns and coord != 'time':
                # Only convert rotational coordinates (not translations)
                if not any(suffix in coord.lower() for suffix in ['_tx', '_ty', '_tz']):
                    data_copy[coord] = np.radians(data_copy[coord])
        
        return data_copy
    
    def export_to_csv(self, output_path: str = None) -> str:
        """Export motion data to CSV format."""
        if self.data is None:
            raise ValueError("No motion data available to export")
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.mot_file_path))[0]
            output_path = f"{base_name}_motion_data.csv"
        
        self.data.to_csv(output_path, index=False)
        print(f"Exported motion data to: {output_path}")
        return output_path


class OpenSimParser:
    """
    Comprehensive OpenSim parser that handles both model (.osim) and motion (.mot) files.
    
    This class provides a unified interface for parsing OpenSim files and extracting
    data suitable for graph neural network models and biomechanical analysis.
    """
    
    def __init__(self, osim_file_path: str = None, mot_file_path: str = None):
        """
        Initialize the OpenSim parser.
        
        Args:
            osim_file_path (str, optional): Path to .osim model file
            mot_file_path (str, optional): Path to .mot motion file
        """
        self.osim_file_path = osim_file_path
        self.mot_file_path = mot_file_path
        
        self.model_parser = None
        self.motion_parser = None
        
        # Parse files if provided
        if osim_file_path:
            self.model_parser = OpenSimModelParser(osim_file_path)
        
        if mot_file_path:
            self.motion_parser = OpenSimMotionParser(mot_file_path)
    
    def load_model(self, osim_file_path: str) -> None:
        """Load an OpenSim model file."""
        self.osim_file_path = osim_file_path
        self.model_parser = OpenSimModelParser(osim_file_path)
    
    def load_motion(self, mot_file_path: str) -> None:
        """Load an OpenSim motion file."""
        self.mot_file_path = mot_file_path
        self.motion_parser = OpenSimMotionParser(mot_file_path)
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary."""
        if not self.model_parser:
            return {}
        
        return {
            'model_name': self.model_parser.model_name,
            'credits': self.model_parser.credits,
            'publications': self.model_parser.publications,
            'units': {
                'length': self.model_parser.length_units,
                'force': self.model_parser.force_units
            },
            'components': {
                'bodies': len(self.model_parser.bodies),
                'joints': len(self.model_parser.joints),
                'coordinates': len(self.model_parser.coordinates),
                'muscles': len(self.model_parser.muscles)
            },
            'coordinate_names': self.model_parser.get_coordinate_names(),
            'joint_hierarchy': self.model_parser.get_joint_hierarchy()
        }
    
    def get_motion_summary(self) -> Dict:
        """Get comprehensive motion summary."""
        if not self.motion_parser:
            return {}
        
        return {
            'file_path': self.motion_parser.mot_file_path,
            'duration': self.motion_parser.get_duration(),
            'frames': len(self.motion_parser.data) if self.motion_parser.data is not None else 0,
            'coordinates': len(self.motion_parser.coordinate_names),
            'coordinate_names': self.motion_parser.coordinate_names,
            'in_degrees': self.motion_parser.in_degrees,
            'data_quality': self.motion_parser.data_quality,
            'sampling_info': {
                'estimated_rate': self.motion_parser.data_quality.get('sampling_rate', 0.0),
                'time_range': [
                    float(self.motion_parser.data['time'].iloc[0]) if self.motion_parser.data is not None and 'time' in self.motion_parser.data.columns else 0.0,
                    float(self.motion_parser.data['time'].iloc[-1]) if self.motion_parser.data is not None and 'time' in self.motion_parser.data.columns else 0.0
                ]
            }
        }
    
    def validate_model_motion_compatibility(self) -> Dict[str, Any]:
        """Validate compatibility between model and motion data."""
        if not self.model_parser or not self.motion_parser:
            return {
                'compatible': False,
                'reason': 'Both model and motion files required'
            }
        
        model_coords = set(self.model_parser.get_coordinate_names())
        motion_coords = set(self.motion_parser.coordinate_names)
        
        # Remove 'time' from motion coordinates for comparison
        motion_coords.discard('time')
        
        common_coords = model_coords.intersection(motion_coords)
        missing_in_motion = model_coords - motion_coords
        extra_in_motion = motion_coords - model_coords
        
        compatibility_score = len(common_coords) / len(model_coords) if model_coords else 0.0
        
        return {
            'compatible': compatibility_score > 0.8,
            'compatibility_score': compatibility_score,
            'common_coordinates': list(common_coords),
            'missing_in_motion': list(missing_in_motion),
            'extra_in_motion': list(extra_in_motion),
            'total_model_coords': len(model_coords),
            'total_motion_coords': len(motion_coords),
            'recommendations': self._generate_compatibility_recommendations(
                compatibility_score, missing_in_motion, extra_in_motion)
        }
    
    def _generate_compatibility_recommendations(self, score: float, missing: set, extra: set) -> List[str]:
        """Generate recommendations for improving model-motion compatibility."""
        recommendations = []
        
        if score < 0.5:
            recommendations.append("Low compatibility: Verify model and motion files are matched")
        elif score < 0.8:
            recommendations.append("Moderate compatibility: Some coordinates may be missing")
        
        if missing:
            recommendations.append(f"Model coordinates missing in motion: {list(missing)[:5]}")
        
        if extra:
            recommendations.append(f"Extra coordinates in motion file: {list(extra)[:5]}")
        
        if not recommendations:
            recommendations.append("Excellent compatibility between model and motion")
        
        return recommendations
    
    def to_graph_format(self) -> Dict[str, Any]:
        """
        Convert OpenSim data to format suitable for graph neural networks.
        
        Returns:
            dict: Dictionary with graph-compatible data structure
        """
        result = {
            'has_model': self.model_parser is not None,
            'has_motion': self.motion_parser is not None
        }
        
        if self.model_parser:
            # Extract coordinate information with biomechanical context
            coordinates = []
            coordinate_types = []
            joint_connections = []
            
            for coord_name, coord in self.model_parser.coordinates.items():
                coordinates.append(coord_name)
                coordinate_types.append(coord.motion_type)
                
                # Add joint connection information
                joint_name = self.model_parser.coordinate_to_joint.get(coord_name)
                if joint_name:
                    joint = self.model_parser.joints[joint_name]
                    joint_connections.append({
                        'coordinate': coord_name,
                        'joint': joint_name,
                        'parent_body': joint.parent_body,
                        'child_body': joint.child_body
                    })
            
            result['model_data'] = {
                'coordinate_names': coordinates,
                'coordinate_types': coordinate_types,
                'joint_connections': joint_connections,
                'body_hierarchy': dict(self.model_parser.body_hierarchy)
            }
        
        if self.motion_parser:
            # Get motion data in appropriate units
            motion_data = self.motion_parser.data
            if motion_data is not None:
                # Convert to radians if needed for rotational coordinates
                if self.motion_parser.in_degrees:
                    motion_data = self.motion_parser.convert_to_radians()
                
                # Extract coordinate positions (excluding time)
                coord_columns = [col for col in motion_data.columns if col != 'time']
                positions = motion_data[coord_columns].values  # Shape: (n_frames, n_coordinates)
                
                result['motion_data'] = {
                    'coordinates': coord_columns,
                    'positions': positions,
                    'time': motion_data['time'].values if 'time' in motion_data.columns else None,
                    'frame_rate': self.motion_parser.data_quality.get('sampling_rate', 30.0),
                    'in_radians': True  # Always convert to radians for neural networks
                }
        
        return result
    
    def save_model_to_osim(self, output_path: str) -> str:
        """Save the parsed model back to .osim format (basic implementation)."""
        if not self.model_parser:
            raise ValueError("No model data loaded")
        
        # This is a simplified implementation - full OpenSim XML writing would be complex
        print("Warning: Model saving is simplified. Use OpenSim API for complete functionality.")
        
        # For now, copy the original file
        import shutil
        shutil.copy2(self.osim_file_path, output_path)
        return output_path
    
    def save_motion_to_mot(self, output_path: str, coordinate_data: pd.DataFrame = None) -> str:
        """Save motion data to .mot format."""
        if not self.motion_parser and coordinate_data is None:
            raise ValueError("No motion data available")
        
        data_to_save = coordinate_data if coordinate_data is not None else self.motion_parser.data
        
        with open(output_path, 'w') as f:
            # Write header
            f.write("Coordinates\n")
            f.write("version=1\n")
            f.write(f"nRows={len(data_to_save)}\n")
            f.write(f"nColumns={len(data_to_save.columns)}\n")
            f.write("inDegrees=yes\n")  # Assume degrees for output
            f.write("\n")
            f.write("Units are S.I. units (second, meters, Newtons, ...)\n")
            f.write("If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n")
            f.write("\n")
            f.write("endheader\n")
            
            # Write column headers
            f.write('\t'.join(data_to_save.columns) + '\n')
            
            # Write data
            for _, row in data_to_save.iterrows():
                formatted_row = []
                for val in row:
                    if pd.isna(val):
                        formatted_row.append('0.00000000')
                    else:
                        formatted_row.append(f'{val:14.8f}')
                f.write('\t'.join(formatted_row) + '\n')
        
        print(f"Saved motion data to: {output_path}")
        return output_path
    
    def __repr__(self) -> str:
        """String representation of the OpenSimParser."""
        model_info = f"Model: {self.model_parser.model_name}" if self.model_parser else "Model: None"
        motion_info = f"Motion: {os.path.basename(self.mot_file_path)}" if self.mot_file_path else "Motion: None"
        return f"OpenSimParser({model_info}, {motion_info})"
