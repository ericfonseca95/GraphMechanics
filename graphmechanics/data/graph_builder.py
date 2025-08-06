"""
Graph construction utilities for motion capture data.

This module provides tools for converting motion capture marker data into
graph structures that represent the kinematic chains and relationships
between body segments based on expert anatomical knowledge and OpenSim conventions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Union, Set
import re
import warnings
from dataclasses import dataclass

__all__ = ['MotionGraphConverter', 'KinematicGraphBuilder', 'OpenSimGraphBuilder', 'create_motion_graph', 
           'BiomechanicalConstraints', 'JointLimits']


@dataclass
class JointLimits:
    """Biomechanical joint angle limits based on anatomical constraints."""
    
    # Hip joint limits (degrees)
    hip_flexion_min: float = -20.0
    hip_flexion_max: float = 120.0
    hip_abduction_min: float = -30.0
    hip_abduction_max: float = 45.0
    hip_rotation_min: float = -45.0
    hip_rotation_max: float = 45.0
    
    # Knee joint limits (degrees)
    knee_flexion_min: float = 0.0
    knee_flexion_max: float = 140.0
    
    # Ankle joint limits (degrees)
    ankle_flexion_min: float = -20.0  # Plantarflexion
    ankle_flexion_max: float = 30.0   # Dorsiflexion
    
    # Shoulder joint limits (degrees)
    shoulder_flexion_min: float = -40.0
    shoulder_flexion_max: float = 180.0
    shoulder_abduction_min: float = 0.0
    shoulder_abduction_max: float = 180.0
    shoulder_rotation_min: float = -90.0
    shoulder_rotation_max: float = 90.0
    
    # Elbow joint limits (degrees)
    elbow_flexion_min: float = 0.0
    elbow_flexion_max: float = 145.0
    
    # Maximum velocities (m/s for linear, rad/s for angular)
    max_linear_velocity: float = 10.0    # m/s (very fast human movement)
    max_angular_velocity: float = 15.0   # rad/s (very fast joint rotation)
    
    # Maximum accelerations (m/s² for linear, rad/s² for angular)
    max_linear_acceleration: float = 50.0   # m/s²
    max_angular_acceleration: float = 100.0  # rad/s²


class BiomechanicalConstraints:
    """
    Enforces biomechanical constraints for realistic human motion prediction.
    
    This class implements Scott Delp's approach to biomechanical validation,
    ensuring that predicted motions respect anatomical limits and physical laws.
    """
    
    def __init__(self, opensim_model_path: Optional[str] = None):
        """
        Initialize biomechanical constraints.
        
        Args:
            opensim_model_path: Path to OpenSim model file for detailed constraints
        """
        self.joint_limits = JointLimits()
        self.opensim_model_path = opensim_model_path
        
        # Standard bone lengths (meters) for adult human
        self.reference_bone_lengths = {
            'femur': 0.43,      # Average femur length
            'tibia': 0.37,      # Average tibia length  
            'humerus': 0.32,    # Average humerus length
            'radius': 0.25,     # Average radius length
            'foot': 0.26,       # Average foot length
            'pelvis_width': 0.28 # Average pelvis width
        }
        
        # Bone length tolerance (±10%)
        self.length_tolerance = 0.10
    
    def validate_pose(self, positions: np.ndarray, 
                     marker_names: Optional[List[str]] = None,
                     previous_pose: Optional[np.ndarray] = None) -> Tuple[bool, List[str]]:
        """
        Validate a pose against biomechanical constraints.
        
        Args:
            positions: Marker positions [num_markers, 3] 
            marker_names: List of marker names (optional)
            previous_pose: Previous pose for temporal validation (optional)
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        # Convert numpy to tensor
        if isinstance(positions, np.ndarray):
            pos_tensor = torch.from_numpy(positions).float()
        else:
            pos_tensor = positions
            
        # Use default marker names if not provided
        if marker_names is None:
            marker_names = [f"marker_{i}" for i in range(len(positions))]
        
        if pos_tensor.dim() == 2:
            pos_tensor = pos_tensor.unsqueeze(0)  # Add batch dimension
        
        batch_size = pos_tensor.shape[0]
        violations = []
        
        # 1. Check bone length preservation
        bone_length_violations = self._check_bone_lengths(pos_tensor, marker_names)
        violations.extend(bone_length_violations)
        
        # 2. Check ground penetration
        ground_violations = self._check_ground_penetration(pos_tensor, marker_names)
        violations.extend(ground_violations)
        
        # 3. Check anatomical range limits
        range_violations = self._check_anatomical_ranges(pos_tensor, marker_names)
        violations.extend(range_violations)
        
        # 4. Check bilateral symmetry (within reason)
        symmetry_violations = self._check_bilateral_symmetry(pos_tensor, marker_names)
        violations.extend(symmetry_violations)
        
        # 5. Temporal consistency if previous pose provided
        if previous_pose is not None:
            if isinstance(previous_pose, np.ndarray):
                prev_tensor = torch.from_numpy(previous_pose).float()
            else:
                prev_tensor = previous_pose
            temporal_violations = self._check_temporal_consistency(prev_tensor, pos_tensor, marker_names)
            violations.extend(temporal_violations)
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def _check_bone_lengths(self, positions: torch.Tensor, marker_names: List[str]) -> List[str]:
        """Check if bone lengths are within physiological ranges."""
        violations = []
        name_to_idx = {name: i for i, name in enumerate(marker_names)}
        
        # Define bone pairs and their reference lengths
        bone_pairs = [
            (['r_hip', 'RHJC_study'], ['r_knee_lateral', 'r_knee_study'], 'femur'),
            (['l_hip', 'LHJC_study'], ['l_knee_lateral', 'L_knee_study'], 'femur'),
            (['r_knee_lateral', 'r_knee_study'], ['r_ankle_lateral', 'r_ankle_study'], 'tibia'),
            (['l_knee_lateral', 'L_knee_study'], ['l_ankle_lateral', 'L_ankle_study'], 'tibia'),
            (['r_shoulder', 'RShoulder'], ['r_elbow_lateral', 'RElbow'], 'humerus'),
            (['l_shoulder', 'LShoulder'], ['l_elbow_lateral', 'LElbow'], 'humerus'),
            (['RASI', 'r.ASIS_study'], ['LASI', 'L.ASIS_study'], 'pelvis_width'),
        ]
        
        for proximal_markers, distal_markers, bone_name in bone_pairs:
            # Find available markers for this bone
            proximal_idx = None
            distal_idx = None
            
            for marker in proximal_markers:
                if marker in name_to_idx:
                    proximal_idx = name_to_idx[marker]
                    break
                    
            for marker in distal_markers:
                if marker in name_to_idx:
                    distal_idx = name_to_idx[marker]
                    break
            
            if proximal_idx is not None and distal_idx is not None:
                # Compute bone length
                bone_vectors = positions[:, distal_idx] - positions[:, proximal_idx]
                bone_lengths = torch.norm(bone_vectors, dim=1)
                
                reference_length = self.reference_bone_lengths[bone_name]
                min_length = reference_length * (1 - self.length_tolerance)
                max_length = reference_length * (1 + self.length_tolerance)
                
                # Check violations
                too_short = bone_lengths < min_length
                too_long = bone_lengths > max_length
                
                if too_short.any():
                    violations.append(f"{bone_name}_too_short")
                if too_long.any():
                    violations.append(f"{bone_name}_too_long")
        
        return violations
    
    def _check_ground_penetration(self, positions: torch.Tensor, marker_names: List[str]) -> List[str]:
        """Check if any markers penetrate the ground plane."""
        violations = []
        name_to_idx = {name: i for i, name in enumerate(marker_names)}
        
        # Find foot markers that should not go below ground
        foot_markers = ['r_heel', 'l_heel', 'RHeel', 'LHeel', 'r_calc_study', 'L_calc_study',
                       'r_toe', 'l_toe', 'RBigToe', 'LBigToe', 'r_toe_study', 'L_toe_study']
        
        ground_level = 0.0  # Assume ground at z=0
        
        for marker in foot_markers:
            if marker in name_to_idx:
                marker_idx = name_to_idx[marker]
                z_positions = positions[:, marker_idx, 2]  # Z coordinate
                
                if (z_positions < ground_level - 0.01).any():  # 1cm tolerance
                    violations.append(f"ground_penetration_{marker}")
        
        return violations
    
    def _check_anatomical_ranges(self, positions: torch.Tensor, marker_names: List[str]) -> List[str]:
        """Check if joint angles are within anatomical ranges."""
        violations = []
        # This is simplified - full implementation would compute actual joint angles
        
        # For now, check basic position ranges (assuming reasonable lab space)
        max_lab_range = 5.0  # 5 meters from origin
        
        if (torch.abs(positions) > max_lab_range).any():
            violations.append("positions_out_of_lab_range")
        
        return violations
    
    def _check_bilateral_symmetry(self, positions: torch.Tensor, marker_names: List[str]) -> List[str]:
        """Check bilateral symmetry within reasonable bounds."""
        violations = []
        name_to_idx = {name: i for i, name in enumerate(marker_names)}
        
        # Define symmetric marker pairs
        symmetric_pairs = [
            ('r_shoulder', 'l_shoulder'), ('RShoulder', 'LShoulder'),
            ('r_hip', 'l_hip'), ('RHip', 'LHip'),
            ('r_knee_lateral', 'l_knee_lateral'), ('RKnee', 'LKnee'),
            ('r_ankle_lateral', 'l_ankle_lateral'), ('RAnkle', 'LAnkle')
        ]
        
        for right_marker, left_marker in symmetric_pairs:
            if right_marker in name_to_idx and left_marker in name_to_idx:
                right_idx = name_to_idx[right_marker]
                left_idx = name_to_idx[left_marker]
                
                right_pos = positions[:, right_idx]
                left_pos = positions[:, left_idx]
                
                # Check Y-coordinate symmetry (assuming Y is medial-lateral)
                right_y = right_pos[:, 1]
                left_y = left_pos[:, 1]
                
                # They should be roughly symmetric about midline
                symmetry_error = torch.abs(right_y + left_y)  # Should sum to ~0
                
                if (symmetry_error > 0.2).any():  # 20cm asymmetry threshold
                    violations.append(f"asymmetry_{right_marker}_{left_marker}")
        
        return violations
    
    def _check_temporal_consistency(self, prev_positions: torch.Tensor, 
                                  curr_positions: torch.Tensor, 
                                  marker_names: List[str]) -> List[str]:
        """Check temporal consistency between consecutive poses."""
        violations = []
        
        # Flatten to 2D if needed
        if prev_positions.dim() == 3:
            prev_positions = prev_positions.squeeze(0)
        if curr_positions.dim() == 3:
            curr_positions = curr_positions.squeeze(0)
        
        # Compute frame-to-frame velocity
        dt = 1.0 / 120.0  # Assume 120 Hz sampling
        velocity = (curr_positions - prev_positions) / dt
        
        # Check for unrealistic velocities (>10 m/s for any marker)
        max_velocity = 10.0  # m/s
        velocity_magnitude = torch.norm(velocity, dim=1)
        
        for i, vel_mag in enumerate(velocity_magnitude):
            if vel_mag > max_velocity:
                marker_name = marker_names[i] if i < len(marker_names) else f"marker_{i}"
                violations.append(f"excessive_velocity_{marker_name}")
        
        # Check for sudden direction changes (acceleration threshold)
        if hasattr(self, '_prev_velocity'):
            acceleration = (velocity - self._prev_velocity) / dt
            max_acceleration = 50.0  # m/s^2
            accel_magnitude = torch.norm(acceleration, dim=1)
            
            for i, accel_mag in enumerate(accel_magnitude):
                if accel_mag > max_acceleration:
                    marker_name = marker_names[i] if i < len(marker_names) else f"marker_{i}"
                    violations.append(f"excessive_acceleration_{marker_name}")
        
        # Store current velocity for next frame
        self._prev_velocity = velocity
        
        return violations
    
    def apply_constraints(self, positions: np.ndarray, 
                         reference_pose: Optional[np.ndarray] = None,
                         marker_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply biomechanical constraints to correct invalid poses.
        
        Args:
            positions: Positions to correct [num_markers, 3]
            reference_pose: Reference pose for correction (optional)
            marker_names: Marker names (optional)
            
        Returns:
            Corrected positions [num_markers, 3]
        """
        corrected = positions.copy()
        
        # Use default marker names if not provided
        if marker_names is None:
            marker_names = [f"marker_{i}" for i in range(len(positions))]
        
        # 1. Ensure no ground penetration (simple correction)
        corrected[:, 2] = np.maximum(corrected[:, 2], 0.0)  # Z >= 0
        
        # 2. Enforce basic joint limits (simplified)
        if reference_pose is not None:
            # Limit position changes to reasonable velocities
            max_displacement = 0.5  # 50cm max change per frame
            displacement = corrected - reference_pose
            displacement_magnitude = np.linalg.norm(displacement, axis=1, keepdims=True)
            
            # Scale down excessive displacements
            excessive_mask = displacement_magnitude > max_displacement
            if excessive_mask.any():
                scale_factor = max_displacement / (displacement_magnitude + 1e-8)
                corrected[excessive_mask.squeeze()] = (
                    reference_pose[excessive_mask.squeeze()] + 
                    displacement[excessive_mask.squeeze()] * scale_factor[excessive_mask.squeeze()]
                )
        
        # 3. Preserve approximate bone lengths
        if reference_pose is not None and len(positions) >= 4:
            # Simple bone length preservation for major segments
            # This is a simplified version - full implementation would need detailed anatomy
            name_to_idx = {name: i for i, name in enumerate(marker_names)}
            
            # Basic pelvis-to-knee preservation if markers available
            for side in ['L', 'R']:
                hip_marker = f"{side}ASI"
                knee_marker = f"{side}KNE"
                
                if hip_marker in name_to_idx and knee_marker in name_to_idx:
                    hip_idx = name_to_idx[hip_marker]
                    knee_idx = name_to_idx[knee_marker]
                    
                    # Get reference bone length
                    ref_length = np.linalg.norm(reference_pose[knee_idx] - reference_pose[hip_idx])
                    
                    # Get current bone vector
                    current_vector = corrected[knee_idx] - corrected[hip_idx]
                    current_length = np.linalg.norm(current_vector)
                    
                    # Preserve bone length
                    if current_length > 0:
                        corrected[knee_idx] = (corrected[hip_idx] + 
                                             (current_vector / current_length) * ref_length)
        
        return corrected
    
    def compute_biomechanical_loss_simple(self, 
                                         predictions: np.ndarray, 
                                         targets: Optional[np.ndarray] = None) -> float:
        """
        Compute biomechanical constraint loss for training (simple version).
        
        Args:
            predictions: Predicted positions [num_markers, 3]
            targets: Target positions (optional) [num_markers, 3]
            
        Returns:
            Biomechanical loss value
        """
        loss = 0.0
        
        # 1. Ground penetration penalty
        ground_penalty = np.sum(np.maximum(0, -predictions[:, 2]))  # Penalty for Z < 0
        loss += ground_penalty * 10.0
        
        # 2. Unrealistic position penalty (too high)
        height_penalty = np.sum(np.maximum(0, predictions[:, 2] - 3.0))  # Penalty for Z > 3m
        loss += height_penalty * 5.0
        
        # 3. If targets provided, check for bone length changes
        if targets is not None:
            # Simple bone length consistency check
            if len(predictions) >= 4:
                # Check distances between first few markers as proxy for bone lengths
                for i in range(len(predictions) - 1):
                    pred_dist = np.linalg.norm(predictions[i+1] - predictions[i])
                    target_dist = np.linalg.norm(targets[i+1] - targets[i])
                    
                    # Penalty for bone length changes > 10%
                    length_change = abs(pred_dist - target_dist) / (target_dist + 1e-8)
                    if length_change > 0.1:
                        loss += length_change * 2.0
        
        return loss
    
    
    
    def compute_biomechanical_loss(
        self, 
        predictions: torch.Tensor, 
        marker_names: List[str],
        edge_index: torch.Tensor,
        loss_weights: Dict[str, float] = None
    ) -> torch.Tensor:
        """
        Compute biomechanical loss for training.
        
        Args:
            predictions: Predicted positions [batch, seq_len, num_markers, 3]
            marker_names: List of marker names
            edge_index: Graph connectivity
            loss_weights: Weights for different loss components
            
        Returns:
            Combined biomechanical loss
        """
        if loss_weights is None:
            loss_weights = {
                'bone_length': 1.0,
                'ground_contact': 2.0,
                'velocity': 0.5,
                'acceleration': 0.3,
                'symmetry': 0.5
            }
        
        batch_size, seq_len, num_markers, _ = predictions.shape
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        # 1. Bone length preservation loss
        bone_length_loss = self._compute_bone_length_loss(predictions, marker_names, edge_index)
        total_loss += loss_weights['bone_length'] * bone_length_loss
        
        # 2. Ground contact loss
        ground_loss = self._compute_ground_contact_loss(predictions, marker_names)
        total_loss += loss_weights['ground_contact'] * ground_loss
        
        # 3. Velocity constraint loss
        if seq_len > 1:
            velocity_loss = self._compute_velocity_constraint_loss(predictions)
            total_loss += loss_weights['velocity'] * velocity_loss
        
        # 4. Acceleration constraint loss
        if seq_len > 2:
            acceleration_loss = self._compute_acceleration_constraint_loss(predictions)
            total_loss += loss_weights['acceleration'] * acceleration_loss
        
        # 5. Bilateral symmetry loss
        symmetry_loss = self._compute_symmetry_loss(predictions, marker_names)
        total_loss += loss_weights['symmetry'] * symmetry_loss
        
        return total_loss
    
    def _compute_bone_length_loss(
        self, 
        predictions: torch.Tensor, 
        marker_names: List[str],
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for bone length preservation."""
        batch_size, seq_len, num_markers, _ = predictions.shape
        device = predictions.device
        
        # Reshape for easier processing
        pred_flat = predictions.view(-1, num_markers, 3)
        
        # Compute distances along edges (bones)
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        source_pos = pred_flat[:, source_nodes]  # [batch*seq, num_edges, 3]
        target_pos = pred_flat[:, target_nodes]  # [batch*seq, num_edges, 3]
        
        edge_lengths = torch.norm(source_pos - target_pos, dim=-1)  # [batch*seq, num_edges]
        
        # Compute temporal consistency of bone lengths
        edge_lengths = edge_lengths.view(batch_size, seq_len, -1)  # [batch, seq, num_edges]
        
        if seq_len > 1:
            # Bone lengths should be constant over time
            length_variance = torch.var(edge_lengths, dim=1)  # [batch, num_edges]
            bone_length_loss = torch.mean(length_variance)
        else:
            bone_length_loss = torch.tensor(0.0, device=device)
        
        return bone_length_loss
    
    def _compute_ground_contact_loss(self, predictions: torch.Tensor, marker_names: List[str]) -> torch.Tensor:
        """Compute loss for ground contact constraints."""
        device = predictions.device
        name_to_idx = {name: i for i, name in enumerate(marker_names)}
        
        # Find foot markers
        foot_markers = ['r_heel', 'l_heel', 'RHeel', 'LHeel', 'r_calc_study', 'L_calc_study']
        foot_indices = [name_to_idx[marker] for marker in foot_markers if marker in name_to_idx]
        
        if not foot_indices:
            return torch.tensor(0.0, device=device)
        
        # Extract z-coordinates of foot markers
        foot_z = predictions[:, :, foot_indices, 2]  # [batch, seq, num_foot_markers]
        
        # Penalize negative z-coordinates (below ground)
        ground_penetration = F.relu(-foot_z)  # Only penalize negative values
        ground_loss = torch.mean(ground_penetration)
        
        return ground_loss
    
    def _compute_velocity_constraint_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute loss for velocity constraints."""
        if predictions.shape[1] < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute velocities (assuming 120 Hz sampling)
        dt = 1.0 / 120.0
        velocities = (predictions[:, 1:] - predictions[:, :-1]) / dt
        
        # Compute velocity magnitudes
        velocity_magnitudes = torch.norm(velocities, dim=-1)
        
        # Penalize excessive velocities
        max_vel = self.joint_limits.max_linear_velocity
        velocity_violations = F.relu(velocity_magnitudes - max_vel)
        
        return torch.mean(velocity_violations)
    
    def _compute_acceleration_constraint_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute loss for acceleration constraints."""
        if predictions.shape[1] < 3:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute accelerations
        dt = 1.0 / 120.0
        velocities = (predictions[:, 1:] - predictions[:, :-1]) / dt
        accelerations = (velocities[:, 1:] - velocities[:, :-1]) / dt
        
        # Compute acceleration magnitudes
        acceleration_magnitudes = torch.norm(accelerations, dim=-1)
        
        # Penalize excessive accelerations
        max_accel = self.joint_limits.max_linear_acceleration
        acceleration_violations = F.relu(acceleration_magnitudes - max_accel)
        
        return torch.mean(acceleration_violations)
    
    def _compute_symmetry_loss(self, predictions: torch.Tensor, marker_names: List[str]) -> torch.Tensor:
        """Compute loss for bilateral symmetry."""
        device = predictions.device
        name_to_idx = {name: i for i, name in enumerate(marker_names)}
        
        symmetric_pairs = [
            ('r_shoulder', 'l_shoulder'), ('RShoulder', 'LShoulder'),
            ('r_hip', 'l_hip'), ('RHip', 'LHip'),
            ('r_knee_lateral', 'l_knee_lateral'), ('RKnee', 'LKnee'),
        ]
        
        symmetry_losses = []
        
        for right_marker, left_marker in symmetric_pairs:
            if right_marker in name_to_idx and left_marker in name_to_idx:
                right_idx = name_to_idx[right_marker]
                left_idx = name_to_idx[left_marker]
                
                right_pos = predictions[:, :, right_idx]  # [batch, seq, 3]
                left_pos = predictions[:, :, left_idx]   # [batch, seq, 3]
                
                # Symmetry constraint: Y-coordinates should be opposite
                right_y = right_pos[:, :, 1]
                left_y = left_pos[:, :, 1]
                
                # They should sum to approximately zero (symmetric about midline)
                symmetry_error = torch.abs(right_y + left_y)
                symmetry_losses.append(torch.mean(symmetry_error))
        
        if symmetry_losses:
            return torch.mean(torch.stack(symmetry_losses))
        else:
            return torch.tensor(0.0, device=device)


class OpenSimGraphBuilder:
    """
    Graph builder specialized for OpenSim marker conventions and anatomical knowledge.
    
    This class implements expert biomechanical knowledge about human anatomy,
    kinematic chains, and OpenSim marker naming conventions to create
    physiologically meaningful graph representations.
    """
    
    def __init__(self):
        """Initialize with OpenSim marker conventions and anatomical knowledge."""
        self.opensim_marker_mapping = self._create_opensim_marker_mapping()
        self.kinematic_chains = self._define_kinematic_chains()
        self.joint_hierarchy = self._define_joint_hierarchy()
        self.segment_definitions = self._define_body_segments()
    
    def _create_opensim_marker_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Create mapping from common marker naming conventions to OpenSim standards.
        
        Returns:
            Dictionary mapping marker variations to standardized OpenSim names
        """
        return {
            # Head and Neck
            'head': {
                'variations': ['HEAD', 'Head', 'head', 'LFHD', 'RFHD', 'LBHD', 'RBHD'],
                'opensim': 'head',
                'segment': 'head'
            },
            'neck': {
                'variations': ['NECK', 'Neck', 'neck', 'C7', 'c7'],
                'opensim': 'C7',
                'segment': 'torso'
            },
            
            # Torso
            'sternum': {
                'variations': ['STRN', 'sternum', 'Sternum', 'CLAV'],
                'opensim': 'sternum',
                'segment': 'torso'
            },
            'xiphoid': {
                'variations': ['XIPH', 'xiphoid', 'Xiphoid'],
                'opensim': 'xiphoid',
                'segment': 'torso'
            },
            
            # Pelvis
            'pelvis_right': {
                'variations': ['RASI', 'r.ASIS', 'R.ASIS', 'RASIS_study', 'r.ASIS_study'],
                'opensim': 'RASI',
                'segment': 'pelvis'
            },
            'pelvis_left': {
                'variations': ['LASI', 'L.ASIS', 'l.ASIS', 'LASIS_study', 'L.ASIS_study'],
                'opensim': 'LASI',
                'segment': 'pelvis'
            },
            'sacrum': {
                'variations': ['SACR', 'sacrum', 'Sacrum', 'PSIS'],
                'opensim': 'SACR',
                'segment': 'pelvis'
            },
            
            # Right Arm
            'r_shoulder': {
                'variations': ['RSHO', 'RShoulder', 'r_shoulder', 'r_shoulder_study'],
                'opensim': 'r_shoulder',
                'segment': 'r_humerus'
            },
            'r_elbow_lateral': {
                'variations': ['RELB', 'RElbow', 'r_lelbow', 'r_lelbow_study'],
                'opensim': 'r_elbow_lateral',
                'segment': 'r_radius'
            },
            'r_elbow_medial': {
                'variations': ['r_melbow', 'r_melbow_study', 'RELBM'],
                'opensim': 'r_elbow_medial',
                'segment': 'r_radius'
            },
            'r_wrist_lateral': {
                'variations': ['RWRA', 'RWrist', 'r_lwrist', 'r_lwrist_study'],
                'opensim': 'r_wrist_lateral',
                'segment': 'r_hand'
            },
            'r_wrist_medial': {
                'variations': ['RWRB', 'r_mwrist', 'r_mwrist_study'],
                'opensim': 'r_wrist_medial',
                'segment': 'r_hand'
            },
            
            # Left Arm
            'l_shoulder': {
                'variations': ['LSHO', 'LShoulder', 'L_shoulder', 'L_shoulder_study'],
                'opensim': 'l_shoulder',
                'segment': 'l_humerus'
            },
            'l_elbow_lateral': {
                'variations': ['LELB', 'LElbow', 'L_lelbow', 'L_lelbow_study'],
                'opensim': 'l_elbow_lateral',
                'segment': 'l_radius'
            },
            'l_elbow_medial': {
                'variations': ['L_melbow', 'L_melbow_study', 'LELBM'],
                'opensim': 'l_elbow_medial',
                'segment': 'l_radius'
            },
            'l_wrist_lateral': {
                'variations': ['LWRA', 'LWrist', 'L_lwrist', 'L_lwrist_study'],
                'opensim': 'l_wrist_lateral',
                'segment': 'l_hand'
            },
            'l_wrist_medial': {
                'variations': ['LWRB', 'L_mwrist', 'L_mwrist_study'],
                'opensim': 'l_wrist_medial',
                'segment': 'l_hand'
            },
            
            # Right Leg
            'r_hip': {
                'variations': ['RHip', 'RHJC', 'RHJC_study'],
                'opensim': 'r_hip',
                'segment': 'r_femur'
            },
            'r_knee_lateral': {
                'variations': ['RKNE', 'RKnee', 'r_knee', 'r_knee_study'],
                'opensim': 'r_knee_lateral',
                'segment': 'r_tibia'
            },
            'r_knee_medial': {
                'variations': ['r_mknee', 'r_mknee_study', 'RKNEM'],
                'opensim': 'r_knee_medial',
                'segment': 'r_tibia'
            },
            'r_ankle_lateral': {
                'variations': ['RANK', 'RAnkle', 'r_ankle', 'r_ankle_study'],
                'opensim': 'r_ankle_lateral',
                'segment': 'r_foot'
            },
            'r_ankle_medial': {
                'variations': ['r_mankle', 'r_mankle_study', 'RANKM'],
                'opensim': 'r_ankle_medial',
                'segment': 'r_foot'
            },
            'r_heel': {
                'variations': ['RHEE', 'RHeel', 'r_calc', 'r_calc_study'],
                'opensim': 'r_heel',
                'segment': 'r_foot'
            },
            'r_toe': {
                'variations': ['RTOE', 'RBigToe', 'r_toe', 'r_toe_study'],
                'opensim': 'r_toe',
                'segment': 'r_foot'
            },
            'r_metatarsal': {
                'variations': ['r_5meta', 'r_5meta_study', 'RSmallToe'],
                'opensim': 'r_metatarsal_5',
                'segment': 'r_foot'
            },
            
            # Left Leg
            'l_hip': {
                'variations': ['LHip', 'LHJC', 'LHJC_study'],
                'opensim': 'l_hip',
                'segment': 'l_femur'
            },
            'l_knee_lateral': {
                'variations': ['LKNE', 'LKnee', 'L_knee', 'L_knee_study'],
                'opensim': 'l_knee_lateral',
                'segment': 'l_tibia'
            },
            'l_knee_medial': {
                'variations': ['L_mknee', 'L_mknee_study', 'LKNEM'],
                'opensim': 'l_knee_medial',
                'segment': 'l_tibia'
            },
            'l_ankle_lateral': {
                'variations': ['LANK', 'LAnkle', 'L_ankle', 'L_ankle_study'],
                'opensim': 'l_ankle_lateral',
                'segment': 'l_foot'
            },
            'l_ankle_medial': {
                'variations': ['L_mankle', 'L_mankle_study', 'LANKM'],
                'opensim': 'l_ankle_medial',
                'segment': 'l_foot'
            },
            'l_heel': {
                'variations': ['LHEE', 'LHeel', 'L_calc', 'L_calc_study'],
                'opensim': 'l_heel',
                'segment': 'l_foot'
            },
            'l_toe': {
                'variations': ['LTOE', 'LBigToe', 'L_toe', 'L_toe_study'],
                'opensim': 'l_toe',
                'segment': 'l_foot'
            },
            'l_metatarsal': {
                'variations': ['L_5meta', 'L_5meta_study', 'LSmallToe'],
                'opensim': 'l_metatarsal_5',
                'segment': 'l_foot'
            }
        }
    
    def _define_kinematic_chains(self) -> Dict[str, List[str]]:
        """
        Define anatomical kinematic chains based on biomechanical knowledge.
        
        Returns:
            Dictionary of kinematic chains with proximal to distal ordering
        """
        return {
            'axial_skeleton': [
                'pelvis', 'L5', 'L4', 'L3', 'L2', 'L1', 
                'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5', 'T4', 'T3', 'T2', 'T1',
                'C7', 'C6', 'C5', 'C4', 'C3', 'C2', 'C1', 'head'
            ],
            'right_arm': [
                'r_clavicle', 'r_shoulder', 'r_humerus', 'r_elbow_lateral', 'r_elbow_medial',
                'r_radius', 'r_ulna', 'r_wrist_lateral', 'r_wrist_medial', 'r_hand'
            ],
            'left_arm': [
                'l_clavicle', 'l_shoulder', 'l_humerus', 'l_elbow_lateral', 'l_elbow_medial',
                'l_radius', 'l_ulna', 'l_wrist_lateral', 'l_wrist_medial', 'l_hand'
            ],
            'right_leg': [
                'r_hip', 'r_femur', 'r_knee_lateral', 'r_knee_medial',
                'r_tibia', 'r_fibula', 'r_ankle_lateral', 'r_ankle_medial',
                'r_foot', 'r_heel', 'r_toe', 'r_metatarsal_5'
            ],
            'left_leg': [
                'l_hip', 'l_femur', 'l_knee_lateral', 'l_knee_medial',
                'l_tibia', 'l_fibula', 'l_ankle_lateral', 'l_ankle_medial',
                'l_foot', 'l_heel', 'l_toe', 'l_metatarsal_5'
            ]
        }
    
    def _define_joint_hierarchy(self) -> Dict[str, str]:
        """
        Define parent-child relationships in the kinematic tree.
        
        Returns:
            Dictionary mapping child joints to parent joints
        """
        return {
            # Spine hierarchy
            'head': 'C7',
            'C7': 'T1',
            'T1': 'T2', 'T2': 'T3', 'T3': 'T4', 'T4': 'T5', 'T5': 'T6',
            'T6': 'T7', 'T7': 'T8', 'T8': 'T9', 'T9': 'T10', 'T10': 'T11',
            'T11': 'T12', 'T12': 'L1', 'L1': 'L2', 'L2': 'L3', 'L3': 'L4',
            'L4': 'L5', 'L5': 'pelvis',
            
            # Right arm hierarchy
            'r_shoulder': 'r_clavicle',
            'r_clavicle': 'T1',
            'r_elbow_lateral': 'r_shoulder',
            'r_elbow_medial': 'r_shoulder',
            'r_wrist_lateral': 'r_elbow_lateral',
            'r_wrist_medial': 'r_elbow_medial',
            'r_hand': 'r_wrist_lateral',
            
            # Left arm hierarchy
            'l_shoulder': 'l_clavicle',
            'l_clavicle': 'T1',
            'l_elbow_lateral': 'l_shoulder',
            'l_elbow_medial': 'l_shoulder',
            'l_wrist_lateral': 'l_elbow_lateral',
            'l_wrist_medial': 'l_elbow_medial',
            'l_hand': 'l_wrist_lateral',
            
            # Right leg hierarchy
            'r_hip': 'pelvis',
            'r_knee_lateral': 'r_hip',
            'r_knee_medial': 'r_hip',
            'r_ankle_lateral': 'r_knee_lateral',
            'r_ankle_medial': 'r_knee_medial',
            'r_heel': 'r_ankle_lateral',
            'r_toe': 'r_ankle_lateral',
            'r_metatarsal_5': 'r_ankle_lateral',
            
            # Left leg hierarchy
            'l_hip': 'pelvis',
            'l_knee_lateral': 'l_hip',
            'l_knee_medial': 'l_hip',
            'l_ankle_lateral': 'l_knee_lateral',
            'l_ankle_medial': 'l_knee_medial',
            'l_heel': 'l_ankle_lateral',
            'l_toe': 'l_ankle_lateral',
            'l_metatarsal_5': 'l_ankle_lateral',
        }
    
    def _define_body_segments(self) -> Dict[str, List[str]]:
        """
        Define body segments for biomechanical analysis.
        
        Returns:
            Dictionary mapping segments to their constituent markers
        """
        return {
            'head': ['head', 'C7'],
            'torso': ['C7', 'sternum', 'xiphoid', 'T8', 'T12'],
            'pelvis': ['RASI', 'LASI', 'SACR', 'pelvis'],
            'r_humerus': ['r_shoulder', 'r_elbow_lateral', 'r_elbow_medial'],
            'r_radius': ['r_elbow_lateral', 'r_elbow_medial', 'r_wrist_lateral', 'r_wrist_medial'],
            'r_hand': ['r_wrist_lateral', 'r_wrist_medial'],
            'l_humerus': ['l_shoulder', 'l_elbow_lateral', 'l_elbow_medial'],
            'l_radius': ['l_elbow_lateral', 'l_elbow_medial', 'l_wrist_lateral', 'l_wrist_medial'],
            'l_hand': ['l_wrist_lateral', 'l_wrist_medial'],
            'r_femur': ['r_hip', 'r_knee_lateral', 'r_knee_medial'],
            'r_tibia': ['r_knee_lateral', 'r_knee_medial', 'r_ankle_lateral', 'r_ankle_medial'],
            'r_foot': ['r_ankle_lateral', 'r_ankle_medial', 'r_heel', 'r_toe', 'r_metatarsal_5'],
            'l_femur': ['l_hip', 'l_knee_lateral', 'l_knee_medial'],
            'l_tibia': ['l_knee_lateral', 'l_knee_medial', 'l_ankle_lateral', 'l_ankle_medial'],
            'l_foot': ['l_ankle_lateral', 'l_ankle_medial', 'l_heel', 'l_toe', 'l_metatarsal_5']
        }
    
    def standardize_marker_names(self, marker_names: List[str]) -> Dict[str, str]:
        """
        Map input marker names to standardized OpenSim conventions.
        
        This uses biomechanical knowledge to handle the enormous variety of
        marker naming conventions across different labs and systems.
        
        Args:
            marker_names: List of marker names from TRC file
            
        Returns:
            Dictionary mapping original names to standardized names
        """
        name_mapping = {}
        
        for marker in marker_names:
            marker_lower = marker.lower().strip()
            standardized = None
            
            # Direct mapping first
            for standard_name, info in self.opensim_marker_mapping.items():
                if marker_lower in [v.lower() for v in info['variations']]:
                    standardized = info['opensim']
                    break
            
            # Fuzzy matching if direct mapping fails
            if standardized is None:
                standardized = self._fuzzy_match_marker(marker_lower)
            
            # Use original name if no mapping found (preserve everything)
            name_mapping[marker] = standardized or marker
        
        return name_mapping
    
    def _fuzzy_match_marker(self, marker_name: str) -> Optional[str]:
        """
        Attempt fuzzy matching for unrecognized marker names using anatomical knowledge.
        
        This is where decades of biomechanics experience pays off - understanding
        how different labs name similar anatomical landmarks.
        
        Args:
            marker_name: Marker name to match
            
        Returns:
            Best matching standardized name or None
        """
        marker_lower = marker_name.lower()
        
        # Lateral/Medial patterns for paired markers
        def has_lateral_indicator(name):
            return any(x in name for x in ['lat', 'lateral', '_l', 'l_', 'left'])
        
        def has_medial_indicator(name):
            return any(x in name for x in ['med', 'medial', '_m', 'm_'])
        
        def has_right_indicator(name):
            return any(x in name for x in ['r_', '_r', 'right', 'rgt'])
        
        def has_left_indicator(name):
            return any(x in name for x in ['l_', '_l', 'left', 'lft'])
        
        # Shoulder complex
        if any(x in marker_lower for x in ['shoulder', 'sho', 'acromion', 'acrm']):
            if has_right_indicator(marker_lower):
                return 'r_shoulder'
            elif has_left_indicator(marker_lower):
                return 'l_shoulder'
            return 'r_shoulder'  # Default to right if unclear
        
        # Elbow complex - critical for arm kinematics
        if any(x in marker_lower for x in ['elbow', 'elb', 'epicondyl']):
            side = 'r' if has_right_indicator(marker_lower) else 'l'
            if has_lateral_indicator(marker_lower):
                return f'{side}_elbow_lateral'
            elif has_medial_indicator(marker_lower):
                return f'{side}_elbow_medial'
            return f'{side}_elbow_lateral'  # Default to lateral
        
        # Wrist complex
        if any(x in marker_lower for x in ['wrist', 'wr', 'styloid']):
            side = 'r' if has_right_indicator(marker_lower) else 'l'
            if has_lateral_indicator(marker_lower) or 'radial' in marker_lower:
                return f'{side}_wrist_lateral'
            elif has_medial_indicator(marker_lower) or 'ulnar' in marker_lower:
                return f'{side}_wrist_medial'
            return f'{side}_wrist_lateral'
        
        # Knee complex - biomechanically critical
        if any(x in marker_lower for x in ['knee', 'kne', 'condyl']):
            side = 'r' if has_right_indicator(marker_lower) else 'l'
            if has_lateral_indicator(marker_lower):
                return f'{side}_knee_lateral'
            elif has_medial_indicator(marker_lower):
                return f'{side}_knee_medial'
            return f'{side}_knee_lateral'
        
        # Ankle complex
        if any(x in marker_lower for x in ['ankle', 'ank', 'malleol']):
            side = 'r' if has_right_indicator(marker_lower) else 'l'
            if has_lateral_indicator(marker_lower):
                return f'{side}_ankle_lateral'
            elif has_medial_indicator(marker_lower):
                return f'{side}_ankle_medial'
            return f'{side}_ankle_lateral'
        
        # Hip/Pelvis - fundamental for gait
        if any(x in marker_lower for x in ['hip', 'asis', 'iliac']):
            if has_right_indicator(marker_lower):
                return 'r_hip'
            elif has_left_indicator(marker_lower):
                return 'l_hip'
            return 'r_hip'
        
        # Spine markers
        if any(x in marker_lower for x in ['c7', 'cervical']):
            return 'C7'
        if any(x in marker_lower for x in ['t10', 'thoracic']):
            return 'T10'
        
        # Foot markers
        if any(x in marker_lower for x in ['heel', 'calc', 'calcaneus']):
            side = 'r' if has_right_indicator(marker_lower) else 'l'
            return f'{side}_heel'
        
        if any(x in marker_lower for x in ['toe', 'hallux', 'big']):
            side = 'r' if has_right_indicator(marker_lower) else 'l'
            return f'{side}_toe'
        
        return None
    
    def create_anatomical_graph(self, marker_names: List[str]) -> torch.Tensor:
        """
        Create graph edges based on anatomical connections and biomechanical constraints.
        
        This leverages decades of biomechanics knowledge to create physiologically
        meaningful graph structures that respect kinematic chains and joint constraints.
        
        Args:
            marker_names: List of marker names in the data
            
        Returns:
            Edge index tensor [2, num_edges]
        """
        # Standardize marker names
        name_mapping = self.standardize_marker_names(marker_names)
        standardized_names = [name_mapping.get(name, name) for name in marker_names]
        
        # Create mapping from standardized names to indices
        name_to_idx = {name: i for i, name in enumerate(standardized_names)}
        
        edges = []
        
        # 1. KINEMATIC CHAIN CONNECTIONS
        # Connect markers within each kinematic chain (proximal to distal)
        for chain_name, chain_markers in self.kinematic_chains.items():
            chain_indices = []
            for marker in chain_markers:
                if marker in name_to_idx:
                    chain_indices.append(name_to_idx[marker])
            
            # Connect adjacent markers in the chain
            for i in range(len(chain_indices) - 1):
                edges.append([chain_indices[i], chain_indices[i + 1]])
                edges.append([chain_indices[i + 1], chain_indices[i]])  # Bidirectional
        
        # 2. RIGID BODY SEGMENT CONNECTIONS
        # Connect markers on the same rigid body segment
        for segment_name, segment_markers in self.segment_definitions.items():
            segment_indices = []
            for marker in segment_markers:
                if marker in name_to_idx:
                    segment_indices.append(name_to_idx[marker])
            
            # Fully connect markers within each rigid segment
            for i in range(len(segment_indices)):
                for j in range(i + 1, len(segment_indices)):
                    edges.append([segment_indices[i], segment_indices[j]])
                    edges.append([segment_indices[j], segment_indices[i]])
        
        # 3. BILATERAL SYMMETRY CONNECTIONS
        # Connect corresponding markers on left and right sides
        bilateral_pairs = [
            ('r_shoulder', 'l_shoulder'),
            ('r_elbow_lateral', 'l_elbow_lateral'),
            ('r_elbow_medial', 'l_elbow_medial'),
            ('r_wrist_lateral', 'l_wrist_lateral'),
            ('r_wrist_medial', 'l_wrist_medial'),
            ('r_hip', 'l_hip'),
            ('r_knee_lateral', 'l_knee_lateral'),
            ('r_knee_medial', 'l_knee_medial'),
            ('r_ankle_lateral', 'l_ankle_lateral'),
            ('r_ankle_medial', 'l_ankle_medial'),
            ('r_heel', 'l_heel'),
            ('r_toe', 'l_toe')
        ]
        
        for right_marker, left_marker in bilateral_pairs:
            if right_marker in name_to_idx and left_marker in name_to_idx:
                right_idx = name_to_idx[right_marker]
                left_idx = name_to_idx[left_marker]
                edges.append([right_idx, left_idx])
                edges.append([left_idx, right_idx])
        
        # 4. SPECIAL ANATOMICAL CONNECTIONS
        # Connect joint-related markers (lateral/medial pairs)
        joint_pairs = [
            ('r_elbow_lateral', 'r_elbow_medial'),
            ('l_elbow_lateral', 'l_elbow_medial'),
            ('r_wrist_lateral', 'r_wrist_medial'),
            ('l_wrist_lateral', 'l_wrist_medial'),
            ('r_knee_lateral', 'r_knee_medial'),
            ('l_knee_lateral', 'l_knee_medial'),
            ('r_ankle_lateral', 'r_ankle_medial'),
            ('l_ankle_lateral', 'l_ankle_medial')
        ]
        
        for marker1, marker2 in joint_pairs:
            if marker1 in name_to_idx and marker2 in name_to_idx:
                idx1 = name_to_idx[marker1]
                idx2 = name_to_idx[marker2]
                edges.append([idx1, idx2])
                edges.append([idx2, idx1])
        
        # 5. TRUNK STABILITY CONNECTIONS
        # Connect core stability markers
        trunk_markers = ['C7', 'T10', 'pelvis', 'RASI', 'LASI', 'SACR']
        trunk_indices = [name_to_idx[marker] for marker in trunk_markers if marker in name_to_idx]
        
        for i in range(len(trunk_indices)):
            for j in range(i + 1, len(trunk_indices)):
                edges.append([trunk_indices[i], trunk_indices[j]])
                edges.append([trunk_indices[j], trunk_indices[i]])
        
        # Remove duplicates and convert to tensor
        if edges:
            edges = list(set(tuple(edge) for edge in edges))
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # Fallback: create a minimal connected graph
            n_nodes = len(marker_names)
            edges = [[i, (i + 1) % n_nodes] for i in range(n_nodes)]
            edges += [[(i + 1) % n_nodes, i] for i in range(n_nodes)]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edge_index
    
    def analyze_marker_set_coverage(self, marker_names: List[str]) -> Dict[str, any]:
        """
        Analyze the anatomical coverage and quality of the marker set.
        
        Args:
            marker_names: List of marker names
            
        Returns:
            Dictionary with coverage analysis
        """
        standardized = self.standardize_marker_names(marker_names)
        
        # Count coverage by body segment
        segment_coverage = {}
        for segment, markers in self.segment_definitions.items():
            covered = sum(1 for marker in markers if marker in standardized.values())
            segment_coverage[segment] = {
                'markers_present': covered,
                'total_markers': len(markers),
                'coverage_ratio': covered / len(markers) if markers else 0
            }
        
        # Assess kinematic chain completeness
        chain_coverage = {}
        for chain, markers in self.kinematic_chains.items():
            covered = sum(1 for marker in markers if marker in standardized.values())
            chain_coverage[chain] = {
                'markers_present': covered,
                'total_markers': len(markers),
                'coverage_ratio': covered / len(markers) if markers else 0
            }
        
        # Overall assessment
        total_possible = sum(len(markers) for markers in self.segment_definitions.values())
        total_covered = sum(info['markers_present'] for info in segment_coverage.values())
        
        return {
            'segment_coverage': segment_coverage,
            'chain_coverage': chain_coverage,
            'overall_coverage': total_covered / total_possible if total_possible > 0 else 0,
            'anatomical_quality': self._assess_anatomical_quality(segment_coverage),
            'recommended_analyses': self._recommend_analyses(segment_coverage, chain_coverage)
        }
    
    def _assess_anatomical_quality(self, segment_coverage: Dict) -> str:
        """Assess the quality of anatomical coverage for biomechanical analysis."""
        avg_coverage = np.mean([info['coverage_ratio'] for info in segment_coverage.values()])
        
        if avg_coverage > 0.8:
            return "Excellent - Full biomechanical analysis possible"
        elif avg_coverage > 0.6:
            return "Good - Most analyses possible with some limitations"
        elif avg_coverage > 0.4:
            return "Fair - Limited to specific segments and movements"
        else:
            return "Poor - Consider additional markers for reliable analysis"
    
    def _recommend_analyses(self, segment_coverage: Dict, chain_coverage: Dict) -> List[str]:
        """Recommend appropriate biomechanical analyses based on marker coverage."""
        recommendations = []
        
        # Lower extremity analysis
        if (segment_coverage.get('r_femur', {}).get('coverage_ratio', 0) > 0.5 and
            segment_coverage.get('r_tibia', {}).get('coverage_ratio', 0) > 0.5):
            recommendations.append("Lower extremity joint kinematics")
        
        # Upper extremity analysis
        if (segment_coverage.get('r_humerus', {}).get('coverage_ratio', 0) > 0.5 and
            segment_coverage.get('r_radius', {}).get('coverage_ratio', 0) > 0.5):
            recommendations.append("Upper extremity joint kinematics")
        
        # Gait analysis
        if (chain_coverage.get('right_leg', {}).get('coverage_ratio', 0) > 0.6 and
            chain_coverage.get('left_leg', {}).get('coverage_ratio', 0) > 0.6):
            recommendations.append("Comprehensive gait analysis")
        
        # Postural analysis
        if (segment_coverage.get('pelvis', {}).get('coverage_ratio', 0) > 0.5 and
            segment_coverage.get('torso', {}).get('coverage_ratio', 0) > 0.3):
            recommendations.append("Postural and balance analysis")
        
        return recommendations


class KinematicGraphBuilder:
    """
    Builds graph representations of human kinematic chains from motion capture data.
    
    This class creates graphs where:
    - Nodes represent markers with 3D coordinates and kinematic features
    - Edges represent anatomical connections (bones, joints)
    - Edge weights represent biomechanical relationships
    - Biomechanical constraints ensure physiologically valid predictions
    """
    
    def __init__(self, connectivity_type: str = 'opensim', 
                 use_anatomical_knowledge: bool = True,
                 enforce_constraints: bool = True,
                 use_biomechanical_constraints: bool = True,
                 marker_names: Optional[List[str]] = None):
        """
        Initialize the graph builder.
        
        Args:
            connectivity_type (str): Type of connectivity to use:
                - 'opensim': OpenSim-based anatomical connections
                - 'skeletal': Basic skeletal structure
                - 'distance': Based on spatial proximity
                - 'custom': User-defined connectivity
            use_anatomical_knowledge (bool): Whether to use expert anatomical knowledge
            enforce_constraints (bool): Whether to enforce biomechanical constraints
            use_biomechanical_constraints (bool): Alias for enforce_constraints
            marker_names (List[str], optional): List of marker names
        """
        self.connectivity_type = connectivity_type
        self.use_anatomical_knowledge = use_anatomical_knowledge
        self.enforce_constraints = enforce_constraints or use_biomechanical_constraints
        self.marker_names = marker_names or []
        
        if use_anatomical_knowledge and connectivity_type == 'opensim':
            self.opensim_builder = OpenSimGraphBuilder()
        else:
            self.opensim_builder = None
            
        # Initialize biomechanical constraints
        if self.enforce_constraints:
            self.biomechanical_constraints = BiomechanicalConstraints()
            self.constraints = self.biomechanical_constraints  # Keep both for compatibility
        else:
            self.biomechanical_constraints = None
            self.constraints = None
            
        self.marker_connections = self._get_default_connections()
    
    def _get_default_connections(self) -> Dict[str, List[str]]:
        """
        Define default skeletal connections for common marker sets.
        
        Returns:
            Dict mapping marker names to their connected neighbors
        """
        # Basic biomechanics marker connections (fallback)
        connections = {
            # Head and neck
            'Neck': ['RShoulder', 'LShoulder', 'C7_study'],
            'C7_study': ['Neck', 'RShoulder', 'LShoulder'],
            
            # Right arm chain
            'RShoulder': ['Neck', 'RElbow', 'r_shoulder_study'],
            'RElbow': ['RShoulder', 'RWrist', 'r_lelbow_study', 'r_melbow_study'],
            'RWrist': ['RElbow', 'r_lwrist_study', 'r_mwrist_study'],
            
            # Left arm chain
            'LShoulder': ['Neck', 'LElbow', 'L_shoulder_study'],
            'LElbow': ['LShoulder', 'LWrist', 'L_lelbow_study', 'L_melbow_study'],
            'LWrist': ['LElbow', 'L_lwrist_study', 'L_mwrist_study'],
            
            # Torso
            'midHip': ['RHip', 'LHip', 'r.ASIS_study', 'L.ASIS_study'],
            
            # Right leg chain
            'RHip': ['midHip', 'RKnee', 'r.ASIS_study', 'RHJC_study'],
            'RKnee': ['RHip', 'RAnkle', 'r_knee_study', 'r_mknee_study'],
            'RAnkle': ['RKnee', 'RBigToe', 'RHeel', 'r_ankle_study', 'r_mankle_study'],
            'RBigToe': ['RAnkle', 'RSmallToe', 'r_toe_study'],
            'RSmallToe': ['RBigToe', 'r_5meta_study'],
            'RHeel': ['RAnkle', 'r_calc_study'],
            
            # Left leg chain
            'LHip': ['midHip', 'LKnee', 'L.ASIS_study', 'LHJC_study'],
            'LKnee': ['LHip', 'LAnkle', 'L_knee_study', 'L_mknee_study'],
            'LAnkle': ['LKnee', 'LBigToe', 'LHeel', 'L_ankle_study', 'L_mankle_study'],
            'LBigToe': ['LAnkle', 'LSmallToe', 'L_toe_study'],
            'LSmallToe': ['LBigToe', 'L_5meta_study'],
            'LHeel': ['LAnkle', 'L_calc_study'],
        }
        
        return connections
    
    def set_custom_connections(self, connections: Dict[str, List[str]]):
        """
        Set custom marker connections.
        
        Args:
            connections (Dict): Dictionary mapping marker names to connected neighbors
        """
        self.marker_connections = connections
        self.connectivity_type = 'custom'
    
    def build_edge_index(self, marker_names: List[str]) -> torch.Tensor:
        """
        Build edge index tensor for the graph.
        
        Args:
            marker_names (List[str]): List of marker names in the data
            
        Returns:
            torch.Tensor: Edge index tensor of shape [2, num_edges]
        """
        if self.connectivity_type == 'opensim' and self.opensim_builder:
            return self.opensim_builder.create_anatomical_graph(marker_names)
        
        edges = []
        marker_to_idx = {name: idx for idx, name in enumerate(marker_names)}
        
        if self.connectivity_type in ['skeletal', 'custom']:
            # Use predefined connections
            for marker, connections in self.marker_connections.items():
                if marker in marker_to_idx:
                    marker_idx = marker_to_idx[marker]
                    for connected_marker in connections:
                        if connected_marker in marker_to_idx:
                            connected_idx = marker_to_idx[connected_marker]
                            edges.append([marker_idx, connected_idx])
                            edges.append([connected_idx, marker_idx])
        
        elif self.connectivity_type == 'distance':
            # Create connections based on all pairs (complete graph)
            for i in range(len(marker_names)):
                for j in range(i + 1, len(marker_names)):
                    edges.append([i, j])
                    edges.append([j, i])
        
        if not edges:
            # Fallback: create a simple chain if no connections found
            for i in range(len(marker_names) - 1):
                edges.append([i, i + 1])
                edges.append([i + 1, i])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def compute_edge_weights(
        self, 
        node_features: torch.Tensor, 
        edge_index: torch.Tensor,
        weight_type: str = 'distance'
    ) -> torch.Tensor:
        """
        Compute edge weights based on node features and biomechanical knowledge.
        
        Args:
            node_features (torch.Tensor): Node features [num_nodes, feature_dim]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            weight_type (str): Type of edge weights:
                - 'uniform': All edges have weight 1
                - 'distance': Inverse Euclidean distance
                - 'anatomical': Based on anatomical knowledge
                - 'kinematic': Based on kinematic relationships
            
        Returns:
            torch.Tensor: Edge weights [num_edges]
        """
        if weight_type == 'uniform':
            return torch.ones(edge_index.size(1))
        
        elif weight_type == 'distance':
            # Compute Euclidean distances between connected nodes
            source_nodes = edge_index[0]
            target_nodes = edge_index[1]
            
            # Extract position features (assuming first 3 features are x,y,z)
            source_pos = node_features[source_nodes, :3]
            target_pos = node_features[target_nodes, :3]
            
            # Compute distances
            distances = torch.norm(source_pos - target_pos, dim=1)
            
            # Convert to weights (inverse distance, avoid division by zero)
            weights = 1.0 / (distances + 1e-6)
            return weights
        
        elif weight_type == 'anatomical':
            # Weight based on anatomical importance
            # This is a simplified example - real implementation would use
            # expert knowledge about joint importance, muscle attachments, etc.
            weights = torch.ones(edge_index.size(1))
            
            # Give higher weights to primary kinematic chain connections
            # (This would be expanded with actual anatomical knowledge)
            primary_connections = {
                # Major joint connections get higher weights
                'hip-knee': 2.0,
                'knee-ankle': 2.0,
                'shoulder-elbow': 2.0,
                'elbow-wrist': 2.0,
                'pelvis-spine': 3.0,
                'spine-head': 1.5
            }
            
            return weights
        
        elif weight_type == 'kinematic':
            # Weight based on kinematic constraints and degrees of freedom
            source_nodes = edge_index[0]
            target_nodes = edge_index[1]
            
            # Extract velocity features if available (features 3:6)
            if node_features.size(1) >= 6:
                source_vel = node_features[source_nodes, 3:6]
                target_vel = node_features[target_nodes, 3:6]
                
                # Compute velocity correlation as weight
                vel_diff = torch.norm(source_vel - target_vel, dim=1)
                weights = torch.exp(-vel_diff)  # Gaussian-like weighting
                return weights
            else:
                return torch.ones(edge_index.size(1))
        
        else:
            return torch.ones(edge_index.size(1))
    
    def validate_motion_sequence(
        self, 
        positions: torch.Tensor, 
        marker_names: List[str],
        return_details: bool = False
    ) -> Union[float, Dict]:
        """
        Validate a motion sequence against biomechanical constraints.
        
        Args:
            positions: Motion sequence [batch, seq_len, num_markers, 3] or [seq_len, num_markers, 3]
            marker_names: List of marker names
            return_details: Whether to return detailed validation results
            
        Returns:
            Validation score (0-1) or detailed validation dictionary
        """
        if self.constraints is None:
            warnings.warn("Biomechanical constraints not enabled. Enable with enforce_constraints=True")
            return 1.0 if not return_details else {'validity_score': 1.0, 'message': 'constraints_disabled'}
        
        # Handle different input shapes
        if positions.dim() == 3:
            positions = positions.unsqueeze(0)  # Add batch dimension
        
        batch_size, seq_len, num_markers, _ = positions.shape
        
        # Validate each frame in the sequence
        all_scores = []
        all_violations = []
        
        for batch_idx in range(batch_size):
            for frame_idx in range(seq_len):
                frame_positions = positions[batch_idx, frame_idx]  # [num_markers, 3]
                validation_result = self.constraints.validate_pose(frame_positions, marker_names)
                all_scores.append(validation_result['validity_score'])
                all_violations.extend(validation_result['violations'])
        
        # Compute overall statistics
        mean_score = np.mean(all_scores)
        min_score = np.min(all_scores)
        violation_counts = {}
        for violation in all_violations:
            violation_counts[violation] = violation_counts.get(violation, 0) + 1
        
        if return_details:
            return {
                'validity_score': mean_score,
                'min_validity_score': min_score,
                'total_violations': len(all_violations),
                'violation_counts': violation_counts,
                'frames_validated': len(all_scores),
                'recommendation': self._get_validation_recommendation(mean_score)
            }
        else:
            return mean_score
    
    def _get_validation_recommendation(self, score: float) -> str:
        """Get recommendation based on validation score."""
        if score > 0.9:
            return "Excellent - Motion is biomechanically valid"
        elif score > 0.7:
            return "Good - Minor biomechanical issues detected"
        elif score > 0.5:
            return "Fair - Consider reviewing motion for biomechanical plausibility"
        else:
            return "Poor - Motion contains significant biomechanical violations"
    
    def compute_biomechanical_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        marker_names: List[str],
        edge_index: torch.Tensor,
        loss_weights: Dict[str, float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute biomechanical loss for training with physics constraints.
        
        This implements Scott Delp's approach to physics-informed learning,
        combining reconstruction loss with biomechanical constraints.
        
        Args:
            predictions: Predicted positions [batch, seq_len, num_markers, 3]
            targets: Target positions [batch, seq_len, num_markers, 3]
            marker_names: List of marker names
            edge_index: Graph connectivity
            loss_weights: Weights for different loss components
            
        Returns:
            Dictionary of loss components
        """
        if self.constraints is None:
            # Return only reconstruction loss if constraints disabled
            reconstruction_loss = F.mse_loss(predictions, targets)
            return {
                'total_loss': reconstruction_loss,
                'reconstruction_loss': reconstruction_loss,
                'biomechanical_loss': torch.tensor(0.0, device=predictions.device)
            }
        
        # Default loss weights from biomechanics literature
        if loss_weights is None:
            loss_weights = {
                'reconstruction': 1.0,     # Standard MSE loss
                'bone_length': 2.0,        # Critical for anatomical validity
                'ground_contact': 3.0,     # Critical for gait analysis
                'velocity': 0.5,           # Smoothness constraint
                'acceleration': 0.3,       # Smoothness constraint
                'symmetry': 0.8,           # Important for human motion
            }
        
        # 1. Reconstruction loss (standard)
        reconstruction_loss = F.mse_loss(predictions, targets)
        
        # 2. Biomechanical constraint loss
        biomechanical_loss = self.constraints.compute_biomechanical_loss(
            predictions, marker_names, edge_index, 
            {k: v for k, v in loss_weights.items() if k != 'reconstruction'}
        )
        
        # 3. Combined loss
        total_loss = (loss_weights['reconstruction'] * reconstruction_loss + 
                     biomechanical_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'biomechanical_loss': biomechanical_loss,
            'loss_weights': loss_weights
        }
    
    def create_temporal_edges(
        self, 
        num_nodes: int, 
        sequence_length: int,
        temporal_connections: int = 1
    ) -> torch.Tensor:
        """
        Create temporal edges connecting the same markers across time steps.
        
        Args:
            num_nodes (int): Number of nodes per time step
            sequence_length (int): Number of time steps
            temporal_connections (int): Number of previous time steps to connect
            
        Returns:
            torch.Tensor: Temporal edge index [2, num_temporal_edges]
        """
        edges = []
        
        for t in range(1, sequence_length):
            for dt in range(1, min(temporal_connections + 1, t + 1)):
                for node in range(num_nodes):
                    current_node = t * num_nodes + node
                    previous_node = (t - dt) * num_nodes + node
                    
                    # Add bidirectional temporal connections
                    edges.append([previous_node, current_node])
                    edges.append([current_node, previous_node])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()


class MotionGraphConverter:
    """
    Converts TRC motion capture data to PyTorch Geometric Data objects for time-series graph transformers.
    """
    
    def __init__(self, graph_builder: Optional[KinematicGraphBuilder] = None):
        """
        Initialize the converter.
        
        Args:
            graph_builder: KinematicGraphBuilder instance to use for graph construction
        """
        self.graph_builder = graph_builder or KinematicGraphBuilder(connectivity_type='opensim')
        
    def compute_kinematic_features(self, positions: np.ndarray, dt: float = 1/120) -> np.ndarray:
        """
        Compute kinematic features from position data.
        
        Args:
            positions: (n_frames, n_joints, 3) position data
            dt: time step between frames
            
        Returns:
            features: (n_frames, n_joints, n_features) kinematic features
        """
        n_frames, n_joints, _ = positions.shape
        
        # Initialize feature array: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z]
        features = np.zeros((n_frames, n_joints, 9))
        
        # Position features
        features[:, :, :3] = positions
        
        # Velocity features (central difference)
        if n_frames > 1:
            features[1:-1, :, 3:6] = (positions[2:] - positions[:-2]) / (2 * dt)
            features[0, :, 3:6] = (positions[1] - positions[0]) / dt
            features[-1, :, 3:6] = (positions[-1] - positions[-2]) / dt
        
        # Acceleration features (second derivative)
        if n_frames > 2:
            velocities = features[:, :, 3:6]
            features[1:-1, :, 6:9] = (velocities[2:] - velocities[:-2]) / (2 * dt)
            features[0, :, 6:9] = (velocities[1] - velocities[0]) / dt
            features[-1, :, 6:9] = (velocities[-1] - velocities[-2]) / dt
            
        return features
    
    def trc_to_pyg_data(self, trc_data: dict, frame_window: int = 10, 
                       include_temporal_edges: bool = True,
                       temporal_connections: int = 2) -> List[Data]:
        """
        Convert TRC data to list of PyG Data objects for time-series modeling.
        
        Args:
            trc_data: Dictionary from TRCParser with 'joint_names', 'positions', 'frame_rate'
            frame_window: Number of frames to include in each graph
            include_temporal_edges: Whether to include temporal connections
            temporal_connections: Number of temporal connections per node
            
        Returns:
            List of PyG Data objects
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            print("PyTorch Geometric not available - returning empty list")
            return []
        
        joint_names = trc_data['joint_names']
        positions = trc_data['positions']
        frame_rate = trc_data.get('frame_rate', 120)
        
        # Compute kinematic features
        features = self.compute_kinematic_features(positions, dt=1/frame_rate)
        
        # Create spatial edge index
        spatial_edge_index = self.graph_builder.build_edge_index(joint_names)
        
        # Create Data objects for sliding windows
        data_objects = []
        n_frames = len(positions)
        
        for start_idx in range(0, n_frames - frame_window + 1, frame_window // 2):
            end_idx = start_idx + frame_window
            
            # Extract window features
            window_features = features[start_idx:end_idx]  # (window, joints, features)
            
            if include_temporal_edges:
                # Create spatio-temporal graph
                num_joints = len(joint_names)
                
                # Flatten features: (window * joints, features)
                node_features = window_features.reshape(-1, window_features.shape[-1])
                
                # Create spatial edges for each time step
                spatial_edges_list = []
                for t in range(frame_window):
                    offset = t * num_joints
                    time_edges = spatial_edge_index + offset
                    spatial_edges_list.append(time_edges)
                
                # Create temporal edges
                temporal_edges = self.graph_builder.create_temporal_edges(
                    num_joints, frame_window, temporal_connections
                )
                
                # Combine spatial and temporal edges
                all_spatial_edges = torch.cat(spatial_edges_list, dim=1)
                edge_index = torch.cat([all_spatial_edges, temporal_edges], dim=1)
                
                # Create edge weights
                edge_weights = self.graph_builder.compute_edge_weights(
                    torch.tensor(node_features, dtype=torch.float),
                    edge_index,
                    weight_type='kinematic'
                )
                
            else:
                # Flatten temporal dimension into node features
                # Each joint gets features from all frames in window
                node_features = window_features.transpose(1, 0, 2).reshape(len(joint_names), -1)
                edge_index = spatial_edge_index
                edge_weights = None
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            
            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_weights,
                num_nodes=x.size(0),
                frame_start=start_idx,
                frame_end=end_idx,
                time_window=frame_window,
                is_temporal=include_temporal_edges
            )
            
            data_objects.append(data)
        
        return data_objects


def create_motion_graph(
    motion_data: pd.DataFrame,
    marker_names: List[str],
    time_window: Optional[Tuple[float, float]] = None,
    connectivity_type: str = 'opensim',
    include_temporal_edges: bool = False,
    temporal_connections: int = 1,
    feature_window: int = 1
) -> Data:
    """
    Create a motion graph from motion capture data with expert anatomical knowledge.
    
    Args:
        motion_data (pd.DataFrame): Motion capture data with columns [Frame, Time, marker_X, marker_Y, marker_Z, ...]
        marker_names (List[str]): List of marker names to include
        time_window (Tuple[float, float], optional): Time window to extract (start_time, end_time)
        connectivity_type (str): Type of graph connectivity ('opensim', 'skeletal', 'distance', 'custom')
        include_temporal_edges (bool): Whether to include temporal connections
        temporal_connections (int): Number of temporal connections per node
        feature_window (int): Number of time steps to include in node features
        
    Returns:
        Data: PyTorch Geometric Data object representing the motion graph
    """
    # Filter data by time window if specified
    if time_window is not None:
        start_time, end_time = time_window
        mask = (motion_data['Time'] >= start_time) & (motion_data['Time'] <= end_time)
        motion_data = motion_data[mask].copy()
    
    # Extract node features for each time step
    time_steps = []
    node_features_per_frame = []
    
    for idx, row in motion_data.iterrows():
        frame_features = []
        for marker in marker_names:
            x_col = f'{marker}_X'
            y_col = f'{marker}_Y'
            z_col = f'{marker}_Z'
            
            if all(col in motion_data.columns for col in [x_col, y_col, z_col]):
                x, y, z = row[x_col], row[y_col], row[z_col]
                frame_features.append([x, y, z])
                
        if frame_features:
            node_features_per_frame.append(frame_features)
            time_steps.append(row['Time'])
    
    if not node_features_per_frame:
        raise ValueError("No valid marker data found")
    
    # Convert to numpy array: [num_frames, num_markers, 3]
    positions = np.array(node_features_per_frame)
    
    # Compute kinematic features (position, velocity, acceleration)
    converter = MotionGraphConverter()
    dt = np.mean(np.diff(time_steps)) if len(time_steps) > 1 else 1/120
    kinematic_features = converter.compute_kinematic_features(positions, dt)
    
    # Create graph builder with expert knowledge
    graph_builder = KinematicGraphBuilder(connectivity_type=connectivity_type, 
                                        use_anatomical_knowledge=True)
    
    if include_temporal_edges:
        # Create spatio-temporal graph
        num_frames, num_markers = kinematic_features.shape[:2]
        
        # Flatten features: [num_frames * num_markers, feature_dim]
        node_features = kinematic_features.reshape(-1, kinematic_features.shape[-1])
        
        # Spatial edges (within each frame)
        spatial_edges = graph_builder.build_edge_index(marker_names)
        spatial_edge_list = []
        
        for frame in range(num_frames):
            frame_offset = frame * num_markers
            frame_edges = spatial_edges + frame_offset
            spatial_edge_list.append(frame_edges)
        
        # Temporal edges (across frames)
        temporal_edges = graph_builder.create_temporal_edges(
            num_markers, num_frames, temporal_connections
        )
        
        # Combine spatial and temporal edges
        all_spatial_edges = torch.cat(spatial_edge_list, dim=1)
        edge_index = torch.cat([all_spatial_edges, temporal_edges], dim=1)
        
        # Compute edge weights
        node_features_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_weights = graph_builder.compute_edge_weights(
            node_features_tensor, edge_index, weight_type='kinematic'
        )
        
    else:
        # Create spatial graph with windowed features
        if feature_window > 1:
            # Include multiple time steps in node features
            windowed_features = []
            for i in range(feature_window, num_frames):
                window = kinematic_features[i-feature_window+1:i+1]  # [window, markers, features]
                # Flatten window into node features: [markers, window*features]
                window_flat = window.transpose(1, 0, 2).reshape(num_markers, -1)
                windowed_features.append(window_flat)
            
            if windowed_features:
                # Use the last windowed features
                node_features = windowed_features[-1]
            else:
                # Fallback to single frame
                node_features = kinematic_features[-1]  # [markers, features]
        else:
            # Use single frame features
            node_features = kinematic_features[-1]  # [markers, features]
        
        # Create spatial edges
        edge_index = graph_builder.build_edge_index(marker_names)
        
        # Compute edge weights
        node_features_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_weights = graph_builder.compute_edge_weights(
            node_features_tensor, edge_index, weight_type='anatomical'
        )
    
    # Create Data object
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_weights,
        time_steps=torch.tensor(time_steps, dtype=torch.float),
        marker_names=marker_names,
        connectivity_type=connectivity_type
    )
    
    return data
