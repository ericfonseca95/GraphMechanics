"""
Biomechanical validation utilities for GraphMechanics.

This module provides comprehensive validation tools for biomechanical motion analysis,
including constraint validation, loss functions, and performance analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import time
from pathlib import Path


class GraphMechanicsValidator:
    """
    Comprehensive validator for graph-based biomechanical motion analysis.
    
    Validates motion predictions against anatomical constraints, physical laws,
    and biomechanical plausibility metrics.
    """
    
    def __init__(self, constraints=None):
        """Initialize validator with biomechanical constraints."""
        if constraints is None:
            from ..data.graph_builder import BiomechanicalConstraints
            self.constraints = BiomechanicalConstraints()
        else:
            self.constraints = constraints
            
        self.validation_history = []
    
    def validate_motion_sequence(self, motion_data: np.ndarray, 
                                marker_names: Optional[List[str]] = None,
                                dt: float = 1/120) -> Dict[str, float]:
        """
        Validate a complete motion sequence.
        
        Args:
            motion_data: Shape (T, N, 3) where T=time, N=markers, 3=xyz
            marker_names: Names of motion capture markers
            dt: Time step between frames
            
        Returns:
            Dictionary of validation metrics
        """
        if motion_data.ndim != 3:
            raise ValueError("Motion data must have shape (T, N, 3)")
            
        T, N, _ = motion_data.shape
        results = {
            'total_frames': T,
            'valid_frames': 0,
            'constraint_violations': 0,
            'velocity_violations': 0,
            'acceleration_violations': 0,
            'temporal_consistency': 0.0,
            'physical_plausibility': 0.0,
            'overall_validity': 0.0
        }
        
        valid_poses = 0
        velocity_violations = 0
        acceleration_violations = 0
        
        # Validate each frame
        for t in range(T):
            pose = motion_data[t]
            is_valid, violations = self.constraints.validate_pose(pose, marker_names)
            
            if is_valid:
                valid_poses += 1
            else:
                results['constraint_violations'] += len(violations)
        
        # Validate velocities and accelerations
        if T > 1:
            velocities = np.diff(motion_data, axis=0) / dt
            max_vel = np.max(np.linalg.norm(velocities, axis=2))
            
            if max_vel > self.constraints.joint_limits.max_linear_velocity:
                velocity_violations += 1
        
        if T > 2:
            accelerations = np.diff(velocities, axis=0) / dt
            max_acc = np.max(np.linalg.norm(accelerations, axis=2))
            
            if max_acc > self.constraints.joint_limits.max_linear_acceleration:
                acceleration_violations += 1
        
        # Calculate metrics
        results['valid_frames'] = valid_poses
        results['velocity_violations'] = velocity_violations
        results['acceleration_violations'] = acceleration_violations
        results['temporal_consistency'] = valid_poses / T if T > 0 else 0.0
        results['physical_plausibility'] = 1.0 - (velocity_violations + acceleration_violations) / max(1, T-1)
        results['overall_validity'] = (results['temporal_consistency'] + results['physical_plausibility']) / 2.0
        
        self.validation_history.append(results)
        return results
    
    def get_validation_summary(self) -> Dict[str, float]:
        """Get summary statistics from validation history."""
        if not self.validation_history:
            return {}
            
        keys = self.validation_history[0].keys()
        summary = {}
        
        for key in keys:
            values = [result[key] for result in self.validation_history]
            if isinstance(values[0], (int, float)):
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_min'] = np.min(values)
                summary[f'{key}_max'] = np.max(values)
                
        return summary


class BiomechanicalLossFunctions:
    """
    Physics-informed loss functions for biomechanical motion prediction.
    
    Implements Scott Delp's biomechanical principles as differentiable loss functions
    that can be integrated into neural network training.
    """
    
    def __init__(self, device='cpu'):
        """Initialize biomechanical loss functions."""
        self.device = device
        from ..data.graph_builder import JointLimits
        self.joint_limits = JointLimits()
    
    def constraint_violation_loss(self, positions: torch.Tensor, 
                                 velocities: torch.Tensor = None) -> torch.Tensor:
        """
        Loss for constraint violations.
        
        Args:
            positions: Tensor of shape (batch, time, joints, 3)
            velocities: Optional velocity tensor
            
        Returns:
            Constraint violation loss
        """
        loss = torch.tensor(0.0, device=self.device)
        
        # Velocity constraint loss
        if velocities is not None:
            vel_magnitudes = torch.norm(velocities, dim=-1)
            max_vel = self.joint_limits.max_linear_velocity
            vel_violations = torch.clamp(vel_magnitudes - max_vel, min=0.0)
            loss += torch.mean(vel_violations ** 2)
        
        return loss
    
    def bone_length_consistency_loss(self, positions: torch.Tensor, 
                                   bone_connections: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Enforce bone length consistency across time.
        
        Args:
            positions: Joint positions tensor (batch, time, joints, 3)
            bone_connections: List of (joint1_idx, joint2_idx) connections
            
        Returns:
            Bone length consistency loss
        """
        loss = torch.tensor(0.0, device=self.device)
        
        for joint1_idx, joint2_idx in bone_connections:
            # Calculate bone vectors
            bone_vectors = positions[:, :, joint1_idx] - positions[:, :, joint2_idx]
            bone_lengths = torch.norm(bone_vectors, dim=-1)  # (batch, time)
            
            # Bone lengths should be consistent across time
            length_variance = torch.var(bone_lengths, dim=1)  # (batch,)
            loss += torch.mean(length_variance)
        
        return loss
    
    def ground_contact_loss(self, foot_positions: torch.Tensor, 
                           ground_level: float = 0.0) -> torch.Tensor:
        """
        Prevent foot penetration through ground.
        
        Args:
            foot_positions: Foot position tensor (batch, time, feet, 3)
            ground_level: Y-coordinate of ground plane
            
        Returns:
            Ground contact loss
        """
        foot_y = foot_positions[:, :, :, 1]  # Y coordinates
        penetration = torch.clamp(ground_level - foot_y, min=0.0)
        return torch.mean(penetration ** 2)
    
    def smoothness_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Promote smooth motion trajectories.
        
        Args:
            positions: Position tensor (batch, time, joints, 3)
            
        Returns:
            Smoothness loss based on acceleration magnitude
        """
        if positions.shape[1] < 3:
            return torch.tensor(0.0, device=self.device)
        
        # Calculate second derivatives (acceleration)
        velocities = positions[:, 1:] - positions[:, :-1]
        accelerations = velocities[:, 1:] - velocities[:, :-1]
        
        # Minimize acceleration magnitude for smoothness
        acc_magnitude = torch.norm(accelerations, dim=-1)
        return torch.mean(acc_magnitude)
    
    def energy_conservation_loss(self, positions: torch.Tensor, 
                                masses: torch.Tensor,
                                dt: float = 1/120) -> torch.Tensor:
        """
        Approximate energy conservation loss.
        
        Args:
            positions: Position tensor (batch, time, joints, 3)
            masses: Mass tensor for each joint (joints,)
            dt: Time step
            
        Returns:
            Energy conservation loss
        """
        if positions.shape[1] < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Calculate kinetic energy
        velocities = (positions[:, 1:] - positions[:, :-1]) / dt
        kinetic_energy = 0.5 * masses[None, None, :, None] * (velocities ** 2)
        kinetic_energy = torch.sum(kinetic_energy, dim=(2, 3))  # (batch, time-1)
        
        # Calculate potential energy (gravitational)
        g = 9.81  # m/s²
        heights = positions[:, :, :, 1]  # Y coordinates
        potential_energy = masses[None, None, :] * g * heights
        potential_energy = torch.sum(potential_energy, dim=2)  # (batch, time)
        
        # Total energy should be approximately constant
        total_energy = kinetic_energy + potential_energy[:, 1:]
        energy_variance = torch.var(total_energy, dim=1)
        
        return torch.mean(energy_variance)
    
    def combined_biomechanical_loss(self, positions: torch.Tensor,
                                  bone_connections: List[Tuple[int, int]],
                                  foot_indices: List[int],
                                  masses: torch.Tensor,
                                  weights: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
        """
        Combine all biomechanical losses with optional weighting.
        
        Args:
            positions: Position tensor (batch, time, joints, 3)
            bone_connections: Bone connectivity
            foot_indices: Indices of foot joints
            masses: Joint masses
            weights: Loss component weights
            
        Returns:
            Dictionary of individual and combined losses
        """
        if weights is None:
            weights = {
                'constraint': 1.0,
                'bone_length': 0.5,
                'ground_contact': 2.0,
                'smoothness': 0.3,
                'energy': 0.1
            }
        
        losses = {}
        
        # Individual loss components
        velocities = positions[:, 1:] - positions[:, :-1] if positions.shape[1] > 1 else None
        losses['constraint'] = self.constraint_violation_loss(positions, velocities)
        losses['bone_length'] = self.bone_length_consistency_loss(positions, bone_connections)
        losses['smoothness'] = self.smoothness_loss(positions)
        losses['energy'] = self.energy_conservation_loss(positions, masses)
        
        # Ground contact loss for feet
        if foot_indices:
            foot_positions = positions[:, :, foot_indices, :]
            losses['ground_contact'] = self.ground_contact_loss(foot_positions)
        else:
            losses['ground_contact'] = torch.tensor(0.0, device=self.device)
        
        # Combined weighted loss
        losses['total'] = sum(weights.get(key, 1.0) * loss 
                            for key, loss in losses.items() 
                            if key != 'total')
        
        return losses


class GraphMechanicsPerformanceAnalyzer:
    """
    Comprehensive performance analysis for GraphMechanics models.
    
    Provides benchmarking, profiling, and comparative analysis tools
    for evaluating graph-based biomechanical models.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.benchmark_results = {}
        self.timing_results = {}
    
    def benchmark_method(self, method_name: str, method_func, *args, **kwargs) -> Dict[str, float]:
        """
        Benchmark a specific method.
        
        Args:
            method_name: Name of the method being benchmarked
            method_func: Function to benchmark
            *args, **kwargs: Arguments for the function
            
        Returns:
            Timing and performance metrics
        """
        times = []
        memory_usage = []
        
        # Warm-up run
        try:
            _ = method_func(*args, **kwargs)
        except Exception:
            pass
        
        # Benchmark runs
        n_runs = 10
        for _ in range(n_runs):
            start_time = time.time()
            
            try:
                result = method_func(*args, **kwargs)
                end_time = time.time()
                times.append(end_time - start_time)
                
                # Estimate memory usage (simplified)
                if torch.is_tensor(result):
                    memory_usage.append(result.numel() * result.element_size())
                elif isinstance(result, (list, tuple)):
                    memory_usage.append(len(result) * 8)  # Rough estimate
                else:
                    memory_usage.append(0)
                    
            except Exception as e:
                print(f"Benchmark failed for {method_name}: {e}")
                times.append(np.inf)
                memory_usage.append(0)
        
        metrics = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'mean_memory': np.mean(memory_usage),
            'total_runs': n_runs
        }
        
        self.benchmark_results[method_name] = metrics
        return metrics
    
    def comparative_analysis(self, methods: Dict[str, callable], 
                           test_data, **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple methods on the same test data.
        
        Args:
            methods: Dictionary of method_name -> function
            test_data: Data to test all methods on
            **kwargs: Additional arguments for methods
            
        Returns:
            Comparative benchmark results
        """
        results = {}
        
        for method_name, method_func in methods.items():
            print(f"Benchmarking {method_name}...")
            results[method_name] = self.benchmark_method(
                method_name, method_func, test_data, **kwargs
            )
        
        return results
    
    def generate_performance_report(self, save_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Performance report as string
        """
        if not self.benchmark_results:
            return "No benchmark results available."
        
        report_lines = [
            "# GraphMechanics Performance Analysis Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Benchmark Results",
            ""
        ]
        
        for method_name, metrics in self.benchmark_results.items():
            report_lines.extend([
                f"### {method_name}",
                f"- Mean execution time: {metrics['mean_time']:.4f}s",
                f"- Standard deviation: {metrics['std_time']:.4f}s",
                f"- Min time: {metrics['min_time']:.4f}s",
                f"- Max time: {metrics['max_time']:.4f}s",
                f"- Average memory usage: {metrics['mean_memory']:.2f} bytes",
                ""
            ])
        
        report = "\n".join(report_lines)
        
        if save_path:
            save_path.write_text(report)
            print(f"Performance report saved to: {save_path}")
        
        return report
    
    def get_best_performing_method(self) -> Tuple[str, Dict[str, float]]:
        """
        Identify the best performing method based on mean execution time.
        
        Returns:
            Tuple of (method_name, metrics)
        """
        if not self.benchmark_results:
            return None, {}
        
        best_method = min(self.benchmark_results.items(), 
                         key=lambda x: x[1]['mean_time'])
        return best_method


class BiomechanicalValidator:
    """
    Final validation system for biomechanical analysis results.
    
    Provides comprehensive validation of analysis results against
    clinical and research standards.
    """
    
    def __init__(self):
        """Initialize biomechanical validator."""
        self.validation_criteria = {
            'temporal_consistency': 0.8,    # 80% of frames should be valid
            'velocity_limit': 10.0,         # m/s maximum velocity
            'acceleration_limit': 50.0,     # m/s² maximum acceleration
            'bone_length_variance': 0.05,   # 5% maximum variance in bone length
            'ground_penetration': 0.01      # 1cm maximum ground penetration
        }
    
    def validate_analysis_results(self, analysis_results: Dict) -> Dict[str, Union[bool, float, str]]:
        """
        Validate complete analysis results.
        
        Args:
            analysis_results: Dictionary containing analysis outputs
            
        Returns:
            Validation report with pass/fail status
        """
        validation_report = {
            'overall_pass': True,
            'detailed_results': {},
            'recommendations': []
        }
        
        # Check temporal consistency
        if 'temporal_consistency' in analysis_results:
            tc = analysis_results['temporal_consistency']
            passes = tc >= self.validation_criteria['temporal_consistency']
            validation_report['detailed_results']['temporal_consistency'] = {
                'value': tc,
                'threshold': self.validation_criteria['temporal_consistency'],
                'passes': passes
            }
            if not passes:
                validation_report['overall_pass'] = False
                validation_report['recommendations'].append(
                    "Improve temporal consistency - consider smoothing or constraint enforcement"
                )
        
        # Check velocity limits
        if 'max_velocity' in analysis_results:
            max_vel = analysis_results['max_velocity']
            passes = max_vel <= self.validation_criteria['velocity_limit']
            validation_report['detailed_results']['velocity_limit'] = {
                'value': max_vel,
                'threshold': self.validation_criteria['velocity_limit'],
                'passes': passes
            }
            if not passes:
                validation_report['overall_pass'] = False
                validation_report['recommendations'].append(
                    "Velocity exceeds physiological limits - check input data or model constraints"
                )
        
        return validation_report
    
    def clinical_validation_score(self, motion_data: np.ndarray) -> float:
        """
        Calculate clinical validation score (0-1).
        
        Args:
            motion_data: Motion capture data
            
        Returns:
            Clinical validation score
        """
        scores = []
        
        # Basic motion quality checks
        if motion_data.size > 0:
            # Check for reasonable position ranges
            pos_range = np.ptp(motion_data, axis=0)  # Range across time
            reasonable_range = np.all(pos_range < 5.0)  # 5m maximum range
            scores.append(float(reasonable_range))
            
            # Check for smoothness
            if motion_data.shape[0] > 2:
                velocities = np.diff(motion_data, axis=0)
                accelerations = np.diff(velocities, axis=0)
                smooth_motion = np.mean(np.abs(accelerations)) < 20.0  # Reasonable acceleration
                scores.append(float(smooth_motion))
        
        return np.mean(scores) if scores else 0.0
