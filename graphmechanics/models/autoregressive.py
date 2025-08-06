"""
Autoregressive Graph Transformer for Motion Capture Data.

This module implements an autoregressive graph transformer that can predict
future joint coordinates based on past motion sequences. The model learns
temporal dependencies in human motion while respecting anatomical constraints
through graph structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional, List, Tuple, Dict
import math

__all__ = ['AutoregressiveGraphTransformer', 'MotionPredictor', 'CausalGraphAttention']


class CausalGraphAttention(nn.Module):
    """
    Causal attention mechanism for graph transformers that respects temporal ordering.
    """
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        assert out_channels % heads == 0
        self.head_dim = out_channels // heads
        
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        self.out_proj = nn.Linear(out_channels, out_channels)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with causal masking for autoregressive generation.
        
        Args:
            x: Node features [batch_size * seq_len * num_nodes, features]
            edge_index: Graph connectivity [2, num_edges]
            temporal_mask: Causal mask for temporal dependencies
            
        Returns:
            Updated node features
        """
        batch_size, seq_len, num_nodes = x.shape[0] // x.shape[1], x.shape[1], x.shape[2]
        
        # Project to Q, K, V
        q = self.q_proj(x).view(-1, self.heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.einsum('nhd,mhd->nhm', q, k) / math.sqrt(self.head_dim)
        
        # Apply causal mask if provided
        if temporal_mask is not None:
            scores = scores.masked_fill(temporal_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        out = torch.einsum('nhm,mhd->nhd', attn_weights, v)
        out = out.view(-1, self.out_channels)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class AutoregressiveGraphTransformer(nn.Module):
    """
    Autoregressive Graph Transformer for motion sequence modeling.
    
    This model can generate future motion by conditioning on past observations
    while respecting anatomical constraints through graph structure.
    """
    
    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        output_dim: int = 3,  # x, y, z coordinates
        max_seq_length: int = 100,
        dropout: float = 0.1,
        use_positional_encoding: bool = True
    ):
        """
        Initialize the autoregressive graph transformer.
        
        Args:
            node_features: Input feature dimension per node
            hidden_dim: Hidden dimension for transformer layers
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            output_dim: Output dimension (3 for x,y,z coordinates)
            max_seq_length: Maximum sequence length for positional encoding
            dropout: Dropout probability
            use_positional_encoding: Whether to use positional encoding
        """
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # Positional encoding for temporal information
        if use_positional_encoding:
            self.register_buffer('positional_encoding', 
                               self._create_positional_encoding(max_seq_length, hidden_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                edge_dim=None,
                beta=True
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len: int, num_nodes: int) -> torch.Tensor:
        """
        Create causal mask for autoregressive generation.
        
        Args:
            seq_len: Sequence length
            num_nodes: Number of nodes per time step
            
        Returns:
            Causal mask tensor
        """
        # Create mask that prevents attention to future time steps
        mask = torch.tril(torch.ones(seq_len, seq_len))
        
        # Expand for multiple nodes
        mask = mask.repeat_interleave(num_nodes, dim=0)
        mask = mask.repeat_interleave(num_nodes, dim=1)
        
        return mask
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        sequence_info: Optional[Dict[str, int]] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for training or inference.
        
        Args:
            x: Node features [total_nodes, features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch tensor for graph batching
            sequence_info: Dict with 'batch_size', 'seq_len', 'num_nodes'
            return_attention: Whether to return attention weights
            
        Returns:
            Predicted coordinates [total_nodes, output_dim]
        """
        total_nodes = x.shape[0]
        
        # Determine sequence structure
        if sequence_info is not None:
            batch_size = sequence_info['batch_size']
            seq_len = sequence_info['seq_len']
            num_nodes = sequence_info['num_nodes']
        else:
            # Infer from batch tensor or assume single sequence
            if batch is not None:
                batch_size = batch.max().item() + 1
                nodes_per_batch = total_nodes // batch_size
                # Assume temporal structure: nodes are ordered as [t0_n0, t0_n1, ..., t1_n0, t1_n1, ...]
                num_nodes = int(torch.sqrt(torch.tensor(nodes_per_batch, dtype=torch.float)).item())
                seq_len = nodes_per_batch // num_nodes
            else:
                # Single sequence case
                batch_size = 1
                # Need to infer seq_len and num_nodes - this is problematic without metadata
                # For now, assume square structure
                sqrt_total = int(torch.sqrt(torch.tensor(total_nodes, dtype=torch.float)).item())
                if sqrt_total * sqrt_total == total_nodes:
                    seq_len = num_nodes = sqrt_total
                else:
                    raise ValueError(
                        "Cannot infer temporal structure from input shape. "
                        "Please provide sequence_info with batch_size, seq_len, num_nodes."
                    )
        
        # Validate dimensions
        expected_total = batch_size * seq_len * num_nodes
        if total_nodes != expected_total:
            raise ValueError(
                f"Input shape mismatch: expected {expected_total} nodes "
                f"({batch_size} batches × {seq_len} timesteps × {num_nodes} nodes), "
                f"got {total_nodes}"
            )
        
        # Input projection
        h = self.input_proj(x)  # [total_nodes, hidden_dim]
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            # Reshape to separate temporal dimension
            h_temporal = h.view(batch_size, seq_len, num_nodes, self.hidden_dim)
            
            # Add positional encoding for each time step
            for t in range(seq_len):
                if t < self.max_seq_length:
                    pos_encoding = self.positional_encoding[t].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
                    h_temporal[:, t, :, :] += pos_encoding
            
            # Flatten back to graph format
            h = h_temporal.view(total_nodes, self.hidden_dim)
        
        # Apply transformer layers with proper temporal masking
        attention_weights = []
        
        for i, (transformer, layer_norm) in enumerate(zip(self.transformer_layers, self.layer_norms)):
            # Store residual for skip connection
            residual = h
            
            # Apply transformer layer
            h_new = transformer(h, edge_index)
            
            # Layer norm with residual connection
            h = layer_norm(residual + h_new)
            
            if return_attention:
                attention_weights.append(h_new)
        
        # Output projection
        output = self.output_proj(h)  # [total_nodes, output_dim]
        
        if return_attention:
            return output, attention_weights
        
        return output
    
    def generate(
        self,
        initial_sequence: torch.Tensor,
        edge_index: torch.Tensor,
        num_steps: int,
        temperature: float = 1.0,
        biomechanical_constraints: Optional[object] = None,
        validate_motion: bool = True
    ) -> torch.Tensor:
        """
        Generate future motion sequence autoregressively with biomechanical validation.
        
        Args:
            initial_sequence: Initial sequence [seq_len, num_nodes, features]
            edge_index: Graph connectivity [2, num_edges]
            num_steps: Number of future steps to generate
            temperature: Sampling temperature for stochastic generation
            biomechanical_constraints: BiomechanicalConstraints instance for validation
            validate_motion: Whether to validate generated motion
            
        Returns:
            Generated sequence [seq_len + num_steps, num_nodes, output_dim]
        """
        self.eval()
        
        seq_len, num_nodes, input_features = initial_sequence.shape
        device = initial_sequence.device
        
        # Validate input dimensions
        if input_features < self.output_dim:
            raise ValueError(f"Input features {input_features} < output dim {self.output_dim}")
        
        # Initialize output sequence with proper coordinate extraction
        generated = torch.zeros(seq_len + num_steps, num_nodes, self.output_dim, 
                              device=device, dtype=initial_sequence.dtype)
        
        # Extract initial coordinates (assume first 3 features are x,y,z)
        generated[:seq_len] = initial_sequence[:, :, :self.output_dim]
        
        # Track generation quality for early stopping
        generation_quality = []
        
        with torch.no_grad():
            for step in range(num_steps):
                try:
                    # Prepare current sequence for model input
                    current_len = seq_len + step
                    current_coords = generated[:current_len]  # [current_len, num_nodes, 3]
                    
                    # Create feature vector for model input
                    if self.node_features > self.output_dim:
                        # Option 1: Pad with velocities and accelerations if available
                        if input_features >= 9:  # pos + vel + acc
                            # Compute velocities and accelerations from generated coordinates
                            velocities = torch.zeros(current_len, num_nodes, 3, device=device)
                            accelerations = torch.zeros(current_len, num_nodes, 3, device=device)
                            
                            if current_len > 1:
                                velocities[1:] = current_coords[1:] - current_coords[:-1]
                            if current_len > 2:
                                accelerations[2:] = velocities[2:] - velocities[1:-1]
                            
                            input_features_expanded = torch.cat([current_coords, velocities, accelerations], dim=-1)
                        else:
                            # Option 2: Zero-pad to match expected input size
                            padding_size = self.node_features - self.output_dim
                            padding = torch.zeros(current_len, num_nodes, padding_size, device=device)
                            input_features_expanded = torch.cat([current_coords, padding], dim=-1)
                    else:
                        input_features_expanded = current_coords
                    
                    # Flatten for model input: [total_nodes, features]
                    input_flat = input_features_expanded.view(-1, self.node_features)
                    
                    # Prepare sequence info for proper temporal handling
                    sequence_info = {
                        'batch_size': 1,
                        'seq_len': current_len,
                        'num_nodes': num_nodes
                    }
                    
                    # Forward pass
                    output = self.forward(input_flat, edge_index, sequence_info=sequence_info)
                    
                    # Reshape output and extract next step prediction
                    output_reshaped = output.view(current_len, num_nodes, self.output_dim)
                    next_step_pred = output_reshaped[-1]  # [num_nodes, output_dim]
                    
                    # Apply temperature scaling for stochastic generation
                    if temperature != 1.0 and temperature > 0:
                        next_step_pred = next_step_pred / temperature
                    
                    # Validate biomechanical constraints if provided
                    if validate_motion and biomechanical_constraints is not None:
                        # Create temporary extended sequence for validation
                        temp_sequence = torch.cat([current_coords, next_step_pred.unsqueeze(0)], dim=0)
                        
                        # Validate the proposed next step
                        is_valid, violations = biomechanical_constraints.validate_pose(
                            temp_sequence[-1].cpu().numpy(),
                            previous_pose=temp_sequence[-2].cpu().numpy() if current_len > 0 else None
                        )
                        
                        if not is_valid:
                            # Apply constraint correction
                            corrected_pose = biomechanical_constraints.apply_constraints(
                                next_step_pred.cpu().numpy(),
                                current_coords[-1].cpu().numpy() if current_len > 0 else None
                            )
                            next_step_pred = torch.from_numpy(corrected_pose).to(device)
                            
                            # Track constraint violations for quality assessment
                            generation_quality.append({'step': step, 'violations': violations})
                    
                    # Add prediction to generated sequence
                    generated[seq_len + step] = next_step_pred
                    
                except Exception as e:
                    print(f"Error during generation at step {step}: {e}")
                    # Use last valid position or interpolation as fallback
                    if step > 0:
                        generated[seq_len + step] = generated[seq_len + step - 1]
                    else:
                        generated[seq_len + step] = generated[seq_len - 1]
        
        # Log generation quality if constraints were used
        if generation_quality and validate_motion:
            violation_rate = len(generation_quality) / num_steps
            if violation_rate > 0.3:  # More than 30% violations
                print(f"Warning: High constraint violation rate ({violation_rate:.2%}) during generation")
        
        return generated


class MotionPredictor(nn.Module):
    """
    High-level interface for motion prediction tasks.
    
    This class wraps the AutoregressiveGraphTransformer and provides
    convenient methods for training and prediction.
    """
    
    def __init__(
        self,
        graph_builder,
        node_features: int = 9,  # pos + vel + acc
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        prediction_horizon: int = 10,
        **kwargs
    ):
        """
        Initialize the motion predictor.
        
        Args:
            graph_builder: KinematicGraphBuilder instance
            node_features: Input feature dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            prediction_horizon: Number of future steps to predict
        """
        super().__init__()
        
        self.graph_builder = graph_builder
        self.prediction_horizon = prediction_horizon
        
        # Autoregressive model
        self.model = AutoregressiveGraphTransformer(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=3,  # x, y, z coordinates
            **kwargs
        )
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(data.x, data.edge_index, data.batch)
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        loss_type: str = 'combined',
        biomechanical_constraints: Optional[object] = None,
        constraint_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute prediction loss with optional biomechanical constraints.
        
        Args:
            predictions: Model predictions [total_nodes, 3]
            targets: Ground truth targets [total_nodes, 3]
            loss_type: Type of loss ('mse', 'l1', 'combined')
            biomechanical_constraints: BiomechanicalConstraints instance
            constraint_weight: Weight for constraint loss term
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Primary reconstruction loss
        if loss_type == 'mse':
            losses['reconstruction'] = self.mse_loss(predictions, targets)
        elif loss_type == 'l1':
            losses['reconstruction'] = self.l1_loss(predictions, targets)
        elif loss_type == 'combined':
            losses['reconstruction'] = (self.mse_loss(predictions, targets) + 
                                     0.1 * self.l1_loss(predictions, targets))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Biomechanical constraint loss
        if biomechanical_constraints is not None:
            try:
                # Reshape predictions for constraint computation
                batch_size = predictions.shape[0] // biomechanical_constraints.num_nodes if hasattr(biomechanical_constraints, 'num_nodes') else 1
                seq_len = predictions.shape[0] // batch_size
                
                pred_reshaped = predictions.view(batch_size, seq_len, -1, 3)
                target_reshaped = targets.view(batch_size, seq_len, -1, 3)
                
                constraint_loss = 0.0
                for b in range(batch_size):
                    for t in range(seq_len):
                        # Compute biomechanical loss for this frame
                        frame_loss = biomechanical_constraints.compute_biomechanical_loss(
                            pred_reshaped[b, t].cpu().numpy(),
                            target_reshaped[b, t].cpu().numpy() if t > 0 else None
                        )
                        constraint_loss += frame_loss
                
                losses['biomechanical'] = torch.tensor(constraint_loss / (batch_size * seq_len), 
                                                     device=predictions.device, requires_grad=True)
            except Exception as e:
                print(f"Warning: Could not compute biomechanical loss: {e}")
                losses['biomechanical'] = torch.tensor(0.0, device=predictions.device)
        else:
            losses['biomechanical'] = torch.tensor(0.0, device=predictions.device)
        
        # Total loss
        losses['total'] = losses['reconstruction'] + constraint_weight * losses['biomechanical']
        
        return losses
    
    def predict_sequence(
        self,
        initial_motion: torch.Tensor,
        marker_names: List[str],
        num_steps: int,
        return_confidence: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Predict future motion sequence.
        
        Args:
            initial_motion: Initial motion sequence [seq_len, num_markers, features]
            marker_names: List of marker names
            num_steps: Number of future steps to predict
            return_confidence: Whether to return prediction confidence
            
        Returns:
            Dictionary with predictions and optionally confidence scores
        """
        self.eval()
        
        # Create graph structure
        edge_index = self.graph_builder.build_edge_index(marker_names)
        
        # Generate predictions
        with torch.no_grad():
            predictions = self.model.generate(
                initial_sequence=initial_motion,
                edge_index=edge_index,
                num_steps=num_steps
            )
        
        results = {
            'predictions': predictions,
            'initial_sequence': initial_motion,
            'marker_names': marker_names
        }
        
        if return_confidence:
            # Compute prediction confidence (simplified)
            # In practice, this could be based on model uncertainty, attention weights, etc.
            variance = torch.var(predictions, dim=0)
            confidence = 1.0 / (1.0 + variance)
            results['confidence'] = confidence
        
        return results
    
    def evaluate_prediction_accuracy(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate prediction accuracy with multiple metrics.
        
        Args:
            predictions: Predicted motion [seq_len, num_markers, 3]
            ground_truth: Ground truth motion [seq_len, num_markers, 3]
            
        Returns:
            Dictionary of evaluation metrics
        """
        with torch.no_grad():
            # Mean squared error
            mse = F.mse_loss(predictions, ground_truth).item()
            
            # Mean absolute error
            mae = F.l1_loss(predictions, ground_truth).item()
            
            # Root mean squared error
            rmse = torch.sqrt(F.mse_loss(predictions, ground_truth)).item()
            
            # Per-joint errors
            joint_errors = torch.norm(predictions - ground_truth, dim=-1).mean(dim=0)
            
            # Velocity error (if sequences are long enough)
            if predictions.shape[0] > 1:
                pred_vel = predictions[1:] - predictions[:-1]
                true_vel = ground_truth[1:] - ground_truth[:-1]
                vel_error = F.l1_loss(pred_vel, true_vel).item()
            else:
                vel_error = 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'velocity_error': vel_error,
            'joint_errors': joint_errors.cpu().numpy()
        }


def create_autoregressive_model(
    marker_names: List[str],
    graph_builder,
    sequence_length: int = 20,
    prediction_horizon: int = 10,
    **model_kwargs
) -> MotionPredictor:
    """
    Factory function to create an autoregressive motion prediction model.
    
    Args:
        marker_names: List of marker names
        graph_builder: KinematicGraphBuilder instance
        sequence_length: Input sequence length
        prediction_horizon: Number of future steps to predict
        **model_kwargs: Additional model parameters
        
    Returns:
        Configured MotionPredictor instance
    """
    model = MotionPredictor(
        graph_builder=graph_builder,
        prediction_horizon=prediction_horizon,
        max_seq_length=sequence_length,
        **model_kwargs
    )
    
    return model
