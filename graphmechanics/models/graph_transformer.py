"""
Graph Transformer for Biomechanical Motion Analysis

This module implements a Graph Transformer architecture using PyTorch Geometric
for analyzing motion capture data represented as graphs. The model is designed
to capture both spatial relationships between markers and temporal dynamics
in human movement.

Key Features:
- Multi-head attention mechanism for spatial-temporal analysis
- Positional encoding for temporal information
- Flexible architecture for various biomechanical tasks
- Support for variable-length sequences and missing markers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, List
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal information in motion sequences.
    
    Adds sinusoidal positional encodings to node features to help the model
    understand temporal relationships in motion data.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Dimension of the model
            max_len (int): Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, time_steps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input features.
        
        Args:
            x (torch.Tensor): Input features [num_nodes, d_model]
            time_steps (torch.Tensor, optional): Time step indices for each node
            
        Returns:
            torch.Tensor: Features with positional encoding added
        """
        if time_steps is not None:
            # Use provided time steps for indexing
            return x + self.pe[time_steps].squeeze(1)
        else:
            # Default sequential indexing
            return x + self.pe[:x.size(0)].squeeze(1)


class GraphTransformerLayer(nn.Module):
    """
    Single layer of the Graph Transformer.
    
    Combines TransformerConv (graph attention) with feedforward networks
    and residual connections for robust feature learning.
    """
    
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int, 
        num_heads: int = 8,
        dropout: float = 0.1,
        concat: bool = False
    ):
        """
        Initialize graph transformer layer.
        
        Args:
            in_dim (int): Input feature dimension
            hidden_dim (int): Hidden dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            concat (bool): Whether to concatenate attention heads
        """
        super().__init__()
        
        self.concat = concat
        out_dim = hidden_dim * num_heads if concat else hidden_dim
        
        # Graph attention layer
        self.attention = TransformerConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=concat,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
            nn.Dropout(dropout)
        )
        
        # Input projection if dimensions don't match
        self.input_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer layer.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, in_dim]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            
        Returns:
            torch.Tensor: Updated node features [num_nodes, out_dim]
        """
        # Project input if necessary
        residual = self.input_proj(x)
        
        # Self-attention with residual connection
        x = self.attention(x, edge_index)
        x = self.norm1(x + residual)
        
        # Feedforward with residual connection
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        
        return x


class GraphTransformer(nn.Module):
    """
    Graph Transformer for biomechanical motion analysis.
    
    This model processes motion capture data represented as graphs, where:
    - Nodes represent markers with X, Y, Z coordinates
    - Edges represent kinematic connections between markers
    - The transformer captures spatial-temporal relationships
    
    Applications:
    - Motion classification (e.g., activity recognition)
    - Movement prediction and forecasting
    - Anomaly detection in movement patterns
    - Feature extraction for biomechanical analysis
    """
    
    def __init__(
        self,
        node_features: int = 3,          # X, Y, Z coordinates
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        num_classes: int = 10,
        dropout: float = 0.1,
        pooling: str = 'mean',           # 'mean', 'max', 'attention'
        use_positional_encoding: bool = True,
        max_sequence_length: int = 1000
    ):
        """
        Initialize the Graph Transformer model.
        
        Args:
            node_features (int): Number of input node features (typically 3 for X,Y,Z)
            hidden_dim (int): Hidden dimension for transformer layers
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            num_classes (int): Number of output classes for classification
            dropout (float): Dropout probability
            pooling (str): Graph pooling method ('mean', 'max', 'attention')
            use_positional_encoding (bool): Whether to use positional encoding
            max_sequence_length (int): Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_positional_encoding = use_positional_encoding
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # Positional encoding for temporal information
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(hidden_dim, max_sequence_length)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                concat=False
            )
            for _ in range(num_layers)
        ])
        
        # Attention pooling layer (if using attention pooling)
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: Optional[torch.Tensor] = None,
        time_steps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Graph Transformer.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, node_features]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            batch (torch.Tensor, optional): Batch assignment for each node
            time_steps (torch.Tensor, optional): Time step for each node
            
        Returns:
            torch.Tensor: Output predictions [batch_size, num_classes]
        """
        # Project input features
        x = self.input_proj(x)
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = self.pos_encoding(x, time_steps)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, edge_index)
        
        # Graph pooling to get graph-level representations
        if batch is None:
            # Single graph case
            x = self._pool_single_graph(x)
        else:
            # Batch of graphs case
            x = self._pool_batch_graphs(x, batch)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def _pool_single_graph(self, x: torch.Tensor) -> torch.Tensor:
        """Pool features for a single graph."""
        if self.pooling == 'mean':
            return torch.mean(x, dim=0, keepdim=True)
        elif self.pooling == 'max':
            return torch.max(x, dim=0, keepdim=True)[0]
        elif self.pooling == 'attention':
            weights = torch.softmax(self.attention_pool(x), dim=0)
            return torch.sum(weights * x, dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    def _pool_batch_graphs(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Pool features for a batch of graphs."""
        if self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)
        elif self.pooling == 'attention':
            # Custom attention pooling for batched graphs
            weights = torch.softmax(self.attention_pool(x), dim=0)
            weighted_x = weights * x
            return global_mean_pool(weighted_x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    def get_node_embeddings(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get node-level embeddings without pooling.
        
        Useful for analyzing individual marker representations or
        for downstream tasks that require node-level features.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, node_features]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            time_steps (torch.Tensor, optional): Time step for each node
            
        Returns:
            torch.Tensor: Node embeddings [num_nodes, hidden_dim]
        """
        # Project input features
        x = self.input_proj(x)
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = self.pos_encoding(x, time_steps)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, edge_index)
        
        return x
    
    def predict(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: Optional[torch.Tensor] = None,
        time_steps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Make predictions with softmax applied.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge connectivity
            batch (torch.Tensor, optional): Batch assignment
            time_steps (torch.Tensor, optional): Time steps
            
        Returns:
            torch.Tensor: Predicted probabilities [batch_size, num_classes]
        """
        logits = self.forward(x, edge_index, batch, time_steps)
        return F.softmax(logits, dim=-1)
