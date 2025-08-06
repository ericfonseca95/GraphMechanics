"""
Motion classification training utilities for GraphMechanics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class MotionClassificationTask:
    """
    Training task for motion classification using GraphMechanics models.
    """
    
    def __init__(self, node_features: int, num_classes: int = 3, device: str = 'cpu'):
        """
        Initialize motion classification task.
        
        Args:
            node_features: Number of input node features
            num_classes: Number of output classes
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Import GraphTransformer
        try:
            from graphmechanics.models.graph_transformer import GraphTransformer
            self.model = GraphTransformer(
                node_features=node_features,
                hidden_dim=128,
                num_layers=4,
                num_heads=8,
                num_classes=num_classes
            ).to(device)
        except ImportError:
            # Fallback to a simple model if GraphTransformer not available
            from torch_geometric.nn import global_mean_pool
            
            class SimpleGraphTransformer(nn.Module):
                def __init__(self, node_features, hidden_dim, num_layers, num_heads, num_classes):
                    super().__init__()
                    self.embedding = nn.Linear(node_features, hidden_dim)
                    self.transformer_layers = nn.ModuleList([
                        nn.TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True)
                        for _ in range(num_layers)
                    ])
                    self.classifier = nn.Linear(hidden_dim, num_classes)
                    self.pool = global_mean_pool
                    
                def forward(self, x, edge_index, batch):
                    x = self.embedding(x)
                    x = self.pool(x, batch)
                    x = x.unsqueeze(1)
                    for layer in self.transformer_layers:
                        x = layer(x)
                    x = x.squeeze(1)
                    return self.classifier(x)
            
            self.model = SimpleGraphTransformer(
                node_features=node_features,
                hidden_dim=128,
                num_layers=4,
                num_heads=8,
                num_classes=num_classes
            ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
        
    def evaluate(self, dataloader):
        """Evaluate on test data."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                total += batch.y.size(0)
                correct += (pred == batch.y).sum().item()
        
        return correct / total
