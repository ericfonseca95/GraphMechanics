"""
Advanced training system for GraphMechanics with proper data handling.

This module implements a comprehensive training system that addresses the critical 
issues identified in the original implementation:
1. Prevents data leakage through proper train/val/test splits
2. Integrates biomechanical constraints during training
3. Implements proper evaluation metrics for motion prediction
4. Provides robust model checkpointing and resumption

Author: Scott Delp & Yannic Kilcher inspired implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

from ..models.autoregressive import MotionPredictor, AutoregressiveGraphTransformer
from .data_validation import create_proper_dataset_splits, MotionDataValidator
from ..data.graph_builder import KinematicGraphBuilder


class MotionPredictionTrainer:
    """
    Advanced trainer for motion prediction models with biomechanical constraints.
    
    Features:
    - Proper data splitting to prevent leakage
    - Biomechanical constraint integration
    - Comprehensive evaluation metrics
    - Model checkpointing and resumption
    - Training visualization and logging
    """
    
    def __init__(self,
                 model: MotionPredictor,
                 graph_builder: KinematicGraphBuilder,
                 experiment_name: str,
                 experiment_dir: Union[str, Path] = "./experiments",
                 device: str = "auto"):
        """
        Initialize the trainer.
        
        Args:
            model: MotionPredictor model to train
            graph_builder: KinematicGraphBuilder for data processing
            experiment_name: Name for this experiment
            experiment_dir: Directory to save experiments
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model = model
        self.graph_builder = graph_builder
        self.experiment_name = experiment_name
        
        # Setup experiment directory
        self.experiment_dir = Path(experiment_dir) / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'constraint_violations': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Data validation
        self.data_validator = MotionDataValidator()
        
        self.logger.info(f"Initialized trainer for experiment: {experiment_name}")
        self.logger.info(f"Using device: {self.device}")
    
    def _setup_logging(self):
        """Setup logging for the experiment."""
        log_file = self.experiment_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"GraphMechanics.{self.experiment_name}")
    
    def prepare_data(self,
                    dataset_paths: List[Union[str, Path]],
                    sequence_length: int = 50,
                    prediction_horizon: int = 10,
                    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                    batch_size: int = 32,
                    validation_split_strategy: str = 'file_level',
                    validate_quality: bool = True) -> Dict[str, Any]:
        """
        Prepare data with proper splitting and validation.
        
        Args:
            dataset_paths: List of paths to TRC files
            sequence_length: Length of input sequences
            prediction_horizon: Number of future steps to predict
            split_ratios: Train/val/test split ratios
            batch_size: Batch size for training
            validation_split_strategy: Strategy for data splitting
            validate_quality: Whether to validate data quality
            
        Returns:
            Dictionary with data loaders and split information
        """
        self.logger.info("Preparing data with proper splitting...")
        
        # Create proper splits
        split_info = create_proper_dataset_splits(
            dataset_paths,
            sequence_length=sequence_length,
            split_ratios=split_ratios,
            validate_quality=validate_quality
        )
        
        if split_info['split_quality']['data_leakage_check'] == 'failed':
            raise ValueError("Data leakage detected! Cannot proceed with training.")
        
        self.logger.info(f"Data quality: {split_info['dataset_quality']['overall_quality']}")
        self.logger.info(f"Split quality: {split_info['split_quality']['overall_quality']}")
        
        # Create datasets for each split
        data_loaders = {}
        split_stats = {}
        
        for split_name, indices in [
            ('train', split_info['data_split'].train_indices),
            ('val', split_info['data_split'].val_indices),
            ('test', split_info['data_split'].test_indices)
        ]:
            # Get file paths for this split
            split_paths = [dataset_paths[i] for i in indices]
            
            # Create dataset
            split_dataset = self._create_motion_dataset(
                split_paths, sequence_length, prediction_horizon
            )
            
            # Create data loader
            data_loaders[split_name] = DataLoader(
                split_dataset,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),
                collate_fn=self._collate_motion_graphs,
                num_workers=0  # Set to 0 to avoid multiprocessing issues
            )
            
            split_stats[split_name] = {
                'num_files': len(split_paths),
                'num_sequences': len(split_dataset),
                'files': [str(p) for p in split_paths]
            }
        
        # Save split information
        split_metadata = {
            'experiment_name': self.experiment_name,
            'split_ratios': split_ratios,
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'batch_size': batch_size,
            'split_stats': split_stats,
            'data_quality': split_info['dataset_quality'],
            'split_quality': split_info['split_quality']
        }
        
        split_file = self.experiment_dir / "data_splits.json"
        with open(split_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(split_metadata, f, indent=2, default=convert_numpy)
        
        self.logger.info(f"Data preparation complete. Split metadata saved to {split_file}")
        
        return {
            'data_loaders': data_loaders,
            'split_info': split_info,
            'split_metadata': split_metadata
        }
    
    def _create_motion_dataset(self, 
                              file_paths: List[Path], 
                              sequence_length: int, 
                              prediction_horizon: int) -> List[Data]:
        """Create a dataset of motion sequences."""
        dataset = []
        
        for file_path in file_paths:
            try:
                # Load and process motion data
                graph_data = self.graph_builder.create_temporal_graph_sequence(
                    str(file_path), 
                    sequence_length=sequence_length + prediction_horizon,
                    stride=sequence_length // 2  # 50% overlap between sequences
                )
                
                # Split each sequence into input and target
                for graph_sequence in graph_data:
                    if len(graph_sequence) >= sequence_length + prediction_horizon:
                        input_sequence = graph_sequence[:sequence_length]
                        target_sequence = graph_sequence[sequence_length:sequence_length + prediction_horizon]
                        
                        # Create combined graph data
                        combined_data = self._combine_sequence_data(input_sequence, target_sequence)
                        dataset.append(combined_data)
                        
            except Exception as e:
                self.logger.warning(f"Failed to process file {file_path}: {e}")
                continue
        
        return dataset
    
    def _combine_sequence_data(self, input_sequence: List[Data], target_sequence: List[Data]) -> Data:
        """Combine input and target sequences into a single Data object."""
        # Combine all input graphs
        input_batch = Batch.from_data_list(input_sequence)
        target_batch = Batch.from_data_list(target_sequence)
        
        # Create combined data object
        combined_data = Data(
            x=input_batch.x,
            edge_index=input_batch.edge_index,
            batch=input_batch.batch,
            y=target_batch.x,  # Target coordinates
            input_seq_len=len(input_sequence),
            target_seq_len=len(target_sequence),
            num_nodes_per_graph=input_sequence[0].x.shape[0]
        )
        
        return combined_data
    
    def _collate_motion_graphs(self, batch: List[Data]) -> Data:
        """Custom collate function for motion graph data."""
        return Batch.from_data_list(batch)
    
    def train(self,
              data_loaders: Dict[str, DataLoader],
              num_epochs: int = 100,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-5,
              scheduler_type: str = 'cosine',
              early_stopping_patience: int = 15,
              constraint_weight: float = 0.1,
              gradient_clip_norm: float = 1.0,
              save_every_n_epochs: int = 10) -> Dict[str, List[float]]:
        """
        Train the model with comprehensive monitoring.
        
        Args:
            data_loaders: Dictionary with train/val/test data loaders
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            scheduler_type: Learning rate scheduler type
            early_stopping_patience: Patience for early stopping
            constraint_weight: Weight for biomechanical constraints
            gradient_clip_norm: Gradient clipping norm
            save_every_n_epochs: Save checkpoint every N epochs
            
        Returns:
            Training history dictionary
        """
        self.logger.info("Starting training...")
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        else:
            scheduler = None
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch(
                data_loaders['train'], optimizer, constraint_weight, gradient_clip_norm
            )
            
            # Validation phase
            val_metrics = self._validate_epoch(data_loaders['val'], constraint_weight)
            
            # Update learning rate
            if scheduler_type == 'plateau' and scheduler is not None:
                scheduler.step(val_metrics['total_loss'])
            elif scheduler is not None:
                scheduler.step()
            
            # Log metrics
            current_lr = optimizer.param_groups[0]['lr']
            self.training_history['train_losses'].append(train_metrics['total_loss'])
            self.training_history['val_losses'].append(val_metrics['total_loss'])
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)
            self.training_history['learning_rates'].append(current_lr)
            
            # Early stopping check
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, optimizer, scheduler, is_best=True)
            else:
                self.epochs_without_improvement += 1
            
            # Log progress
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['total_loss']:.6f}, "
                    f"Val Loss: {val_metrics['total_loss']:.6f}, "
                    f"LR: {current_lr:.2e}"
                )
            
            # Save periodic checkpoint
            if epoch % save_every_n_epochs == 0 and epoch > 0:
                self._save_checkpoint(epoch, optimizer, scheduler, is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation
        self.logger.info("Training completed. Running final evaluation...")
        test_metrics = self._evaluate_model(data_loaders['test'])
        
        # Save final results
        self._save_training_results(test_metrics)
        
        return self.training_history
    
    def _train_epoch(self, 
                    data_loader: DataLoader, 
                    optimizer: optim.Optimizer,
                    constraint_weight: float,
                    gradient_clip_norm: float) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_reconstruction_loss = 0.0
        total_constraint_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch)
            
            # Compute losses
            loss_dict = self.model.compute_loss(
                predictions, 
                batch.y,
                biomechanical_constraints=self.graph_builder.biomechanical_constraints,
                constraint_weight=constraint_weight
            )
            
            # Backward pass
            loss_dict['total'].backward()
            
            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)
            
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict['total'].item()
            total_reconstruction_loss += loss_dict['reconstruction'].item()
            total_constraint_loss += loss_dict['biomechanical'].item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_reconstruction_loss / num_batches,
            'constraint_loss': total_constraint_loss / num_batches
        }
    
    def _validate_epoch(self, data_loader: DataLoader, constraint_weight: float) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_reconstruction_loss = 0.0
        total_constraint_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(batch)
                
                # Compute losses
                loss_dict = self.model.compute_loss(
                    predictions, 
                    batch.y,
                    biomechanical_constraints=self.graph_builder.biomechanical_constraints,
                    constraint_weight=constraint_weight
                )
                
                # Accumulate losses
                total_loss += loss_dict['total'].item()
                total_reconstruction_loss += loss_dict['reconstruction'].item()
                total_constraint_loss += loss_dict['biomechanical'].item()
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_reconstruction_loss / num_batches,
            'constraint_loss': total_constraint_loss / num_batches
        }
    
    def _evaluate_model(self, data_loader: DataLoader) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                predictions = self.model(batch)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Compute comprehensive metrics
        metrics = {
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'per_coordinate_mse': [
                mean_squared_error(targets[:, i], predictions[:, i]) 
                for i in range(min(3, targets.shape[1]))
            ]
        }
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, 
                        scheduler: Optional[Any], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.experiment_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.experiment_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation loss: {self.best_val_loss:.6f}")
    
    def _save_training_results(self, test_metrics: Dict[str, float]):
        """Save comprehensive training results."""
        results = {
            'experiment_name': self.experiment_name,
            'training_history': self.training_history,
            'final_test_metrics': test_metrics,
            'best_val_loss': self.best_val_loss
        }
        
        results_path = self.experiment_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create training plots
        self._create_training_plots()
        
        self.logger.info(f"Training results saved to {results_path}")
    
    def _create_training_plots(self):
        """Create visualization plots for training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(1, len(self.training_history['train_losses']) + 1)
        axes[0, 0].plot(epochs, self.training_history['train_losses'], label='Train')
        axes[0, 0].plot(epochs, self.training_history['val_losses'], label='Validation')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(epochs, self.training_history['learning_rates'])
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Component losses (if available)
        if self.training_history['train_metrics']:
            train_recon = [m['reconstruction_loss'] for m in self.training_history['train_metrics']]
            train_constraint = [m['constraint_loss'] for m in self.training_history['train_metrics']]
            
            axes[1, 0].plot(epochs, train_recon, label='Reconstruction')
            axes[1, 0].plot(epochs, train_constraint, label='Constraint')
            axes[1, 0].set_title('Training Loss Components')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Validation metrics
        if self.training_history['val_metrics']:
            val_recon = [m['reconstruction_loss'] for m in self.training_history['val_metrics']]
            val_constraint = [m['constraint_loss'] for m in self.training_history['val_metrics']]
            
            axes[1, 1].plot(epochs, val_recon, label='Reconstruction')
            axes[1, 1].plot(epochs, val_constraint, label='Constraint')
            axes[1, 1].set_title('Validation Loss Components')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.experiment_dir / "training_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_training_experiment(dataset_paths: List[Union[str, Path]],
                             experiment_name: str,
                             model_config: Optional[Dict[str, Any]] = None,
                             training_config: Optional[Dict[str, Any]] = None,
                             experiment_dir: Union[str, Path] = "./experiments") -> MotionPredictionTrainer:
    """
    Convenience function to create a complete training experiment.
    
    Args:
        dataset_paths: List of paths to TRC files
        experiment_name: Name for the experiment
        model_config: Model configuration dictionary
        training_config: Training configuration dictionary
        experiment_dir: Directory for experiments
        
    Returns:
        Configured MotionPredictionTrainer instance
    """
    # Default configurations
    default_model_config = {
        'node_features': 9,
        'hidden_dim': 128,
        'num_heads': 8,
        'num_layers': 6,
        'prediction_horizon': 10
    }
    
    default_training_config = {
        'sequence_length': 50,
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'constraint_weight': 0.1
    }
    
    # Merge with user configs
    model_config = {**default_model_config, **(model_config or {})}
    training_config = {**default_training_config, **(training_config or {})}
    
    # Create graph builder with biomechanical constraints
    graph_builder = KinematicGraphBuilder(
        marker_names=None,  # Will be inferred from data
        use_biomechanical_constraints=True
    )
    
    # Create model
    model = MotionPredictor(
        graph_builder=graph_builder,
        **model_config
    )
    
    # Create trainer
    trainer = MotionPredictionTrainer(
        model=model,
        graph_builder=graph_builder,
        experiment_name=experiment_name,
        experiment_dir=experiment_dir
    )
    
    return trainer
