"""Training logger module for structured JSON logging of training runs."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainingLogger:
    """Creates structured JSON log files for training runs."""

    def __init__(self, output_dir: str = 'observability') -> None:
        """Initialize training logger.

        Args:
            output_dir: Base directory path for output logs.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_logs: List[Dict[str, Any]] = []
        self.run_metadata: Dict[str, Any] = {}

    def log_run_start(
        self,
        model: Any,
        layer_sizes: List[int],
        activation: str,
        use_batch_norm: bool,
        dropout: float,
        use_residual: bool,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        dataset_size: int,
        feature_names: List[str],
        optimizer_name: str,
        scheduler_name: str,
        criterion_name: str,
    ) -> None:
        """Log all metadata about the training run.

        Args:
            model: PyTorch model object.
            layer_sizes: List of layer hidden dimensions.
            activation: Activation function name.
            use_batch_norm: Whether batch normalization is used.
            dropout: Dropout probability.
            use_residual: Whether residual connections are used.
            learning_rate: Initial learning rate.
            epochs: Total training epochs.
            batch_size: Batch size used for training.
            dataset_size: Size of training dataset.
            feature_names: List of input feature names.
            optimizer_name: Name of optimizer used.
            scheduler_name: Name of learning rate scheduler used.
            criterion_name: Name of loss function used.
        """
        param_count = sum(p.numel() for p in model.parameters())
        self.run_metadata = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'architecture': {
                'layer_sizes': layer_sizes,
                'activation': activation,
                'use_batch_norm': use_batch_norm,
                'dropout': dropout,
                'use_residual': use_residual,
                'parameter_count': param_count,
                'layer_details': self._get_layer_details(model),
            },
            'training': {
                'learning_rate': learning_rate,
                'epochs': epochs,
                'batch_size': batch_size,
                'dataset_size': dataset_size,
                'optimizer': optimizer_name,
                'scheduler': scheduler_name,
                'criterion': criterion_name,
                'samples_per_param_ratio': dataset_size / param_count if param_count > 0 else 0.0,
            },
            'features': feature_names,
        }
        # Save immediately
        with open(self.run_dir / 'run_metadata.json', 'w') as f:
            json.dump(self.run_metadata, f, indent=2)

    def _get_layer_details(self, model: Any) -> List[Dict[str, Any]]:
        """Extract exact architecture details including which activation is at each layer."""
        details: List[Dict[str, Any]] = []
        for i in range(len(model.layers)):
            layer_info: Dict[str, Any] = {
                'layer_index': i,
                'type': 'Linear',
                'in_features': model.layers[i].in_features,
                'out_features': model.layers[i].out_features,
                'parameters': model.layers[i].weight.numel() + model.layers[i].bias.numel(),
            }
            if hasattr(model, 'acts') and i < len(model.acts):
                layer_info['activation'] = type(model.acts[i]).__name__
                layer_info['activation_config'] = str(model.acts[i])
            if hasattr(model, 'norms') and i < len(model.norms):
                norm_type = type(model.norms[i]).__name__
                layer_info['normalisation'] = norm_type
                if norm_type == 'BatchNorm1d':
                    layer_info['norm_parameters'] = model.norms[i].weight.numel() * 2
            details.append(layer_info)
        return details

    def log_epoch(
        self,
        epoch: int,
        loss: float,
        lr: float,
        gradient_norms: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log a single epoch's data.

        Args:
            epoch: Epoch index.
            loss: Loss value.
            lr: Current learning rate.
            gradient_norms: Optional dictionary mapping layer names/indices to gradient norm values.
        """
        entry: Dict[str, Any] = {
            'epoch': epoch,
            'loss': loss,
            'learning_rate': lr,
        }
        if gradient_norms:
            entry['gradient_norms'] = gradient_norms
        self.epoch_logs.append(entry)

    def log_evaluation(self, metrics_dict: Dict[str, Any]) -> None:
        """Log evaluation metrics (MSE, RMSE, MAE, R², etc.)"""
        eval_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics_dict,
        }
        with open(self.run_dir / 'evaluation.json', 'w') as f:
            json.dump(eval_data, f, indent=2)

    def log_activation_stats(self, activation_stats: Dict[str, Any]) -> None:
        """Log activation statistics from ActivationTracker"""
        with open(self.run_dir / 'activation_stats.json', 'w') as f:
            json.dump(activation_stats, f, indent=2)

    def save(self) -> None:
        """Save all accumulated logs"""
        with open(self.run_dir / 'epoch_logs.json', 'w') as f:
            json.dump(self.epoch_logs, f, indent=2)

        # Also save a summary
        if self.epoch_logs:
            summary = {
                'run_id': self.run_id,
                'total_epochs': len(self.epoch_logs),
                'final_loss': self.epoch_logs[-1]['loss'],
                'best_loss': min(e['loss'] for e in self.epoch_logs),
                'best_epoch': min(range(len(self.epoch_logs)), key=lambda i: self.epoch_logs[i]['loss']),
            }
            with open(self.run_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
