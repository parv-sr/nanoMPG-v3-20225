"""Activation tracker module for tracking PyTorch model layer activations via forward hooks."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn


class ActivationTracker:
    """Tracks activation statistics per layer using PyTorch forward hooks."""

    def __init__(self, model: nn.Module) -> None:
        """Initialize ActivationTracker.

        Args:
            model: PyTorch model module containing activation layers (`model.acts`).
        """
        self.model = model
        self._hooks: List[Any] = []
        self._current_stats: Dict[int, Dict[str, Any]] = {}
        self._history: List[Dict[str, Any]] = []  # per-epoch history
        self._registered: bool = False

    def register_hooks(self) -> None:
        """Register forward hooks on all activation layers"""
        if self._registered:
            return
        for i, act in enumerate(self.model.acts):
            hook = act.register_forward_hook(
                lambda module, inp, output, idx=i: self._record_stats(idx, output)
            )
            self._hooks.append(hook)
        self._registered = True

    def _record_stats(self, layer_idx: int, output: torch.Tensor) -> None:
        """Record statistics for a single activation layer"""
        with torch.no_grad():
            flat = output.detach().float()
            self._current_stats[layer_idx] = {
                'mean': flat.mean().item(),
                'std': flat.std().item(),
                'min': flat.min().item(),
                'max': flat.max().item(),
                'dead_neuron_pct': (flat.abs() < 0.01).float().mean().item() * 100,
                'near_zero_pct': (flat.abs() < 0.1).float().mean().item() * 100,
                'activation_type': type(self.model.acts[layer_idx]).__name__,
                'layer_size': output.shape[-1] if output.dim() > 1 else output.shape[0],
            }

    def snapshot(self, epoch: int) -> None:
        """Take a snapshot of current activation stats (call after a forward pass)"""
        snapshot = {'epoch': epoch}
        for idx, stats in self._current_stats.items():
            snapshot[f'layer_{idx}'] = stats.copy()
        self._history.append(snapshot)
        self._current_stats.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of activation statistics across training"""
        if not self._history:
            return {}

        summary: Dict[str, Any] = {
            'num_snapshots': len(self._history),
            'layers': {}
        }

        # Get layer indices from last snapshot
        last = self._history[-1]
        for key in last:
            if key.startswith('layer_'):
                idx = int(key.split('_')[1])
                layer_stats = [h[key] for h in self._history if key in h]
                summary['layers'][key] = {
                    'activation_type': layer_stats[-1]['activation_type'],
                    'layer_size': layer_stats[-1]['layer_size'],
                    'final_dead_neuron_pct': layer_stats[-1]['dead_neuron_pct'],
                    'avg_dead_neuron_pct': float(np.mean([s['dead_neuron_pct'] for s in layer_stats])),
                    'final_mean': layer_stats[-1]['mean'],
                    'final_std': layer_stats[-1]['std'],
                    'mean_range': [layer_stats[-1]['min'], layer_stats[-1]['max']],
                }
        return summary

    def get_history(self) -> List[Dict[str, Any]]:
        """Get full activation stats history"""
        return self._history

    def remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._registered = False

    def save(self, output_dir: Union[str, Path]) -> None:
        """Save activation tracking data to disk"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'activation_history.json', 'w') as f:
            json.dump(self._history, f, indent=2)

        with open(path / 'activation_summary.json', 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
