from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    fc1_weights: List[Any] = field(default_factory=list)
    predictions: List[float] = field(default_factory=list)
    targets: List[float] = field(default_factory=list)
    residuals: List[float] = field(default_factory=list)
    model_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    gradient_norms: List[Dict[str, float]] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
