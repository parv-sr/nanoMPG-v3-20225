from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class EvaluationMetrics:
    """Container for standard regression evaluation metrics."""
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    max_error: float = 0.0
    median_ae: float = 0.0


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
    eval_metrics: Optional[EvaluationMetrics] = None
