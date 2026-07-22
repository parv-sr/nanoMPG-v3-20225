import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import copy

from training.history import TrainingHistory, EvaluationMetrics


def train(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimiser: torch.optim.Optimizer,
        epochs: int,
        history: TrainingHistory,
        scheduler=None,
        snapshot_interval: int = 10,
) -> TrainingHistory:
    """
    Train the model and record history.

    Parameters
    ----------
    snapshot_interval : int
        Save a model snapshot every N epochs (default 10).
        Reduces memory usage for long training runs.
    """

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        epoch_grad_norms: dict = {}

        for x, y in train_loader:
            predictions = model(x)
            loss = criterion(predictions, y)
            optimiser.zero_grad()
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in epoch_grad_norms:
                        epoch_grad_norms[name] = []
                    epoch_grad_norms[name].append(param.grad.data.norm(2).item())

            optimiser.step()
            running_loss += loss.item()
            batch_count += 1

        avg_loss = running_loss / batch_count
        avg_grad_norms = {
            name: sum(vals) / len(vals)
            for name, vals in epoch_grad_norms.items()
        }

        history.train_loss.append(avg_loss)
        history.fc1_weights.append(model.layers[0].weight.data.clone().cpu().numpy())
        history.gradient_norms.append(avg_grad_norms)
        history.learning_rates.append(optimiser.param_groups[0]["lr"])

        # Snapshot at configured interval to control memory
        if (epoch + 1) % snapshot_interval == 0 or epoch == 0:
            history.model_snapshots.append(copy.deepcopy(model.state_dict()))

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch: {epoch + 1:04d} | "
            f"Loss: {avg_loss:.6f} | "
            f"LR: {optimiser.param_groups[0]['lr']:.8f}"
        )

    return history


def evaluate(
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        history: TrainingHistory
) -> TrainingHistory:

    model.eval()
    running_loss = 0.0
    batch_count = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x)
            loss = criterion(preds, y)
            running_loss += loss.item()
            batch_count += 1
            all_predictions.append(preds)
            all_targets.append(y)

    avg_loss = running_loss / batch_count

    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy().flatten()
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy().flatten()
    all_residuals = all_predictions - all_targets

    history.test_loss.append(avg_loss)
    history.predictions = all_predictions.tolist()
    history.targets = all_targets.tolist()
    history.residuals = all_residuals.tolist()

    # ── Compute evaluation metrics ───────────────────────────────
    mse = float(np.mean(all_residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(all_residuals)))
    max_err = float(np.max(np.abs(all_residuals)))
    median_ae = float(np.median(np.abs(all_residuals)))

    ss_res = float(np.sum(all_residuals ** 2))
    ss_tot = float(np.sum((all_targets - all_targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    history.eval_metrics = EvaluationMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        max_error=max_err,
        median_ae=median_ae,
    )

    # Print summary table
    print("\n" + "=" * 50)
    print("         EVALUATION METRICS")
    print("=" * 50)
    print(f"  MSE          : {mse:.6f}")
    print(f"  RMSE         : {rmse:.6f}")
    print(f"  MAE          : {mae:.6f}")
    print(f"  R²           : {r2:.6f}")
    print(f"  Max Error    : {max_err:.6f}")
    print(f"  Median AE    : {median_ae:.6f}")
    print("=" * 50 + "\n")

    return history
