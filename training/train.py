import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy

from training.history import TrainingHistory


def train(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimiser: torch.optim.Optimizer,
        epochs: int,
        history: TrainingHistory,
        scheduler=None
) -> TrainingHistory:

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
        history.model_snapshots.append(copy.deepcopy(model.state_dict()))
        history.gradient_norms.append(avg_grad_norms)
        history.learning_rates.append(optimiser.param_groups[0]["lr"])

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

    return history
