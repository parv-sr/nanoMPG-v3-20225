import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import os
from pathlib import Path
import copy

from training.history import TrainingHistory
from model.model import MPGModel
from data.dataset import NormalisationStats


class Visualiser:
    def __init__(
            self,
            history: TrainingHistory,
            model: MPGModel,
            output_dir: str | Path,
            stats: NormalisationStats | None = None,
            sample_input: torch.Tensor | None = None
    ):
        self.history = history
        self.model = model
        self.output_dir = Path(output_dir)
        self.stats = stats
        self.sample_input = sample_input
        self.plots_dir = self.output_dir / "plots"
        self.animations_dir = self.output_dir / "animations"
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.animations_dir, exist_ok=True)

    def plot_loss_curve(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(self.history.train_loss) + 1)
        ax.plot(epochs, self.history.train_loss, color="#2196F3", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss (MSE)")
        ax.set_title("Training Loss Curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "loss_curve.png", dpi=150)
        plt.close(fig)

    def plot_test_loss(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        if len(self.history.test_loss) == 1:
            epochs = range(1, len(self.history.train_loss) + 1)
            ax.plot(
                epochs, self.history.train_loss,
                color="#2196F3", linewidth=1.5, alpha=0.5, label="Training Loss"
            )
            ax.axhline(
                y=self.history.test_loss[0], color="#F44336",
                linewidth=2, linestyle="--",
                label=f"Test Loss: {self.history.test_loss[0]:.6f}"
            )
        else:
            epochs = range(1, len(self.history.test_loss) + 1)
            ax.plot(epochs, self.history.test_loss, color="#F44336", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title("Test Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "test_loss.png", dpi=150)
        plt.close(fig)

    def plot_prediction_vs_target(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        targets = np.array(self.history.targets)
        predictions = np.array(self.history.predictions)
        ax.scatter(targets, predictions, alpha=0.3, s=10, color="#4CAF50")
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot(
            [min_val, max_val], [min_val, max_val],
            color="#F44336", linewidth=2, linestyle="--", label="y = x"
        )
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.set_title("Prediction vs Target")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "prediction_vs_target.png", dpi=150)
        plt.close(fig)

    def plot_residuals(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        predictions = np.array(self.history.predictions)
        residuals = np.array(self.history.residuals)
        ax.scatter(predictions, residuals, alpha=0.3, s=10, color="#FF9800")
        ax.axhline(y=0, color="#F44336", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Residual (Prediction - Target)")
        ax.set_title("Residual Plot")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "residuals.png", dpi=150)
        plt.close(fig)

    def plot_residual_distribution(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        residuals = np.array(self.history.residuals)
        ax.hist(residuals, bins=50, color="#9C27B0", alpha=0.7, edgecolor="white")
        ax.axvline(x=0, color="#F44336", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Residual Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Residual Distribution")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "residual_distribution.png", dpi=150)
        plt.close(fig)

    def plot_weight_evolution(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        weight_values = [w[0, 0] for w in self.history.fc1_weights]
        epochs = range(1, len(weight_values) + 1)
        ax.plot(epochs, weight_values, color="#009688", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weight Value")
        ax.set_title("First Layer Weight Evolution (fc1[0,0])")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "weight_evolution.png", dpi=150)
        plt.close(fig)

    def plot_hidden_layer_activations(self) -> None:
        if self.sample_input is None:
            return

        self.model.eval()
        activations = []
        layer_names = []

        with torch.no_grad():
            x = self.sample_input

            x = self.model.fc1(x)
            x = self.model.activation(x)
            activations.append(x.cpu().numpy())
            layer_names.append("Layer 1")

            x = self.model.fc2(x)
            x = self.model.activation(x)
            activations.append(x.cpu().numpy())
            layer_names.append("Layer 2")

            x = self.model.fc3(x)
            x = self.model.activation(x)
            activations.append(x.cpu().numpy())
            layer_names.append("Layer 3")

            x = self.model.fc4(x)
            x = self.model.activation(x)
            activations.append(x.cpu().numpy())
            layer_names.append("Layer 4")

            x = self.model.fc5(x)
            x = self.model.activation(x)
            activations.append(x.cpu().numpy())
            layer_names.append("Layer 5")

        num_layers = len(activations)
        fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 6))

        if num_layers == 1:
            axes = [axes]

        for idx, (act, name) in enumerate(zip(activations, layer_names)):
            im = axes[idx].imshow(act[:32], aspect="auto", cmap="viridis")
            axes[idx].set_title(name)
            axes[idx].set_xlabel("Neuron")
            axes[idx].set_ylabel("Sample")
            plt.colorbar(im, ax=axes[idx], fraction=0.046)

        fig.suptitle("Hidden Layer Activations", fontsize=14)
        fig.tight_layout()
        fig.savefig(self.plots_dir / "hidden_layer_activations.png", dpi=150)
        plt.close(fig)

    def _hidden_function(self, engine_size, horsepower, weight, cylinders, age):
        return (
            55
            - 3 * engine_size
            - 0.00012 * (horsepower ** 0.5)
            - 0.005 * weight
            + 4 * np.sin(cylinders)
            + 2 * np.log(age + 1)
        )

    def _normalise_features(self, features: np.ndarray) -> np.ndarray:
        if self.stats is None:
            return features
        return (features - self.stats.feature_mean) / self.stats.feature_std

    def _predict_with_snapshot(self, snapshot, inputs: torch.Tensor) -> np.ndarray:
        temp_model = MPGModel()
        temp_model.load_state_dict(snapshot)
        temp_model.eval()
        with torch.no_grad():
            return temp_model(inputs).cpu().numpy().flatten()

    def animate_learning_curve(self) -> None:
        if not self.history.model_snapshots or self.stats is None:
            return

        horsepower_fixed = 200.0
        weight_fixed = 2000.0
        cylinders_fixed = 6.0
        age_fixed = 5.0

        engine_sizes = np.linspace(1.0, 5.0, 200)

        features_raw = np.column_stack([
            engine_sizes,
            np.full_like(engine_sizes, horsepower_fixed),
            np.full_like(engine_sizes, weight_fixed),
            np.full_like(engine_sizes, cylinders_fixed),
            np.full_like(engine_sizes, age_fixed)
        ])

        true_targets = self._hidden_function(
            engine_sizes, horsepower_fixed, weight_fixed,
            cylinders_fixed, age_fixed
        )

        features_normalised = self._normalise_features(features_raw)
        inputs_tensor = torch.tensor(features_normalised, dtype=torch.float32)

        total_snapshots = len(self.history.model_snapshots)
        max_frames = 100
        if total_snapshots <= max_frames:
            frame_indices = list(range(total_snapshots))
        else:
            frame_indices = np.linspace(
                0, total_snapshots - 1, max_frames, dtype=int
            ).tolist()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            engine_sizes, true_targets,
            color="#F44336", linewidth=2, label="True Function"
        )
        pred_line, = ax.plot([], [], color="#2196F3", linewidth=2, label="Network Prediction")
        epoch_text = ax.text(
            0.02, 0.95, "", transform=ax.transAxes,
            fontsize=12, verticalalignment="top"
        )

        ax.set_xlabel("Engine Size")
        ax.set_ylabel("Target (MPG)")
        ax.set_title("Learning Curve Animation")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        y_min = true_targets.min() - 5
        y_max = true_targets.max() + 5
        ax.set_ylim(y_min, y_max)

        def update(frame_num):
            idx = frame_indices[frame_num]
            snapshot = self.history.model_snapshots[idx]
            preds = self._predict_with_snapshot(snapshot, inputs_tensor)
            pred_line.set_data(engine_sizes, preds)
            epoch_text.set_text(f"Epoch: {idx + 1}")
            return pred_line, epoch_text

        anim = animation.FuncAnimation(
            fig, update, frames=len(frame_indices), interval=50, blit=True
        )

        try:
            anim.save(
                self.animations_dir / "learning_curve.mp4",
                writer="ffmpeg", fps=20, dpi=150
            )
        except Exception:
            anim.save(
                self.animations_dir / "learning_curve.gif",
                writer="pillow", fps=20, dpi=100
            )

        plt.close(fig)

    def animate_surface(self) -> None:
        if not self.history.model_snapshots or self.stats is None:
            return

        cylinders_fixed = 6.0
        age_fixed = 5.0
        weight_fixed = 2000.0

        engine_sizes = np.linspace(1.0, 5.0, 30)
        horsepowers = np.linspace(90.0, 500.0, 30)
        E, H = np.meshgrid(engine_sizes, horsepowers)

        features_raw = np.column_stack([
            E.ravel(),
            H.ravel(),
            np.full(E.size, weight_fixed),
            np.full(E.size, cylinders_fixed),
            np.full(E.size, age_fixed)
        ])

        true_targets = self._hidden_function(
            E.ravel(), H.ravel(), weight_fixed,
            cylinders_fixed, age_fixed
        ).reshape(E.shape)

        features_normalised = self._normalise_features(features_raw)
        inputs_tensor = torch.tensor(features_normalised, dtype=torch.float32)

        total_snapshots = len(self.history.model_snapshots)
        max_frames = 60
        if total_snapshots <= max_frames:
            frame_indices = list(range(total_snapshots))
        else:
            frame_indices = np.linspace(
                0, total_snapshots - 1, max_frames, dtype=int
            ).tolist()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            E, H, true_targets,
            alpha=0.3, color="#F44336"
        )

        initial_preds = self._predict_with_snapshot(
            self.history.model_snapshots[frame_indices[0]], inputs_tensor
        ).reshape(E.shape)
        surface = [ax.plot_surface(E, H, initial_preds, alpha=0.6, color="#2196F3")]

        ax.set_xlabel("Engine Size")
        ax.set_ylabel("Horsepower")
        ax.set_zlabel("Target (MPG)")
        ax.set_title("Surface Fit Animation")

        def update(frame_num):
            idx = frame_indices[frame_num]
            snapshot = self.history.model_snapshots[idx]
            preds = self._predict_with_snapshot(snapshot, inputs_tensor).reshape(E.shape)
            surface[0].remove()
            surface[0] = ax.plot_surface(E, H, preds, alpha=0.6, color="#2196F3")
            ax.set_title(f"Surface Fit - Epoch {idx + 1}")
            return surface

        anim = animation.FuncAnimation(
            fig, update, frames=len(frame_indices), interval=100
        )

        try:
            anim.save(
                self.animations_dir / "surface_fit.mp4",
                writer="ffmpeg", fps=10, dpi=150
            )
        except Exception:
            anim.save(
                self.animations_dir / "surface_fit.gif",
                writer="pillow", fps=10, dpi=100
            )

        plt.close(fig)

    def generate_all(self) -> None:
        print("Generating loss curve...")
        self.plot_loss_curve()
        print("Generating test loss plot...")
        self.plot_test_loss()
        print("Generating prediction vs target...")
        self.plot_prediction_vs_target()
        print("Generating residuals plot...")
        self.plot_residuals()
        print("Generating residual distribution...")
        self.plot_residual_distribution()
        print("Generating weight evolution...")
        self.plot_weight_evolution()
        print("Generating hidden layer activations...")
        self.plot_hidden_layer_activations()
        print("Generating learning curve animation...")
        self.animate_learning_curve()
        print("Generating surface fit animation...")
        self.animate_surface()
        print("All visualisations generated.")
