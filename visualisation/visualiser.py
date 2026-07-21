import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from typing import List, Union

from training.history import TrainingHistory
from model.model import MPGModel
from data.dataset import NormalisationStats

try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None

try:
    import umap
except ImportError:
    umap = None


class Visualiser:
    def __init__(
            self,
            history: TrainingHistory,
            model: MPGModel,
            output_dir: Union[str, Path],
            stats: Union[NormalisationStats, None] = None,
            sample_input: Union[torch.Tensor, None] = None,
            feature_names: Union[List[str], None] = None,
            test_features: Union[torch.Tensor, None] = None,
            test_targets: Union[torch.Tensor, None] = None,
            criterion: Union[nn.Module, None] = None
    ):
        self.history = history
        self.model = model
        self.output_dir = Path(output_dir)
        self.stats = stats
        self.sample_input = sample_input
        self.feature_names = feature_names
        self.test_features = test_features
        self.test_targets = test_targets
        self.criterion = criterion

        self.plots_dir = self.output_dir / 'plots'
        self.animations_dir = self.output_dir / 'animations'

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.animations_dir.mkdir(parents=True, exist_ok=True)

    def _hidden_function(
            self, engine_size, horsepower, weight, cylinders, age,
            turbo_pressure, fuel_octane, drivetrain_ratio
    ):
        return (
            42
            - 2.8 * engine_size
            + 12 * np.exp(-0.005 * horsepower)
            - 0.004 * weight
            + 3.2 * np.sin(cylinders * np.pi / 6)
            - 0.8 * np.log(age + 1)
            - 1.5 * turbo_pressure * np.sqrt(engine_size)
            + 0.12 * (fuel_octane - 87) * np.log(horsepower + 1)
            - 0.35 * drivetrain_ratio * np.log(weight / 1000)
            + 0.6 * engine_size * np.sin(cylinders * np.pi / 6)
            - 0.0018 * horsepower * np.log(age + 1)
            + 0.00007 * weight * engine_size ** 1.5
            + 1.1 * np.cos(engine_size * drivetrain_ratio)
            - 0.0000025 * horsepower ** 2 / (age + 1)
            + 0.4 * turbo_pressure * np.sin(fuel_octane / 15)
        )

    def _normalise_features(self, features):
        if self.stats is None:
            return features
        mean = torch.tensor(self.stats.feature_mean, dtype=torch.float32, device=features.device)
        std = torch.tensor(self.stats.feature_std, dtype=torch.float32, device=features.device)
        return (features - mean) / std

    def _build_temp_model(self):
        # Detect if the source model uses batch norm (has BatchNorm1d in norms)
        use_batch_norm = any(
            isinstance(m, nn.BatchNorm1d) for m in self.model.norms
        )
        # Detect dropout probability
        dropout = self.model.drop.p if isinstance(self.model.drop, nn.Dropout) else 0.0
        # Detect activation names per hidden layer
        act_names = []
        act_map = {
            nn.Sigmoid: "sigmoid", nn.ReLU: "relu", nn.LeakyReLU: "leaky_relu",
            nn.ELU: "elu", nn.GELU: "gelu", nn.SiLU: "silu", nn.Mish: "mish",
            nn.Tanh: "tanh", nn.PReLU: "prelu",
        }
        for act in self.model.acts:
            act_names.append(act_map.get(type(act), "silu"))

        return MPGModel(
            layer_sizes=self.model.layer_sizes,
            activation=act_names if act_names else "silu",
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            use_residual=self.model.use_residual,
        )

    def _predict_with_snapshot(self, snapshot, inputs):
        temp_model = self._build_temp_model()
        temp_model.load_state_dict(snapshot)
        temp_model.eval()
        with torch.no_grad():
            return temp_model(inputs).cpu().numpy().flatten()

    def plot_loss_curve(self):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.plot(self.history.train_loss, color='#2196F3')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'loss_curve.png')
        plt.close(fig)

    def plot_test_loss(self):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.plot(self.history.train_loss, color='#2196F3')
        if len(self.history.test_loss) == 1:
            ax.axhline(self.history.test_loss[0], color='r', linestyle='--')
        elif len(self.history.test_loss) > 1:
            ax.plot(self.history.test_loss, color='r')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'test_loss.png')
        plt.close(fig)

    def plot_prediction_vs_target(self):
        if not self.history.predictions or not self.history.targets:
            return
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        truth = np.array(self.history.targets)
        pred = np.array(self.history.predictions)
        ax.scatter(truth, pred, alpha=0.5)
        min_val = min(truth.min(), pred.min())
        max_val = max(truth.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='r', linestyle='--')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'prediction_vs_target.png')
        plt.close(fig)

    def plot_residuals(self):
        if not self.history.predictions or not self.history.residuals:
            return
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.scatter(self.history.predictions, self.history.residuals, alpha=0.5)
        ax.axhline(0, color='r', linestyle='--')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'residuals.png')
        plt.close(fig)

    def plot_residual_distribution(self):
        if not self.history.residuals:
            return
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.hist(self.history.residuals, bins=50, alpha=0.7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'residual_distribution.png')
        plt.close(fig)

    def plot_weight_evolution(self):
        if not self.history.fc1_weights:
            return
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        weights = [w[0, 0] for w in self.history.fc1_weights]
        ax.plot(weights)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'weight_evolution.png')
        plt.close(fig)

    def plot_hidden_layer_activations(self):
        if self.sample_input is None:
            return
        activations = self.model.get_hidden_activations(self.sample_input)
        fig, axes = plt.subplots(len(activations), 1, figsize=(10, 2 * len(activations)), dpi=150)
        if len(activations) == 1:
            axes = [axes]
        for i, act in enumerate(activations):
            act_np = act[:32].cpu().detach().numpy()
            im = axes[i].imshow(act_np, aspect='auto', cmap='viridis')
            fig.colorbar(im, ax=axes[i])
            axes[i].grid(False)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'hidden_layer_activations.png')
        plt.close(fig)

    def plot_gradient_flow(self):
        if not self.history.gradient_norms:
            return
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        last_grads = self.history.gradient_norms[-1]
        weight_grads = {k: v for k, v in last_grads.items() if 'weight' in k}
        ax.bar(list(weight_grads.keys()), list(weight_grads.values()))
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'gradient_flow.png')
        plt.close(fig)

    def plot_learning_rate(self):
        if not self.history.learning_rates:
            return
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.plot(self.history.learning_rates)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'learning_rate.png')
        plt.close(fig)

    def plot_partial_dependence(self):
        if self.feature_names is None or self.test_features is None:
            return
        num_features = len(self.feature_names)
        cols = 3
        rows = (num_features + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), dpi=150)
        axes = axes.flatten()
        mean_features = self.test_features.mean(dim=0)
        for i, name in enumerate(self.feature_names):
            min_val = self.test_features[:, i].min().item()
            max_val = self.test_features[:, i].min().item() if False else self.test_features[:, i].max().item()
            x_vals = torch.linspace(min_val, max_val, 50)
            preds = []
            for val in x_vals:
                feat = mean_features.clone()
                feat[i] = val
                with torch.no_grad():
                    pred = self.model(feat.unsqueeze(0)).item()
                preds.append(pred)
            axes[i].plot(x_vals.numpy(), preds)
            axes[i].grid(True, alpha=0.3)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'partial_dependence.png')
        plt.close(fig)

    def plot_dead_neurons(self):
        if self.test_features is None:
            return
        activations = self.model.get_hidden_activations(self.test_features)
        dead_pcts = []
        for act in activations:
            dead_mask = (act.abs() < 0.01).float()
            dead_pct = dead_mask.mean(dim=0).mean().item() * 100
            dead_pcts.append(dead_pct)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.bar(range(len(dead_pcts)), dead_pcts)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'dead_neurons.png')
        plt.close(fig)

    def plot_feature_importance(self):
        if self.test_features is None or self.feature_names is None:
            return
        samples = self.test_features[:200]
        baseline = torch.zeros_like(samples)
        steps = 50
        importances = torch.zeros(samples.shape[1])
        for i in range(samples.size(0)):
            sample = samples[i:i+1]
            base = baseline[i:i+1]
            alphas = torch.linspace(0, 1, steps).view(-1, 1).to(sample.device)
            path = base + alphas * (sample - base)
            path.requires_grad_(True)
            preds = self.model(path)
            preds.sum().backward()
            grads = path.grad.mean(dim=0)
            importances += (grads * (sample - base).squeeze(0)).abs().cpu()
        importances /= samples.size(0)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.bar(self.feature_names, importances.numpy())
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'feature_importance.png')
        plt.close(fig)

    def plot_latent_space_tsne(self):
        if self.test_features is None or self.test_targets is None or TSNE is None:
            return
        activations = self.model.get_hidden_activations(self.test_features)
        if not activations:
            return
        mid_idx = len(activations) // 2
        mid_acts = activations[mid_idx].cpu().detach().numpy()
        tsne = TSNE(n_components=2, perplexity=30)
        embedded = tsne.fit_transform(mid_acts)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        sc = ax.scatter(embedded[:, 0], embedded[:, 1], c=self.test_targets.cpu().numpy().flatten(), cmap='viridis', alpha=0.6)
        fig.colorbar(sc, ax=ax)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'latent_space_tsne.png')
        plt.close(fig)

    def plot_latent_space_umap(self):
        if self.test_features is None or self.test_targets is None or umap is None:
            return
        activations = self.model.get_hidden_activations(self.test_features)
        if not activations:
            return
        mid_idx = len(activations) // 2
        mid_acts = activations[mid_idx].cpu().detach().numpy()
        reducer = umap.UMAP()
        embedded = reducer.fit_transform(mid_acts)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        sc = ax.scatter(embedded[:, 0], embedded[:, 1], c=self.test_targets.cpu().numpy().flatten(), cmap='viridis', alpha=0.6)
        fig.colorbar(sc, ax=ax)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'latent_space_umap.png')
        plt.close(fig)

    def plot_loss_landscape(self):
        if self.test_features is None or self.test_targets is None or self.criterion is None:
            return
        samples_x = self.test_features[:500]
        samples_y = self.test_targets[:500]
        params = torch.cat([p.flatten() for p in self.model.parameters()]).detach()
        dir1 = torch.randn_like(params)
        dir2 = torch.randn_like(params)
        dir1 = dir1 / dir1.norm() * params.norm()
        dir2 = dir2 / dir2.norm() * params.norm()
        grid_size = 21
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(grid_size):
            for j in range(grid_size):
                new_params = params + X[i, j] * dir1 + Y[i, j] * dir2
                temp_model = self._build_temp_model()
                idx = 0
                for p in temp_model.parameters():
                    numel = p.numel()
                    p.data.copy_(new_params[idx:idx+numel].view_as(p))
                    idx += numel
                with torch.no_grad():
                    preds = temp_model(samples_x)
                    loss = self.criterion(preds, samples_y).item()
                Z[i, j] = loss
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        cp = ax.contourf(X, Y, Z, levels=30, cmap='viridis')
        fig.colorbar(cp, ax=ax)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.plots_dir / 'loss_landscape.png')
        plt.close(fig)

    def animate_learning_curve(self):
        if not self.history.model_snapshots:
            return
        engine_sizes = np.linspace(0.8, 6.5, 200)
        true_vals = self._hidden_function(engine_sizes, 200, 2000, 6, 5, 0.8, 91, 3.5)
        raw_inputs = np.zeros((200, 8))
        raw_inputs[:, 0] = engine_sizes
        raw_inputs[:, 1] = 200
        raw_inputs[:, 2] = 2000
        raw_inputs[:, 3] = 6
        raw_inputs[:, 4] = 5
        raw_inputs[:, 5] = 0.8
        raw_inputs[:, 6] = 91
        raw_inputs[:, 7] = 3.5
        norm_inputs = self._normalise_features(torch.tensor(raw_inputs, dtype=torch.float32))
        snapshots = self.history.model_snapshots
        if len(snapshots) > 100:
            indices = np.linspace(0, len(snapshots) - 1, 100).astype(int)
            snapshots = [snapshots[i] for i in indices]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.plot(engine_sizes, true_vals, 'g--', label='True')
        line, = ax.plot([], [], 'r-', label='Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.8, 6.5)
        ax.set_ylim(min(true_vals) - 10, max(true_vals) + 10)
        def init():
            line.set_data([], [])
            return line,
        def update(frame):
            preds = self._predict_with_snapshot(frame, norm_inputs)
            line.set_data(engine_sizes, preds)
            return line,
        anim = animation.FuncAnimation(fig, update, frames=snapshots, init_func=init, blit=True)
        try:
            writer = animation.FFMpegWriter(fps=10)
            anim.save(str(self.animations_dir / "learning_curve.mp4"), writer=writer)
        except Exception:
            writer = animation.PillowWriter(fps=10)
            anim.save(str(self.animations_dir / "learning_curve.gif"), writer=writer)
        plt.close(fig)

    def animate_surface(self):
        if not self.history.model_snapshots:
            return
        e_size = np.linspace(0.8, 6.5, 30)
        hp = np.linspace(80, 700, 30)
        E, H = np.meshgrid(e_size, hp)
        raw_inputs = np.zeros((900, 8))
        raw_inputs[:, 0] = E.flatten()
        raw_inputs[:, 1] = H.flatten()
        raw_inputs[:, 2] = 2000
        raw_inputs[:, 3] = 6
        raw_inputs[:, 4] = 5
        raw_inputs[:, 5] = 0.8
        raw_inputs[:, 6] = 91
        raw_inputs[:, 7] = 3.5
        norm_inputs = self._normalise_features(torch.tensor(raw_inputs, dtype=torch.float32))
        snapshots = self.history.model_snapshots
        if len(snapshots) > 60:
            indices = np.linspace(0, len(snapshots) - 1, 60).astype(int)
            snapshots = [snapshots[i] for i in indices]
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        def update(frame):
            ax.clear()
            preds = self._predict_with_snapshot(frame, norm_inputs)
            Z = preds.reshape(30, 30)
            ax.plot_surface(E, H, Z, cmap='viridis', alpha=0.8)
            ax.set_zlim(-50, 150)
        anim = animation.FuncAnimation(fig, update, frames=snapshots, blit=False)
        try:
            writer = animation.FFMpegWriter(fps=10)
            anim.save(str(self.animations_dir / "surface_fit.mp4"), writer=writer)
        except Exception:
            writer = animation.PillowWriter(fps=10)
            anim.save(str(self.animations_dir / "surface_fit.gif"), writer=writer)
        plt.close(fig)

    def animate_activation_distributions(self):
        if not self.history.model_snapshots or self.sample_input is None:
            return
        snapshots = self.history.model_snapshots
        if len(snapshots) > 30:
            indices = np.linspace(0, len(snapshots) - 1, 30).astype(int)
            snapshots = [snapshots[i] for i in indices]
        num_layers = self.model.num_hidden
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 2 * num_layers), dpi=150)
        if num_layers == 1:
            axes = [axes]
        def update(frame):
            temp_model = self._build_temp_model()
            temp_model.load_state_dict(frame)
            temp_model.eval()
            with torch.no_grad():
                acts = temp_model.get_hidden_activations(self.sample_input)
            for i, ax in enumerate(axes):
                ax.clear()
                ax.hist(acts[i].cpu().numpy().flatten(), bins=50, alpha=0.7)
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
        anim = animation.FuncAnimation(fig, update, frames=snapshots, blit=False)
        try:
            writer = animation.FFMpegWriter(fps=10)
            anim.save(str(self.animations_dir / "activation_distributions.mp4"), writer=writer)
        except Exception:
            writer = animation.PillowWriter(fps=10)
            anim.save(str(self.animations_dir / "activation_distributions.gif"), writer=writer)
        plt.close(fig)

    def animate_latent_evolution(self):
        if not self.history.model_snapshots or self.test_features is None or self.test_targets is None or TSNE is None:
            return
        snapshots = self.history.model_snapshots
        if len(snapshots) > 20:
            indices = np.linspace(0, len(snapshots) - 1, 20).astype(int)
            snapshots = [snapshots[i] for i in indices]
        samples_x = self.test_features[:500]
        samples_y = self.test_targets[:500].cpu().numpy().flatten()
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        def update(frame):
            ax.clear()
            temp_model = self._build_temp_model()
            temp_model.load_state_dict(frame)
            temp_model.eval()
            with torch.no_grad():
                acts = temp_model.get_hidden_activations(samples_x)
            if not acts:
                return
            mid_idx = len(acts) // 2
            mid_acts = acts[mid_idx].cpu().numpy()
            tsne = TSNE(n_components=2, perplexity=30)
            embedded = tsne.fit_transform(mid_acts)
            ax.scatter(embedded[:, 0], embedded[:, 1], c=samples_y, cmap='viridis', alpha=0.6)
            ax.grid(True, alpha=0.3)
        anim = animation.FuncAnimation(fig, update, frames=snapshots, blit=False)
        try:
            writer = animation.FFMpegWriter(fps=10)
            anim.save(str(self.animations_dir / "latent_evolution.mp4"), writer=writer)
        except Exception:
            writer = animation.PillowWriter(fps=10)
            anim.save(str(self.animations_dir / "latent_evolution.gif"), writer=writer)
        plt.close(fig)

    def generate_all(self):
        print("plot_loss_curve")
        self.plot_loss_curve()
        print("plot_test_loss")
        self.plot_test_loss()
        print("plot_prediction_vs_target")
        self.plot_prediction_vs_target()
        print("plot_residuals")
        self.plot_residuals()
        print("plot_residual_distribution")
        self.plot_residual_distribution()
        print("plot_weight_evolution")
        self.plot_weight_evolution()
        print("plot_hidden_layer_activations")
        self.plot_hidden_layer_activations()
        print("plot_gradient_flow")
        self.plot_gradient_flow()
        print("plot_learning_rate")
        self.plot_learning_rate()
        print("plot_partial_dependence")
        self.plot_partial_dependence()
        print("plot_dead_neurons")
        self.plot_dead_neurons()
        print("plot_feature_importance")
        self.plot_feature_importance()
        print("plot_latent_space_tsne")
        self.plot_latent_space_tsne()
        print("plot_latent_space_umap")
        self.plot_latent_space_umap()
        print("plot_loss_landscape")
        self.plot_loss_landscape()
        print("animate_learning_curve")
        self.animate_learning_curve()
        print("animate_surface")
        self.animate_surface()
        print("animate_activation_distributions")
        self.animate_activation_distributions()
        print("animate_latent_evolution")
        self.animate_latent_evolution()

