"""
Regenerate visualisations from a saved checkpoint without retraining.
Usage: python generate_visualisations.py [checkpoint_path]
"""
from dotenv import load_dotenv
load_dotenv()

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import MPGDataset
from model.model import MPGModel
from visualisation.visualiser import Visualiser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALISATION_DIR = os.path.join(BASE_DIR, "visualisations")


def find_latest_checkpoint(bin_dir: str) -> str:
    """Find the latest checkpoint file in the bin directory."""
    files = [f for f in os.listdir(bin_dir) if f.startswith("MPG-v") and f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"No checkpoint files found in {bin_dir}")
    # Sort by version number
    def version_key(f):
        try:
            return int(f.replace(".pt", "").split("-")[1][1:])
        except (IndexError, ValueError):
            return 0
    files.sort(key=version_key, reverse=True)
    return os.path.join(bin_dir, files[0])


def main():
    bin_dir = os.path.join(BASE_DIR, "bin")
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_checkpoint(bin_dir)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    layer_sizes = checkpoint["layer_sizes"]
    activation = checkpoint.get("activation", "silu")
    use_batch_norm = checkpoint.get("use_batch_norm", True)
    dropout = checkpoint.get("dropout", 0.1)
    use_residual = checkpoint.get("use_residual", False)
    feature_names = checkpoint.get("feature_names", [
        "engine_size", "horsepower", "weight", "cylinders",
        "age", "turbo_pressure", "fuel_octane", "drivetrain_ratio",
        "compression_ratio", "aerodynamic_drag", "tire_width",
        "ambient_temp", "altitude", "ethanol_blend"
    ])
    stats = checkpoint["normalisation_stats"]
    history = checkpoint["training_history"]

    model = MPGModel(
        layer_sizes=layer_sizes,
        activation=activation,
        use_batch_norm=use_batch_norm,
        dropout=0.0,   # no dropout for inference
        use_residual=use_residual,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    print(f"Model loaded: {layer_sizes}, params={checkpoint['parameter_count']}")
    print(f"Snapshots available: {len(history.model_snapshots)}")
    if history.eval_metrics:
        m = history.eval_metrics
        print(f"Eval metrics: MSE={m.mse:.4f}, RMSE={m.rmse:.4f}, MAE={m.mae:.4f}, R²={m.r2:.4f}")

    testing_set = None
    sample_batch = None
    criterion = nn.MSELoss()

    data_path = os.getenv(
        "TRAINING_DATA",
        os.path.join(BASE_DIR, "training_data", "training_25k.csv")
    )
    if os.path.exists(data_path):
        training_set = MPGDataset(filepath=data_path, train=True).prepare()
        testing_set = MPGDataset(
            filepath=data_path, train=False, stats=training_set.stats
        ).prepare()
        test_loader = DataLoader(dataset=testing_set, batch_size=64, shuffle=False)
        sample_batch = next(iter(test_loader))[0]

    os.makedirs(VISUALISATION_DIR, exist_ok=True)

    visualiser = Visualiser(
        history=history,
        model=model,
        output_dir=VISUALISATION_DIR,
        stats=stats,
        sample_input=sample_batch,
        feature_names=feature_names,
        test_features=testing_set.x if testing_set else None,
        test_targets=testing_set.y if testing_set else None,
        criterion=criterion,
    )

    visualiser.generate_all()
    print("Done!")


if __name__ == "__main__":
    main()
