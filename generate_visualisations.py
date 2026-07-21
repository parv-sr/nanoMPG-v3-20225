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
DEFAULT_CHECKPOINT = os.path.join(BASE_DIR, "bin", "MPG-v1-19841.pt")
DATA = os.path.join(BASE_DIR, "training_data", "training_10k.csv")
VISUALISATION_DIR = os.path.join(BASE_DIR, "visualisations")

FEATURE_NAMES = [
    "engine_size", "horsepower", "weight", "cylinders",
    "age", "turbo_pressure", "fuel_octane", "drivetrain_ratio"
]


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    layer_sizes = checkpoint["layer_sizes"]
    activation = checkpoint.get("activation", "silu")
    stats = checkpoint["normalisation_stats"]
    history = checkpoint["training_history"]

    model = MPGModel(
        layer_sizes=layer_sizes,
        activation=activation,
        use_batch_norm=True,
        dropout=0.1,
        use_residual=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded: {layer_sizes}, params={checkpoint['parameter_count']}")
    print(f"Snapshots available: {len(history.model_snapshots)}")

    testing_set = None
    sample_batch = None
    criterion = nn.MSELoss()

    if os.path.exists(DATA):
        training_set = MPGDataset(filepath=DATA, train=True).prepare()
        testing_set = MPGDataset(
            filepath=DATA, train=False, stats=training_set.stats
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
        feature_names=FEATURE_NAMES,
        test_features=testing_set.x if testing_set else None,
        test_targets=testing_set.y if testing_set else None,
        criterion=criterion,
    )

    visualiser.generate_all()
    print("Done!")


if __name__ == "__main__":
    main()
