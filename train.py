from dotenv import load_dotenv
load_dotenv()

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import MPGDataset
from model.model import MPGModel
from training.history import TrainingHistory
from training.train import train, evaluate
from visualisation.visualiser import Visualiser

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.getenv(
    "TRAINING_DATA",
    os.path.join(BASE_DIR, "training_data", "training_10k.csv")
)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "bin")
VISUALISATION_DIR = os.path.join(BASE_DIR, "visualisations")

LAYER_SIZES = [8, 64, 128, 64, 32, 1]
ACTIVATION = "silu"
USE_BATCH_NORM = True
DROPOUT = 0.1
USE_RESIDUAL = False

FEATURE_NAMES = [
    "engine_size", "horsepower", "weight", "cylinders",
    "age", "turbo_pressure", "fuel_octane", "drivetrain_ratio"
]


def get_next_version(bin_dir: str) -> int:
    os.makedirs(bin_dir, exist_ok=True)
    existing = [f for f in os.listdir(bin_dir) if f.startswith("MPG-v") and f.endswith(".pt")]
    if not existing:
        return 1
    versions = []
    for f in existing:
        try:
            parts = f.replace(".pt", "").split("-")
            v = int(parts[1][1:])
            versions.append(v)
        except (IndexError, ValueError):
            continue
    return max(versions) + 1 if versions else 1


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VISUALISATION_DIR, exist_ok=True)

    training_set = MPGDataset(filepath=DATA, train=True).prepare()
    testing_set = MPGDataset(
        filepath=DATA, train=False, stats=training_set.stats
    ).prepare()

    train_loader = DataLoader(
        dataset=training_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=testing_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = MPGModel(
        layer_sizes=LAYER_SIZES,
        activation=ACTIVATION,
        use_batch_norm=USE_BATCH_NORM,
        dropout=DROPOUT,
        use_residual=USE_RESIDUAL
    )

    param_count = model.parameter_count()
    print(f"Model: {LAYER_SIZES}")
    print(f"Parameters: {param_count}")
    print(f"Activation: {ACTIVATION}")

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    history = TrainingHistory()

    history = train(
        model, train_loader, criterion, optimiser,
        EPOCHS, history, scheduler=scheduler
    )
    history = evaluate(model, test_loader, criterion, history)

    version = get_next_version(CHECKPOINT_DIR)
    checkpoint_name = f"MPG-v{version}-{param_count}.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "epochs": EPOCHS,
        "normalisation_stats": training_set.stats,
        "training_history": history,
        "version": version,
        "parameter_count": param_count,
        "layer_sizes": LAYER_SIZES,
        "activation": ACTIVATION
    }
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, checkpoint_name))
    print(f"Checkpoint saved: {checkpoint_name}")

    sample_batch = next(iter(test_loader))[0]

    visualiser = Visualiser(
        history=history,
        model=model,
        output_dir=VISUALISATION_DIR,
        stats=training_set.stats,
        sample_input=sample_batch,
        feature_names=FEATURE_NAMES,
        test_features=testing_set.x,
        test_targets=testing_set.y,
        criterion=criterion
    )
    visualiser.generate_all()


if __name__ == "__main__":
    main()
