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
EPOCHS = 1000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.getenv(
    "TRAINING_DATA",
    os.path.join(BASE_DIR, "training_data", "training_10k.csv")
)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "bin")
VISUALISATION_DIR = os.path.join(BASE_DIR, "visualisations")


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

    model = MPGModel()
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = TrainingHistory()

    history = train(model, train_loader, criterion, optimiser, EPOCHS, history)
    history = evaluate(model, test_loader, criterion, history)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "epochs": EPOCHS,
        "normalisation_stats": training_set.stats,
        "training_history": history
    }
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "checkpoint.pt"))

    sample_batch = next(iter(test_loader))[0]

    visualiser = Visualiser(
        history=history,
        model=model,
        output_dir=VISUALISATION_DIR,
        stats=training_set.stats,
        sample_input=sample_batch
    )
    visualiser.generate_all()


if __name__ == "__main__":
    main()
