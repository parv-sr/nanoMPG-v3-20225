import os
import glob
import json
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

from dotenv import load_dotenv
load_dotenv()

from model.model import MPGModel
from training.history import TrainingHistory
from training.train import train, evaluate
from data.dataset import MPGDataset
from visualisation.visualiser import Visualiser

FEATURE_NAMES = [
    "engine_size", "horsepower", "weight", "cylinders",
    "age", "turbo_pressure", "fuel_octane", "drivetrain_ratio",
    "compression_ratio", "aerodynamic_drag", "tire_width",
    "ambient_temp", "altitude", "ethanol_blend"
]

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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

def build_scheduler(name: str, optimiser, epochs: int, steps_per_epoch: int = 1):
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    elif name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimiser, step_size=200, gamma=0.5)
    elif name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=50)
    elif name == 'one_cycle':
        max_lr = optimiser.param_groups[0]['lr'] * 10
        return torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=max_lr, total_steps=epochs)
    return None

def run_experiment(config_path: str, data_path: str, base_dir: str) -> dict:
    config = load_config(config_path)
    name = config['name']

    results_dir = os.path.join(base_dir, "experiments", "results", name)
    vis_dir = os.path.join(results_dir, "visualisations")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    training_set = MPGDataset(filepath=data_path, train=True).prepare()
    testing_set = MPGDataset(filepath=data_path, train=False, stats=training_set.stats).prepare()

    train_loader = DataLoader(dataset=training_set, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=testing_set, batch_size=config['training']['batch_size'], shuffle=False)

    model = MPGModel(**config['model'])
    param_count = model.parameter_count()

    print(f"Experiment: {name}")
    print(f"Architecture: {config['model']['layer_sizes']}")
    print(f"Parameters: {param_count}")

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = build_scheduler(config['training']['scheduler'], optimiser, config['training']['epochs'])

    history = TrainingHistory()

    history = train(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimiser=optimiser,
        epochs=config['training']['epochs'],
        history=history,
        scheduler=scheduler
    )

    history = evaluate(model=model, test_loader=test_loader, criterion=criterion, history=history)

    bin_dir = os.path.join(base_dir, "bin")
    version = get_next_version(bin_dir)
    checkpoint_name = f"MPG-v{version}-{param_count}.pt"
    checkpoint_path = os.path.join(bin_dir, checkpoint_name)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "epochs": config['training']['epochs'],
        "normalisation_stats": training_set.stats,
        "training_history": history,
        "version": version,
        "parameter_count": param_count,
        "layer_sizes": config['model']['layer_sizes'],
        "activation": config['model']['activation']
    }
    torch.save(checkpoint, checkpoint_path)

    final_train_loss = history.train_loss[-1] if history.train_loss else 0.0
    final_test_loss = history.test_loss[-1] if history.test_loss else 0.0

    summary = {
        "name": name,
        "version": version,
        "param_count": param_count,
        "final_train_loss": final_train_loss,
        "final_test_loss": final_test_loss,
        "checkpoint_name": checkpoint_name
    }

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    sample_batch = next(iter(test_loader))[0]
    visualiser = Visualiser(
        history=history,
        model=model,
        output_dir=vis_dir,
        stats=training_set.stats,
        sample_input=sample_batch,
        feature_names=FEATURE_NAMES,
        test_features=testing_set.x,
        test_targets=testing_set.y,
        criterion=criterion
    )
    visualiser.generate_all()

    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str)
    group.add_argument('--all', action='store_true')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_path = os.path.join(base_dir, "training_data", "training_10k.csv")
    data_path = os.getenv("TRAINING_DATA", default_data_path)

    if args.config:
        summary = run_experiment(args.config, data_path, base_dir)
        print(summary)
    elif args.all:
        configs_dir = os.path.join(base_dir, "experiments", "configs")
        config_files = glob.glob(os.path.join(configs_dir, "*.yaml"))

        results = []
        for cf in config_files:
            res = run_experiment(cf, data_path, base_dir)
            results.append(res)

        print("\nExperiment Summary:")
        print(f"{'Name':<20} | {'Version':<7} | {'Params':<8} | {'Train Loss':<10} | {'Test Loss':<10}")
        print("-" * 65)
        for r in results:
            print(f"{r['name']:<20} | v{r['version']:<6} | {r['param_count']:<8} | {r['final_train_loss']:<10.4f} | {r['final_test_loss']:<10.4f}")
