# Nonlinear Regression MPG Prediction

## Project Overview
This project implements a PyTorch-based neural network for nonlinear regression, specifically designed to predict vehicle fuel economy (MPG) using a synthetic, highly complex automotive dataset.

## Architecture Highlights
The core model is a Multi-Layer Perceptron (MLP) featuring:
- Configurable hidden layers and neurons
- Variety of activation functions (ReLU, SiLU, GELU, Mish, etc.)
- Optional Batch Normalization and Dropout layers
- Optional Residual (skip) connections for deep network training

For a detailed breakdown of the model, see the [Architecture Documentation](architecture.md) and the [Activation Guide](activation_guide.md).

## Directory Structure
```text
nonlinear_regression/
├── data/           # Dataset generation scripts and generated CSV files
├── model/          # Neural network architectures and PyTorch module definitions
├── training/       # Training loops, loss functions, and optimization routines
├── visualisation/  # Scripts for generating training/evaluation plots
├── observability/  # Logging, metrics tracking, and experiment monitoring
├── experiments/    # Experiment configurations and hyperparameter sweeps
└── docs/           # Project documentation (you are here)
```

## How to Run

1. **Train a model**:
   ```bash
   python train.py --config config/default.yaml
   ```
2. **Generate visualizations**:
   ```bash
   python generate_visualisations.py --run-dir path/to/run
   ```
3. **Run experiment sweeps**:
   ```bash
   python experiments/runner.py --sweep-config experiments/sweep.yaml
   ```

## Dependencies
The project requires the following primary libraries:
- `torch`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `python-dotenv`
- `pyyaml`
- `pillow`

For a deeper dive into the data, please refer to the [Dataset Specification](dataset.md). If you're interested in deployment, see the [Inference Plan](inference_plan.md).
