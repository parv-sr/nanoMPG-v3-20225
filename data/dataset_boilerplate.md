from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class CarDataset(Dataset):
    """
    PyTorch Dataset for the synthetic car regression dataset.

    Responsibilities
    ----------------
    1. Load the CSV.
    2. Split into training/testing sets.
    3. Normalize the feature columns.
    4. Convert everything into PyTorch tensors.
    5. Serve one sample at a time.
    """

    def __init__(
        self,
        filepath: str | Path,
        train: bool = True,
        train_split: float = 0.8,
        random_seed: int = 42
    ) -> None:

        self.filepath = Path(filepath)

        self.train = train
        self.train_split = train_split
        self.random_seed = random_seed

        # Raw dataframe
        self.df: pd.DataFrame | None = None

        # Raw NumPy arrays
        self.features: np.ndarray | None = None
        self.targets: np.ndarray | None = None

        # PyTorch tensors
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None

        # Normalization statistics
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    # ----------------------------------------------------
    # Data preparation
    # ----------------------------------------------------

    def load_csv(self) -> "CarDataset":
        """
        Load the CSV into a DataFrame.
        """
        pass

    def extract_features_and_targets(self) -> "CarDataset":
        """
        Separate the feature matrix X
        from the target vector y.
        """
        pass

    def train_test_split(self) -> "CarDataset":
        """
        Shuffle and split the dataset.

        Should leave this object containing
        only the requested partition.
        """
        pass

    def normalize(self) -> "CarDataset":
        """
        Standard-score normalize
        every feature column.

        x = (x - mean) / std
        """
        pass

    def to_tensors(self) -> "CarDataset":
        """
        Convert NumPy arrays
        into torch.Tensors.
        """
        pass

    def prepare(self) -> "CarDataset":
        """
        Complete preprocessing pipeline.
        """

        (
            self
            .load_csv()
            .extract_features_and_targets()
            .train_test_split()
            .normalize()
            .to_tensors()
        )

        return self

    # ----------------------------------------------------
    # Required PyTorch methods
    # ----------------------------------------------------

    def __len__(self) -> int:
        """
        Number of samples.
        """
        pass

    def __getitem__(self, index: int):
        """
        Returns

        (
            feature_tensor,
            target_tensor
        )
        """
        pass