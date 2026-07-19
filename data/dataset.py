from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from typing import List, Any, Tuple
from dataclasses import dataclass
import os

NUM_SAMPLES = int(os.getenv("SAMPLE_SIZE"))

@dataclass
class NormalisationStats:
    feature_mean: np.ndarray
    feature_std: np.ndarray


class MPGDataset(Dataset):
    def __init__(
            self,
            filepath: str | Path,
            train: bool,
            stats: NormalisationStats | None = None,
            train_split: float = 0.8,
            random_seed: int = 42
    ) -> None:
        self.fp = Path(filepath)

        self.train = train
        self.train_split = train_split
        self.random_seed = random_seed

        self.df: pd.DataFrame | None = None   # Raw pandas dataframe

        self.features: np.ndarray | None = None  #numpy arrays
        self.targets: np.ndarray | None = None

        self.x: torch.Tensor | None = None      #pytorch tensors
        self.y: torch.Tensor | None = None

        self.stats = stats


    def load_csv(self) -> "MPGDataset":
        self.df = pd.read_csv(self.fp)
        return self
    
    def extract_features_and_targets(self) -> "MPGDataset":
        self.targets = self.df["targets"].to_numpy()
        self.features = self.df.drop(columns=['targets']).to_numpy()
        return self
    
    def train_test_split(self) -> "MPGDataset" | np.ndarray[np.float64]:
        shuffled_indices: List[int] = list(range(len(self.features)))
        np.random.default_rng(seed=self.random_seed).shuffle(shuffled_indices)
        split_idx = int(self.train_split * len(shuffled_indices))

        train_indices = shuffled_indices[:split_idx]
        test_indices = shuffled_indices[split_idx:]

        features_train = self.features[train_indices]
        features_test = self.features[test_indices]

        targets_train = self.targets[train_indices]
        targets_test = self.targets[test_indices]

        if not self.train:
            self.targets = targets_test
            self.features = features_test
            return self
        else:
            self.targets = targets_train
            self.features = features_train
            return self

    def normalise(self) -> "MPGDataset":
        if self.train:
            feature_mean = np.mean(self.features, axis=0)
            feature_std = np.std(self.features, axis=0)

            feature_std[feature_std == 0] = 1.0

            self.stats = NormalisationStats(
                feature_mean=feature_mean,
                feature_std=feature_std
            )

        else:
            if self.stats is None:
                raise ValueError(
                    "Test dataset requires NormalisationStats from the training dataset."
                )
            feature_mean = self.stats.feature_mean
            feature_std = self.stats.feature_std

        self.features = (self.features - feature_mean) / feature_std

        return self

    def to_tensors(self) -> "MPGDataset":
        self.x = torch.tensor(self.features, dtype=torch.float32)
        self.y = torch.tensor(self.targets, dtype=torch.float32)
        self.y = self.y[:, None]       # convert from (8000, ) to (8000, 1)
        return self

    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        return (self.x[index], self.y[index])
    
    def prepare(self) -> "MPGDataset":
        self.load_csv().extract_features_and_targets().train_test_split().normalise().to_tensors()
        return self
    

if __name__ == "__main__":
    train_dataset = MPGDataset(
        filepath=r"C:\F DRIVE\nonlinear_regression\training_data\training_10k.csv",
        train=True,
        train_split=0.8,
        random_seed=42
    ).prepare()

    test_dataset = MPGDataset(
        filepath=r"C:\F DRIVE\nonlinear_regression\training_data\training_10k.csv",
        train=False,
        stats=train_dataset.stats,
        train_split=0.8,
        random_seed=42
    ).prepare()
    