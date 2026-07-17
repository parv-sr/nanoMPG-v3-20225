from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from typing import List, Any


class MPGDataset(Dataset):
    def __init__(
            self,
            filepath: str | Path,
            train: bool,
            train_split: float = 0.8,
            random_seed: int = 42
    ) -> None:
        self.fp = Path(filepath)

        self.train = train
        self.train_split = train_split
        self.random_seed = random_seed

        self.df: pd.DataFrame | None = None   # Raw pandas dataframe

        self.features: np.ndarray | None = None  #numpy arrays
        self.features_train: np.ndarray | None = None
        self.features_test: np.ndarray | None = None

        self.targets: np.ndarray | None = None
        self.targets_train: np.ndarray | None = None
        self.targets_test: np.ndarray | None = None

        self.x: torch.Tensor | None = None      #pytorch tensors
        self.y: torch.Tensor | None = None


        self.feature_mean: np.ndarray | None = None     #normalisation stats
        self.feature_std: np.ndarray | None = None


    def load_csv(self) -> "MPGDataset":
        self.df = pd.read_csv(self.fp)
        return self
    
    def extract_features_and_targets(self) -> "MPGDataset":
        self.targets = self.df["targets"].to_numpy()
        self.features = self.df.drop(columns=['targets']).to_numpy()
        return self
    
    def train_test_split(self) -> "MPGDataset" | np.ndarray[np.float64]:
        shuffled_indices: List[int] = list(range(10000))
        np.random.default_rng(seed=self.random_seed).shuffle(shuffled_indices)
        split_idx = int(self.train_split * len(shuffled_indices))

        train_indices = shuffled_indices[:split_idx]
        test_indices = shuffled_indices[split_idx:]

        self.features_train = self.features[train_indices]
        self.features_test = self.features[test_indices]

        self.targets_train = self.targets[train_indices]
        self.targets_test = self.targets[test_indices]
        
        if not self.train:
            return self
        else:
            return self.features_test, self.targets_test

    def normalise(self) -> "MPGDataset":
        features_mean = np.mean(self.features_train, axis=0)
        features_stddev = np.std(self.features_train, axis=0)

        targets_mean = np.mean(self.targets_train, axis=0)
        targets_stddev = np.std(self.targets_train, axis=0)

        features_stddev[features_stddev == 0] = 1.0
        if targets_stddev == 0:
            targets_stddev = 1.0

        self.features_train = (self.features_train - features_mean) / features_stddev
        self.targets_train = (self.targets_train - targets_mean) / targets_stddev

        return self

    def to_tensors(self) -> torch.Tensor:
        self.x = torch.from_numpy(self.features_train)
        self.y = torch.from_numpy(self.targets_train)

    def __len__(self) -> int:
        return len(self.features_train)
    
    def __getitem__(index: int) -> Any:
        pass