import pandas as pd
import numpy as np
from typing import Dict

import logging
import os

logger = logging.getLogger(__name__)

rng = np.random.default_rng()
NUM_SAMPLES = os.getenv("SAMPLE_SIZE")


class DataWorks:
    def __init__(self, filepath: str, synthetic: bool = True, samples: int = 10000):
        self.fp = filepath
        self.synthetic = synthetic
        self.samples = samples
        self.df: pd.DataFrame | None = None

        self.engine_size: np.ndarray
        self.horsepower: np.ndarray
        self.weight: np.ndarray
        self.cylinders: np.ndarray
        self.age: np.ndarray
        self.turbo_pressure: np.ndarray
        self.fuel_octane: np.ndarray
        self.drivetrain_ratio: np.ndarray

        self.noise: np.ndarray

        self.target: np.ndarray

        self.data: Dict[str, np.ndarray]

    def load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.fp)
        return df

    def generate_features(self) -> "DataWorks":
        self.engine_size = rng.lognormal(mean=0.7, sigma=0.35, size=self.samples)
        np.clip(self.engine_size, a_min=0.8, a_max=6.5, out=self.engine_size)

        raw_hp = rng.gamma(shape=4.0, scale=45.0, size=self.samples) + 70
        self.horsepower = np.clip(raw_hp, 80, 700).astype(int)

        self.weight = rng.normal(loc=1500, scale=350, size=self.samples)
        np.clip(self.weight, a_min=900, a_max=3000, out=self.weight)
        self.weight = self.weight.astype(int)

        cylinder_options = np.array([3, 4, 5, 6, 8, 10, 12])
        cylinder_probs = np.array([0.05, 0.40, 0.03, 0.28, 0.18, 0.02, 0.04])
        self.cylinders = rng.choice(cylinder_options, size=self.samples, p=cylinder_probs)

        self.age = rng.exponential(scale=6.0, size=self.samples)
        np.clip(self.age, a_min=0, a_max=30, out=self.age)
        self.age = self.age.astype(int)

        is_turbo = rng.random(size=self.samples) < 0.45
        turbo_vals = rng.normal(loc=1.2, scale=0.4, size=self.samples)
        np.clip(turbo_vals, a_min=0.3, a_max=3.0, out=turbo_vals)
        self.turbo_pressure = np.where(is_turbo, turbo_vals, 0.0)

        octane_options = np.array([87, 89, 91, 93, 95])
        octane_probs = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
        self.fuel_octane = rng.choice(octane_options, size=self.samples, p=octane_probs)

        self.drivetrain_ratio = rng.normal(loc=3.5, scale=0.6, size=self.samples)
        np.clip(self.drivetrain_ratio, a_min=2.0, a_max=5.5, out=self.drivetrain_ratio)

        return self

    def generate_targets(self) -> "DataWorks":
        self.target = (
            42
            - 2.8 * self.engine_size
            + 12 * np.exp(-0.005 * self.horsepower)
            - 0.004 * self.weight
            + 3.2 * np.sin(self.cylinders * np.pi / 6)
            - 0.8 * np.log(self.age + 1)
            - 1.5 * self.turbo_pressure * np.sqrt(self.engine_size)
            + 0.12 * (self.fuel_octane - 87) * np.log(self.horsepower + 1)
            - 0.35 * self.drivetrain_ratio * np.log(self.weight / 1000)
            + 0.6 * self.engine_size * np.sin(self.cylinders * np.pi / 6)
            - 0.0018 * self.horsepower * np.log(self.age + 1)
            + 0.00007 * self.weight * self.engine_size ** 1.5
            + 1.1 * np.cos(self.engine_size * self.drivetrain_ratio)
            - 0.0000025 * self.horsepower ** 2 / (self.age + 1)
            + 0.4 * self.turbo_pressure * np.sin(self.fuel_octane / 15)
        )
        return self

    def add_noise(self) -> "DataWorks":
        self.noise = rng.normal(loc=0, scale=0.4, size=self.samples)
        self.target += self.noise
        return self

    def to_csv(self) -> None:
        self.df = pd.DataFrame(self.data)
        self.df.to_csv(self.fp, index=False)
        return

    def build_dataset(self) -> "DataWorks":
        self.data = {
            "engine_size": self.engine_size,
            "horsepower": self.horsepower,
            "weight": self.weight,
            "cylinders": self.cylinders,
            "age": self.age,
            "turbo_pressure": self.turbo_pressure,
            "fuel_octane": self.fuel_octane,
            "drivetrain_ratio": self.drivetrain_ratio,
            "targets": self.target
        }
        return self


if __name__ == "__main__":
    dw = DataWorks("training_data/training_10k.csv", samples=int(NUM_SAMPLES))
    logger.info("Dataset generation started...")
    try:
        dw.generate_features().generate_targets().add_noise().build_dataset().to_csv()
        logger.info(f"Data generation successful. Saved to: {dw.fp}")
    except Exception as e:
        logger.exception(f"An error occured: {e}")
