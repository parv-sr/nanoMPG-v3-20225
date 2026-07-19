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

        self.engine_size: np.ndarray[float]
        self.horsepower: np.ndarray[int]
        self.weight: np.ndarray
        self.cylinders: np.ndarray
        self.age: np.ndarray

        self.noise: np.ndarray

        self.target: np.ndarray

        self.data: Dict[str, np.ndarray[float]]

    def load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.fp)
        return df
    
    def generate_features(self) -> "DataWorks":
        self.engine_size = rng.normal(loc=2.5, scale=0.8, size=self.samples)
        np.clip(self.engine_size, a_min=1.0, a_max=5.0, out=self.engine_size)

        self.horsepower = np.random.randint(low=90, high=500, size=self.samples)
        self.weight = np.random.randint(low=800, high=3500, size=self.samples)

        possible_num_cylinders = np.array([3, 4, 5, 6, 8, 10, 12])
        self.cylinders = rng.choice(possible_num_cylinders, size=self.samples)

        self.age = np.random.randint(low=0, high=25, size=self.samples)

        return self
    
    def generate_targets(self) -> "DataWorks":
        """
        Hidden nonlinear function that governs the data's nature.
        F(E, H, W, C, A) = 55 - 3E - 0.00012*H^0.5 - 0.005W + 4*sin(C) + 2*ln(A + 1)
        """
        self.target = (55 - 3*self.engine_size - 0.00012*(self.horsepower**0.5) - 0.005*self.weight + 4*np.sin(self.cylinders) + 2*np.log(self.age + 1))
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
            "engine_size" : self.engine_size,
            "horsepower" : self.horsepower,
            "weight" : self.weight,
            "cylinders" : self.cylinders,
            "age" : self.age,
            "targets" : self.target
        }
        return self
    
    
if __name__ == "__main__":
    dw = DataWorks("training_data/training_10k.csv", samples=NUM_SAMPLES)
    logger.info("Dataset generation started...")
    try:
        dw.generate_features().generate_targets().add_noise().build_dataset().to_csv()
        logger.info(f"Data generation successful. Saved to: {dw.fp}")
    except Exception as e:
        logger.exception(f"An error occured: {e}")



        

