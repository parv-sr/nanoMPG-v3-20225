import pandas as pd
import numpy as np
from typing import Dict

import logging
import os

logger = logging.getLogger(__name__)

rng = np.random.default_rng(seed=42)
NUM_SAMPLES = int(os.getenv("SAMPLE_SIZE", "25000"))

FEATURE_NAMES = [
    "engine_size", "horsepower", "weight", "cylinders",
    "age", "turbo_pressure", "fuel_octane", "drivetrain_ratio",
    "compression_ratio", "aerodynamic_drag", "tire_width",
    "ambient_temp", "altitude", "ethanol_blend"
]


class DataWorks:
    def __init__(self, filepath: str, synthetic: bool = True, samples: int = 25000):
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
        self.compression_ratio: np.ndarray
        self.aerodynamic_drag: np.ndarray
        self.tire_width: np.ndarray
        self.ambient_temp: np.ndarray
        self.altitude: np.ndarray
        self.ethanol_blend: np.ndarray

        self.noise: np.ndarray
        self.target: np.ndarray
        self.data: Dict[str, np.ndarray]

    def load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.fp)
        return df

    # ------------------------------------------------------------------ #
    #                       FEATURE GENERATION                            #
    # ------------------------------------------------------------------ #
    def generate_features(self) -> "DataWorks":
        n = self.samples

        # --- Original 8 features ---
        self.engine_size = rng.lognormal(mean=0.7, sigma=0.35, size=n)
        np.clip(self.engine_size, 0.8, 6.5, out=self.engine_size)

        raw_hp = rng.gamma(shape=4.0, scale=45.0, size=n) + 70
        self.horsepower = np.clip(raw_hp, 80, 700).astype(int)

        self.weight = rng.normal(loc=1500, scale=350, size=n)
        np.clip(self.weight, 900, 3000, out=self.weight)
        self.weight = self.weight.astype(int)

        cyl_opts = np.array([3, 4, 5, 6, 8, 10, 12])
        cyl_probs = np.array([0.05, 0.40, 0.03, 0.28, 0.18, 0.02, 0.04])
        self.cylinders = rng.choice(cyl_opts, size=n, p=cyl_probs)

        self.age = rng.exponential(scale=6.0, size=n)
        np.clip(self.age, 0, 30, out=self.age)
        self.age = self.age.astype(int)

        is_turbo = rng.random(size=n) < 0.45
        turbo_vals = rng.normal(loc=1.2, scale=0.4, size=n)
        np.clip(turbo_vals, 0.3, 3.0, out=turbo_vals)
        self.turbo_pressure = np.where(is_turbo, turbo_vals, 0.0)

        oct_opts = np.array([87, 89, 91, 93, 95])
        oct_probs = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
        self.fuel_octane = rng.choice(oct_opts, size=n, p=oct_probs)

        self.drivetrain_ratio = rng.normal(loc=3.5, scale=0.6, size=n)
        np.clip(self.drivetrain_ratio, 2.0, 5.5, out=self.drivetrain_ratio)

        # --- New 6 features ---

        # Compression ratio: higher = more thermally efficient, but
        # constrained by fuel octane in practice.  Mean ~10.5 for gasoline.
        self.compression_ratio = rng.normal(loc=10.5, scale=1.2, size=n)
        np.clip(self.compression_ratio, 8.0, 14.0, out=self.compression_ratio)

        # Aerodynamic drag coefficient (Cd). Sedans ~0.28-0.32,
        # SUVs ~0.35-0.42, sports cars ~0.25-0.30.
        self.aerodynamic_drag = rng.normal(loc=0.32, scale=0.05, size=n)
        np.clip(self.aerodynamic_drag, 0.22, 0.50, out=self.aerodynamic_drag)

        # Tire section width (mm). Wider = more rolling resistance.
        self.tire_width = rng.normal(loc=225, scale=25, size=n)
        np.clip(self.tire_width, 175, 315, out=self.tire_width)
        self.tire_width = self.tire_width.astype(int)

        # Ambient temperature (Celsius). Affects air density, lubricant
        # viscosity, and A/C load.
        self.ambient_temp = rng.normal(loc=20, scale=10, size=n)
        np.clip(self.ambient_temp, -10, 45, out=self.ambient_temp)

        # Altitude (metres above sea level). Affects air density, hence
        # engine volumetric efficiency and aerodynamic drag.
        self.altitude = rng.exponential(scale=400, size=n)
        np.clip(self.altitude, 0, 3000, out=self.altitude)
        self.altitude = self.altitude.astype(int)

        # Ethanol blend percentage.  Common options: E0, E10, E15, E85.
        eth_opts = np.array([0, 10, 15, 85])
        eth_probs = np.array([0.40, 0.35, 0.15, 0.10])
        self.ethanol_blend = rng.choice(eth_opts, size=n, p=eth_probs)

        return self

    # ------------------------------------------------------------------ #
    #                 PHYSICS-BASED TARGET GENERATION                     #
    # ------------------------------------------------------------------ #
    def generate_targets(self) -> "DataWorks":
        """
        Generate MPG targets using a physics-inspired nonlinear function.

        The formula captures real automotive relationships:
        - Otto-cycle thermal efficiency  (compression ratio)
        - Aerodynamic drag losses         (Cd, air density)
        - Rolling resistance              (tire width, weight)
        - Drivetrain efficiency            (gear ratio)
        - Forced induction effects         (turbo, altitude)
        - Fuel energy content              (ethanol blend, octane)
        - Environmental factors            (altitude, temperature)
        - Aging and degradation
        - ~35 terms with cross-feature interactions
        """
        e   = self.engine_size
        hp  = self.horsepower.astype(float)
        w   = self.weight.astype(float)
        cyl = self.cylinders.astype(float)
        age = self.age.astype(float)
        tp  = self.turbo_pressure
        oct = self.fuel_octane.astype(float)
        dr  = self.drivetrain_ratio
        cr  = self.compression_ratio
        cd  = self.aerodynamic_drag
        tw  = self.tire_width.astype(float)
        tmp = self.ambient_temp
        alt = self.altitude.astype(float)
        eth = self.ethanol_blend.astype(float)

        # Derived physical quantities
        rho = np.exp(-alt / 8500) * (293.0 / (273.0 + tmp))   # relative air density
        eta = 1.0 - 1.0 / (cr ** 0.35)                        # Otto-cycle efficiency

        self.target = (
            # ── Thermal efficiency base ──────────────────────────
            80.0 * eta                              # ~44 MPG for CR=10.5

            # ── Engine displacement ──────────────────────────────
            - 2.8 * e
            + 0.6 * e * np.sin(cyl * np.pi / 6.0)

            # ── Horsepower (diminishing marginal cost) ───────────
            + 12.0 * np.exp(-0.005 * hp)
            - 0.0018 * hp * np.log(age + 1.0)

            # ── Vehicle weight ───────────────────────────────────
            - 0.004 * w
            + 0.00007 * w * e ** 1.5
            - 0.35 * dr * np.log(w / 1000.0)

            # ── Cylinder configuration ───────────────────────────
            + 3.2 * np.sin(cyl * np.pi / 6.0)

            # ── Age degradation ──────────────────────────────────
            - 0.8 * np.log(age + 1.0)
            - 0.0000025 * hp ** 2 / (age + 1.0)

            # ── Turbocharger effects ─────────────────────────────
            - 1.5 * tp * np.sqrt(e)
            + 0.4 * tp * np.sin(oct / 15.0)
            + 0.25 * tp * rho               # turbo more effective in dense air

            # ── Fuel properties ──────────────────────────────────
            + 0.12 * (oct - 87.0) * np.log(hp + 1.0)
            - 0.065 * eth                    # ethanol has ~33 % less energy/gal
            + 0.0008 * eth * (oct - 87.0)    # high-octane + ethanol synergy

            # ── Aerodynamic drag ─────────────────────────────────
            - 10.0 * cd ** 2                 # quadratic Cd penalty
            - 0.0025 * cd * w               # heavy + poor aero compounds
            + 2.5 * (0.30 - cd)              # reward below-average Cd

            # ── Tire rolling resistance ──────────────────────────
            - 0.015 * tw * (w / 1000.0)
            + 0.006 * (tw - 225.0) * np.log(hp + 1.0) / 10.0

            # ── Altitude effects ─────────────────────────────────
            + 0.0004 * alt * eta             # thin air ⇒ slightly less drag
            - 0.00025 * alt * tp             # turbo compensates for altitude

            # ── Temperature effects ──────────────────────────────
            + 0.04 * tmp * np.log(cr)        # warmer ⇒ better combustion
            - 0.0008 * np.abs(tmp - 20.0) * w / 1000.0  # extreme temps hurt

            # ── Higher-order cross-feature interactions ───────────
            + 1.1 * np.cos(e * dr)
            - 0.12 * np.sqrt(hp) * cd
            + 0.015 * cr * np.log(e + 1.0) * cyl
            - 0.00008 * w * np.exp(-0.01 * hp)
            + 0.4 * np.tanh((cr - 10.0) * 2.0) * eta   # CR sweet-spot
            - 0.02 * np.sqrt(tw) * cd * rho             # tyre–aero–air interaction
            + 0.003 * (oct - 87.0) * np.log(cr) * np.sqrt(e)  # fuel–engine match
            - 0.0006 * alt * cd * np.sqrt(w / 1000.0)   # altitude–aero–weight
            + 0.15 * np.sin(dr * np.pi / 2.0) * eta     # drivetrain resonance
        )

        return self

    # ------------------------------------------------------------------ #
    #                        NOISE INJECTION                              #
    # ------------------------------------------------------------------ #
    def add_noise(self) -> "DataWorks":
        """
        Add heteroscedastic Gaussian noise.

        Base σ = 2.5 (vs 0.4 previously), with additional variance
        for older vehicles and heavier vehicles — mirroring real-world
        measurement / driving-style variability.
        """
        base_sigma = 2.5
        # Heteroscedastic component: older and heavier vehicles are noisier
        hetero = (
            0.08 * self.age
            + 0.0005 * (self.weight - 1500)
        )
        sigma = base_sigma + hetero
        self.noise = rng.normal(loc=0, scale=np.abs(sigma), size=self.samples)
        self.target += self.noise
        return self

    # ------------------------------------------------------------------ #
    #                      DATASET ASSEMBLY                               #
    # ------------------------------------------------------------------ #
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
            "compression_ratio": self.compression_ratio,
            "aerodynamic_drag": self.aerodynamic_drag,
            "tire_width": self.tire_width,
            "ambient_temp": self.ambient_temp,
            "altitude": self.altitude,
            "ethanol_blend": self.ethanol_blend,
            "targets": self.target,
        }
        return self

    def to_csv(self) -> None:
        self.df = pd.DataFrame(self.data)
        self.df.to_csv(self.fp, index=False)
        return


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    n = int(os.getenv("SAMPLE_SIZE", "25000"))
    dw = DataWorks("training_data/training_25k.csv", samples=n)
    logger.info("Dataset generation started...")
    try:
        dw.generate_features().generate_targets().add_noise().build_dataset().to_csv()
        logger.info(f"Data generation successful. Saved to: {dw.fp}")
        print(f"Generated {n} samples -> {dw.fp}")
        print(f"Target range: {dw.target.min():.2f} to {dw.target.max():.2f}")
        print(f"Target mean:  {dw.target.mean():.2f}, std: {dw.target.std():.2f}")
    except Exception as e:
        logger.exception(f"An error occured: {e}")
