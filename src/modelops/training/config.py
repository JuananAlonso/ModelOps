"""Configuración del experimento, separada de la ubicación de sus artefactos."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    test_size: float = 0.2

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("La semilla debe ser un entero entre 0 y 2**32 - 1.")
        if (
            isinstance(self.test_size, bool)
            or not isinstance(self.test_size, (float, int))
            or not math.isfinite(self.test_size)
            or not 0 < self.test_size < 1
        ):
            raise ValueError("test_size debe ser una fracción finita entre 0 y 1.")
