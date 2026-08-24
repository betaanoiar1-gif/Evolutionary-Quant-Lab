from __future__ import annotations

import importlib
import os
import platform
import random
import sys
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    python: str
    platform: str
    cpu_count: int
    dependencies: dict[str, str]
    seed: int


class EnvironmentEngine:
    REQUIRED = ("numpy", "pandas", "pydantic", "yaml")

    def __init__(self, seed: int = 42) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.seed = seed

    def seed_all(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

    def inspect(self) -> EnvironmentReport:
        versions: dict[str, str] = {}
        for name in self.REQUIRED:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        return EnvironmentReport(sys.version.split()[0], platform.platform(), os.cpu_count() or 1, versions, self.seed)

    def validate(self) -> EnvironmentReport:
        report = self.inspect()
        if sys.version_info < (3, 11):
            raise RuntimeError("Python 3.11+ is required")
        self.seed_all()
        return report
