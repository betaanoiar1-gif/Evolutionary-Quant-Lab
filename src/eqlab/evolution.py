from __future__ import annotations

import random
from dataclasses import dataclass

from .backtest import BacktestEngine, PerformanceReport
from .dna import StrategyDNA


class StrategyGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def generate(self, n: int = 100) -> list[StrategyDNA]:
        if n < 1:
            raise ValueError("n must be positive")
        seen: set[str] = set()
        out: list[StrategyDNA] = []
        while len(out) < n:
            d = StrategyDNA(
                fast=self.rng.randint(5, 40),
                slow=self.rng.randint(50, 200),
                rsi_period=self.rng.randint(7, 30),
                rsi_entry=self.rng.uniform(35, 65),
                stop_atr=self.rng.uniform(1, 4),
                take_atr=self.rng.uniform(1.5, 6),
                allow_short=self.rng.random() < .5,
                use_adx_filter=self.rng.random() < .35,
                adx_min=self.rng.uniform(15, 35),
                use_volume_filter=self.rng.random() < .25,
                volume_multiplier=self.rng.uniform(.8, 1.8),
                cooldown_bars=self.rng.randint(0, 20),
                max_bars_in_trade=self.rng.choice([0, 12, 24, 48, 96]),
                risk_per_trade=self.rng.uniform(.005, .02),
                max_exposure=self.rng.uniform(.25, 1.0),
            )
            if d.fingerprint() not in seen:
                seen.add(d.fingerprint())
                out.append(d)
        return out


class EvolutionEngine:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def mutate(self, d: StrategyDNA) -> StrategyDNA:
        values = d.to_dict()
        key = self.rng.choice([
            "fast", "slow", "rsi_period", "rsi_entry", "stop_atr", "take_atr",
            "use_adx_filter", "adx_min", "use_volume_filter", "volume_multiplier",
            "cooldown_bars", "max_bars_in_trade", "risk_per_trade", "max_exposure",
        ])
        if key in ("fast", "slow", "rsi_period", "cooldown_bars", "max_bars_in_trade"):
            values[key] = max(0 if key in ("cooldown_bars", "max_bars_in_trade") else 2,
                              int(values[key] + self.rng.randint(-5, 5)))
        elif key == "rsi_entry":
            values[key] = min(70.0, max(30.0, values[key] + self.rng.uniform(-5, 5)))
        elif key in ("stop_atr", "take_atr"):
            values[key] = max(.5, values[key] + self.rng.uniform(-.5, .5))
        elif key in ("adx_min",):
            values[key] = min(50.0, max(5.0, values[key] + self.rng.uniform(-5, 5)))
        elif key in ("volume_multiplier",):
            values[key] = min(3.0, max(.5, values[key] + self.rng.uniform(-.2, .2)))
        elif key in ("risk_per_trade",):
            values[key] = min(.05, max(.0025, values[key] * self.rng.uniform(.7, 1.3)))
        elif key == "max_exposure":
            values[key] = min(1.0, max(.1, values[key] + self.rng.uniform(-.1, .1)))
        else:
            values[key] = not values[key]
        values["fast"] = min(values["fast"], values["slow"] - 1)
        values["slow"] = max(values["slow"], values["fast"] + 1)
        values["version"] = 2
        return StrategyDNA(**values)

    def crossover(self, a: StrategyDNA, b: StrategyDNA) -> StrategyDNA:
        fields = a.to_dict()
        other = b.to_dict()
        for key in fields:
            if key == "version":
                continue
            if self.rng.random() < .5:
                fields[key] = other[key]
        fields["fast"] = min(fields["fast"], fields["slow"] - 1)
        fields["slow"] = max(fields["slow"], fields["fast"] + 1)
        fields["version"] = 2
        return StrategyDNA(**fields)


@dataclass(frozen=True, slots=True)
class Candidate:
    dna: StrategyDNA
    report: PerformanceReport
    score: float


def score(report: PerformanceReport) -> float:
    if report.trades < 5:
        return -1e9
    pf_bonus = min(report.profit_factor, 5.0) * .2
    return (
        report.sharpe
        + report.sortino
        + report.total_return * 3
        + report.win_rate
        + pf_bonus
        + report.expectancy * 10
        - report.max_drawdown * 4
        - report.complexity * .001
    )


class SearchEngine:
    def __init__(self, backtester: BacktestEngine | None = None, seed: int = 42) -> None:
        self.bt = backtester or BacktestEngine()
        self.ev = EvolutionEngine(seed)
        self.gen = StrategyGenerator(seed)

    def evaluate(self, df, population):
        return [self._candidate(df, d) for d in population]

    def _candidate(self, df, d):
        report = self.bt.run(df, d)
        return Candidate(d, report, score(report))

    def evolve(self, df, population, generations: int = 5, elite: int = 10, mutation_rate: float = .8):
        if not population:
            raise ValueError("population must not be empty")
        if generations < 1 or elite < 1:
            raise ValueError("generations and elite must be positive")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0,1]")
        pop = list(dict.fromkeys(population))
        archive: dict[str, Candidate] = {}
        target = len(pop)
        for _ in range(generations):
            ranked = sorted(self.evaluate(df, pop), key=lambda c: c.score, reverse=True)
            for candidate in ranked:
                archive[candidate.dna.fingerprint()] = candidate
            keep = min(elite, len(ranked))
            parents = [c.dna for c in ranked[:keep]]
            next_pop = parents[:]
            while len(next_pop) < target:
                a = self.ev.rng.choice(parents)
                b = self.ev.rng.choice(parents)
                child = self.ev.crossover(a, b)
                if self.ev.rng.random() < mutation_rate:
                    child = self.ev.mutate(child)
                if child.fingerprint() not in {d.fingerprint() for d in next_pop}:
                    next_pop.append(child)
            pop = next_pop
        return sorted(archive.values(), key=lambda c: c.score, reverse=True)
