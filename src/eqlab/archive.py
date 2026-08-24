from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .evolution import Candidate


class StrategyArchive:
    """Append-only development archive with DNA fingerprint deduplication."""

    def __init__(self, path: str | Path = "data/strategy_archive.jsonl") -> None:
        self.path = Path(path)

    def fingerprints(self) -> set[str]:
        if not self.path.exists():
            return set()
        fingerprints: set[str] = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                fingerprints.add(json.loads(line)["fingerprint"])
        return fingerprints

    def save(self, candidates: Iterable[Candidate]) -> int:
        existing = self.fingerprints()
        records = []
        for candidate in candidates:
            fingerprint = candidate.dna.fingerprint()
            if fingerprint in existing:
                continue
            existing.add(fingerprint)
            records.append({
                "dna": candidate.dna.to_dict(),
                "fingerprint": fingerprint,
                "score": candidate.score,
                "return": candidate.report.total_return,
                "drawdown": candidate.report.max_drawdown,
                "sharpe": candidate.report.sharpe,
                "trades": candidate.report.trades,
            })
        if not records:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return len(records)
