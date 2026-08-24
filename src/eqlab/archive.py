from __future__ import annotations
import json
from pathlib import Path
from .evolution import Candidate

class StrategyArchive:
    def __init__(self, path="data/strategy_archive.jsonl"): self.path=Path(path)
    def save(self, candidates):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open("a",encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps({"dna":c.dna.to_dict(),"fingerprint":c.dna.fingerprint(),"score":c.score,"return":c.report.total_return,"drawdown":c.report.max_drawdown})+"\n")
    def fingerprints(self):
        if not self.path.exists(): return set()
        return {json.loads(x)["fingerprint"] for x in self.path.read_text().splitlines() if x.strip()}
