from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .backtest import PerformanceReport
from .validation import RobustnessReport

class ReportWriter:
    def write(self, path: str | Path, performance: PerformanceReport, robustness: RobustnessReport | None = None) -> Path:
        payload={"performance":asdict(performance),"robustness":asdict(robustness) if robustness else None}
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8"); return path
