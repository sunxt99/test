from __future__ import annotations
from dataclasses import dataclass
from serving.request import Request

@dataclass
class SimulationResult:
    finished: list[Request]
    # Time accounting
    t_end: float
    busy_time: float