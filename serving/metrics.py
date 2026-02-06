from __future__ import annotations
from dataclasses import dataclass
from serving.request import Request

@dataclass
class SimulationResult:
    # Time accounting
    t_end: float
    busy_time: float

    # Finished and running requests
    finished: list[Request]
    running: list[Request]

    # Hardware utilization
