from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    model_name: str
    hidden_size: int
    query_head_num: int
    kv_head_num: int
    intermediate_size: int
    layer_num: int
    head_dim: int = None
    kv_hidden_size: int = None

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.query_head_num
        self.kv_hidden_size = self.head_dim * self.kv_head_num

Mode = Literal["preempt", "reserve"]

@dataclass
class SystemConfig:
    req_type_num: int
    lam: float                 # req/s
    t_end: float               # seconds

    # QoS / priority
    priority_ratio: float = 0.05
    mode: Mode = "preempt"     # "preempt" or "reserve"

    # Capacity controls
    max_batch_hi: int = 16     # preempt mode cap_total when priority_present
    max_batch_lo: int = 256    # normal-only cap_total; also total cap in reserve mode
    reserve_hi: int = 16       # reserved slots for priority in reserve mode (hard reserve)

    # Idle batching wait (seconds)
    max_wait_s: float = 0.0       # normal-only idle wait
    max_wait_hi_s: float = 0.0    # priority-present idle wait (default 0 protects latency)

    seed: int = 0
    verbose: bool = False