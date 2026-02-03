from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .request import Request

@dataclass
class SimulationResult:
    finished: list[Request]
    # Time accounting
    t_end: float
    busy_time: float


def _pct(x, p):
    if len(x) == 0:
        return float("nan")
    return float(np.percentile(x, p))


def _fmt_ms(x_s: float) -> str:
    if np.isnan(x_s):
        return "nan"
    return f"{x_s * 1000.0:.3f}"


def _summarize_group(name: str, reqs: list[Request], t_end: float) -> str:
    n = len(reqs)
    if n == 0:
        return f"[{name}] finished=0\n"

    lat = np.array([r.finish_time - r.arrival_time for r in reqs], dtype=float)
    qd = np.array([r.start_time - r.arrival_time for r in reqs], dtype=float)
    st = np.array([r.finish_time - r.start_time for r in reqs], dtype=float)
    prompt = np.array([r.prompt_tokens for r in reqs], dtype=float)
    gen_tgt = np.array([r.target_gen_tokens for r in reqs], dtype=float)
    tpot = np.array([r.target_gen_tokens/(r.finish_time - r.start_time)/1000 for r in reqs], dtype=float)


    lines = []
    lines.append(f"[{name}] finished={n}, throughput={n / t_end:.6f} req/s")
    lines.append(f"  prompt_tokens mean={float(np.mean(prompt)):.3f}, p50={_pct(prompt, 50):.3f}, p95={_pct(prompt, 95):.3f}")
    lines.append(f"  target_gen_tokens mean={float(np.mean(gen_tgt)):.3f}, p50={_pct(gen_tgt, 50):.3f}, p95={_pct(gen_tgt, 95):.3f}")
    lines.append("  latency_ms: " +
                 f"mean={_fmt_ms(float(np.mean(lat)))}, p50={_fmt_ms(_pct(lat, 50))}, p95={_fmt_ms(_pct(lat, 95))}, p99={_fmt_ms(_pct(lat, 99))}")
    lines.append("  queueing_ms: " +
                 f"mean={_fmt_ms(float(np.mean(qd)))}, p50={_fmt_ms(_pct(qd, 50))}, p95={_fmt_ms(_pct(qd, 95))}, p99={_fmt_ms(_pct(qd, 99))}")
    lines.append("  service_ms: " +
                 f"mean={_fmt_ms(float(np.mean(st)))}, p50={_fmt_ms(_pct(st, 50))}, p95={_fmt_ms(_pct(st, 95))}, p99={_fmt_ms(_pct(st, 99))}")
    lines.append("  TPOT_token/s: " +
                 f"mean={_fmt_ms(float(np.mean(tpot)))}, p50={_fmt_ms(_pct(tpot, 100-50))}, p95={_fmt_ms(_pct(tpot, 100-95))}, p99={_fmt_ms(_pct(tpot, 100-99))}")
    lines.append("")
    return "\n".join(lines)


def summarize_metrics(res: SimulationResult) -> str:
    finished = res.finished
    n = len(finished)

    util = res.busy_time / res.t_end if res.t_end > 0 else float("nan")

    prio = [r for r in finished if getattr(r, "is_priority", False)]
    normal = [r for r in finished if not getattr(r, "is_priority", False)]

    lines = []
    lines.append("=== Simulation Summary ===")
    lines.append(f"horizon={res.t_end:.6f}s, busy_time={res.busy_time:.6f}s, utilization={util:.6f}")
    lines.append(f"finished_total={n}, throughput_total={n / res.t_end:.6f} req/s")
    lines.append("")
    lines.append(_summarize_group("ALL", finished, res.t_end))
    lines.append(_summarize_group("PRIORITY", prio, res.t_end))
    lines.append(_summarize_group("NORMAL", normal, res.t_end))
    return "\n".join(lines)
