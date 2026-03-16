from __future__ import annotations
from typing import List
import numpy as np

from serving.request import Request
from serving.metrics import SimulationResult


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

    lat = np.array([r.finish_time + r.accum_delay_time - r.arrival_time for r in reqs], dtype=float)
    qd = np.array([r.start_time - r.arrival_time for r in reqs], dtype=float)
    st = np.array([r.finish_time + r.accum_delay_time - r.start_time for r in reqs], dtype=float)
    prompt = np.array([r.prompt_tokens for r in reqs], dtype=float)
    gen_tgt = np.array([r.target_gen_tokens for r in reqs], dtype=float)
    usr_throughput = np.array([r.target_gen_tokens/(r.finish_time + r.accum_delay_time - r.start_time)/1000 for r in reqs], dtype=float)
    tpot = np.array([(r.finish_time + r.accum_delay_time - r.start_time) / r.target_gen_tokens for r in reqs], dtype=float)

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
    lines.append("  user_throughput_token/s: " +
                 f"mean={_fmt_ms(float(np.mean(usr_throughput)))}, p10={_fmt_ms(_pct(usr_throughput, 10))}, p50={_fmt_ms(_pct(usr_throughput, 50))}, p90={_fmt_ms(_pct(usr_throughput, 90))}")
    lines.append("  TPOT_ms: " +
                 f"mean={_fmt_ms(float(np.mean(tpot)))}, p90={_fmt_ms(_pct(tpot, 90))}, p95={_fmt_ms(_pct(tpot, 95))}, p99={_fmt_ms(_pct(tpot, 99))}")
    lines.append("")
    return "\n".join(lines)


def summarize_metrics(res_list: List[SimulationResult]) -> str:
    # processed 包括 running 和 finished
    processed = []
    finished = []
    for res in res_list:
        finished.extend(res.finished)
        processed.extend(res.finished)
        processed.extend(res.running)
    finished_num = len(finished)

    busy_time = np.mean([res.busy_time for res in res_list])
    t_end = np.mean([res.t_end for res in res_list])

    util = busy_time / t_end if t_end > 0 else float("nan")

    # prio = [r for r in finished if getattr(r, "is_priority", False)]
    # normal = [r for r in finished if not getattr(r, "is_priority", False)]

    gen_tokens = [r.gen_tokens for r in processed]
    total_gen_tokens = np.sum(gen_tokens)

    lines = []
    lines.append("=== Simulation Summary ===")
    lines.append(f"horizon={t_end:.6f}s, busy_time={busy_time:.6f}s, utilization={util:.6f}")
    lines.append(f"finished_total={finished_num}, throughput_total={finished_num / t_end:.6f} req/s")
    lines.append(f"mean_gen_tokens={np.mean(gen_tokens)} token")
    lines.append(f"throughput={total_gen_tokens/t_end} token/s")
    lines.append("")
    lines.append(_summarize_group("ALL", finished, float(t_end)))
    # lines.append(_summarize_group("PRIORITY", prio, float(t_end)))
    # lines.append(_summarize_group("NORMAL", normal, float(t_end)))

    return "\n".join(lines)


def _summarize_group_data(name: str, reqs: list[Request], t_end: float):
    n = len(reqs)
    if n == 0:
        return f"[{name}] finished=0\n"

    lat = np.array([r.finish_time + r.accum_delay_time - r.arrival_time for r in reqs], dtype=float)
    qd = np.array([r.start_time - r.arrival_time for r in reqs], dtype=float)
    st = np.array([r.finish_time + r.accum_delay_time - r.start_time for r in reqs], dtype=float)
    prompt = np.array([r.prompt_tokens for r in reqs], dtype=float)
    gen_tgt = np.array([r.target_gen_tokens for r in reqs], dtype=float)
    usr_throughput = np.array([r.target_gen_tokens/(r.finish_time + r.accum_delay_time - r.start_time)/1000 for r in reqs], dtype=float)
    tpot = np.array([(r.finish_time + r.accum_delay_time - r.start_time) / r.target_gen_tokens for r in reqs], dtype=float)

    # lines = []
    # lines.append(f"[{name}] finished={n}, throughput={n / t_end:.6f} req/s")
    # lines.append(f"  prompt_tokens mean={float(np.mean(prompt)):.3f}, p50={_pct(prompt, 50):.3f}, p95={_pct(prompt, 95):.3f}")
    # lines.append(f"  target_gen_tokens mean={float(np.mean(gen_tgt)):.3f}, p50={_pct(gen_tgt, 50):.3f}, p95={_pct(gen_tgt, 95):.3f}")
    # lines.append("  latency_ms: " +
    #              f"mean={_fmt_ms(float(np.mean(lat)))}, p50={_fmt_ms(_pct(lat, 50))}, p95={_fmt_ms(_pct(lat, 95))}, p99={_fmt_ms(_pct(lat, 99))}")
    # lines.append("  queueing_ms: " +
    #              f"mean={_fmt_ms(float(np.mean(qd)))}, p50={_fmt_ms(_pct(qd, 50))}, p95={_fmt_ms(_pct(qd, 95))}, p99={_fmt_ms(_pct(qd, 99))}")
    # lines.append("  service_ms: " +
    #              f"mean={_fmt_ms(float(np.mean(st)))}, p50={_fmt_ms(_pct(st, 50))}, p95={_fmt_ms(_pct(st, 95))}, p99={_fmt_ms(_pct(st, 99))}")
    # lines.append("  user_throughput_token/s: " +
    #              f"mean={_fmt_ms(float(np.mean(usr_throughput)))}, p10={_fmt_ms(_pct(usr_throughput, 10))}, p50={_fmt_ms(_pct(usr_throughput, 50))}, p90={_fmt_ms(_pct(usr_throughput, 90))}")
    # lines.append("  TPOT_ms: " +
    #              f"mean={_fmt_ms(float(np.mean(tpot)))}, p10={_fmt_ms(_pct(tpot, 10))}, p50={_fmt_ms(_pct(tpot, 50))}, p90={_fmt_ms(_pct(tpot, 90))}")
    # lines.append("")
    # return "\n".join(lines)


    # P90/95/99 latency
    # return _fmt_ms(_pct(lat, 90))

    # P90/95/99 TPOT
    return float(_fmt_ms(_pct(tpot, 99)))

def summarize_metrics_data(res_list: List[SimulationResult]):
    # processed 包括 running 和 finished
    processed = []
    finished = []
    for res in res_list:
        finished.extend(res.finished)
        processed.extend(res.finished)
        processed.extend(res.running)
    finished_num = len(finished)

    busy_time = np.mean([res.busy_time for res in res_list])
    t_end = np.mean([res.t_end for res in res_list])

    util = busy_time / t_end if t_end > 0 else float("nan")

    # prio = [r for r in finished if getattr(r, "is_priority", False)]
    # normal = [r for r in finished if not getattr(r, "is_priority", False)]

    gen_tokens = np.sum([r.gen_tokens for r in processed], dtype=float)

    # lines = []
    # lines.append("=== Simulation Summary ===")
    # lines.append(f"horizon={t_end:.6f}s, busy_time={busy_time:.6f}s, utilization={util:.6f}")
    # lines.append(f"finished_total={finished_num}, throughput_total={finished_num / t_end:.6f} req/s")
    # lines.append(f"throughput={gen_tokens/t_end} token/s")
    # lines.append("")
    # lines.append(_summarize_group("ALL", finished, float(t_end)))

    return [gen_tokens/t_end, _summarize_group_data("ALL", finished, float(t_end))]