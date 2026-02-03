import argparse
import time

from serving.simulator import SimulatorConfig, run_simulation
from serving.metrics import summarize_metrics
from serving.config import ModelConfig

MODEL_CONFIGS = {
    0: ModelConfig("llama3-8B", 4096, 32, 8, 14336, 32),
    1: ModelConfig("qwen-14B", 5120, 40, 8, 13824, 48),
    2: ModelConfig("qwen-32B", 5120, 40, 8, 27848, 64),
    3: ModelConfig("llama3-70B", 8192, 64, 8, 28672, 80)
}

def parse_args():
    p = argparse.ArgumentParser(description="LLM decode batching simulator (no prefill) with priority QoS.")
    # Model
    p.add_argument("--model-index", type = int, required=True, help="The index of the model to run.")

    # Request
    p.add_argument("--req-type-num", type=int, required=True, help="Number of the type of requests.")
    p.add_argument("--lam", type=float, required=True, help="Arrival rate (req/s).")
    p.add_argument("--t-end", type=float, required=True, help="Simulation end time horizon (s).")

    # QoS / Priority
    p.add_argument("--priority-ratio", type=float, default=0.05, help="Fraction of priority requests (default 0.05).")
    p.add_argument("--mode", type=str, choices=["preempt", "reserve"], default="preempt",
                   help="QoS mode: preempt (shrink to max-batch-hi when priority exists) or reserve (reserve slots for priority).")

    # Capacity controls
    p.add_argument("--max-batch-hi", type=int, default=16,
                   help="Max batch size when priority exists (preempt mode). Default 16.")
    p.add_argument("--max-batch-lo", type=int, default=256,
                   help="Max batch size when NO priority exists; also total cap in reserve mode. Default 256.")
    p.add_argument("--reserve-hi", type=int, default=16,
                   help="Reserved slots for priority in reserve mode (reduces normal concurrency). Default 16.")

    # Idle batching waits
    p.add_argument("--max-wait-ms", type=float, default=0.0,
                   help="Idle batching wait for normal-only mode (ms). Default 0.")
    p.add_argument("--max-wait-hi-ms", type=float, default=0.0,
                   help="Idle batching wait when priority is waiting (ms). Default 0 (protect priority latency).")

    # Debug and Logging
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--verbose", action="store_true", help="Print verbose logs.")

    return p.parse_args()


def main():
    args = parse_args()

    model_cfg = MODEL_CONFIGS[args.model_index]

    sim_cfg = SimulatorConfig(
        req_type_num=args.req_type_num,
        lam=args.lam,
        t_end=args.t_end,
        priority_ratio=args.priority_ratio,
        mode=args.mode,
        max_batch_hi=args.max_batch_hi,
        max_batch_lo=args.max_batch_lo,
        reserve_hi=args.reserve_hi,
        max_wait_s=args.max_wait_ms / 1000.0,
        max_wait_hi_s=args.max_wait_hi_ms / 1000.0,
        seed=args.seed,
        verbose=args.verbose,
    )

    start_t = time.perf_counter()

    res = run_simulation(sim_cfg, model_cfg)
    print(summarize_metrics(res))

    end_t = time.perf_counter()
    print(f"Elapsed: {end_t - start_t:.6f}s")


if __name__ == "__main__":
    main()