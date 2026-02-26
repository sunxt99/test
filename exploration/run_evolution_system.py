# run_evolution_system_v3.py
from __future__ import annotations

from typing import Any, List, Sequence
import argparse

from system.config import SystemConfig, ModelConfig

from exploration.decoder import RootInit

pareto_mode = False
if pareto_mode:
    from exploration.evolution_pareto import InitConfig, EvoConfig, evolve
else:
    from exploration.evolution import InitConfig, EvoConfig, evolve

from exploration.fitness_adapter import make_fitness_fn, default_result_to_fitness
from exploration.ind_io import print_individual, save_individual_json, load_individual_json

from parallelism.pcase import (
    build_case_0, build_case_1, build_case_2, build_case_3, build_case_4,
    build_case_5, build_case_6, build_case_7, build_case_8, build_case_9,
)


MODEL_CONFIGS = {
    0: ModelConfig("llama3-8B", 4096, 32, 8, 14336, 32),
    1: ModelConfig("qwen-14B", 5120, 40, 8, 13824, 48),
    2: ModelConfig("qwen-32B", 5120, 40, 8, 27848, 64),
    3: ModelConfig("llama3-70B", 8192, 64, 8, 28672, 80)
}

def result_to_fitness(sim_results: List[Any]) -> float:
    return default_result_to_fitness(sim_results)

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
    p.add_argument("--max-batch-hi", type=int, default=32,
                   help="Max batch size when priority exists (preempt mode). Default 16.")
    p.add_argument("--max-batch-lo", type=int, default=128,
                   help="Max batch size when NO priority exists; also total cap in reserve mode. Default 256.")
    p.add_argument("--reserve-hi", type=int, default=32,
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

def main() -> None:
    args = parse_args()
    sys_cfg = SystemConfig(
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
    model_cfg = MODEL_CONFIGS[args.model_index]

    devices: Sequence[int] = list(range(8))
    req_prob = [0.8, 0.2]

    root_init = RootInit(
        dp_attr=[[0.0, 1.0] for _ in range(args.req_type_num)],
        pp_attr=[0, model_cfg.layer_num - 1],
        tp_attr=[0.0, 1.0],
    )

    fitness_fn = make_fitness_fn(
        sys_cfg,
        model_cfg,
        pareto_mode=pareto_mode,
        req_prob=req_prob,
        hcase_idx=0,
        base_case_idx_for_init=3,
        result_to_fitness=result_to_fitness,
    )

    init_cfg = InitConfig(population_size=30,
                          max_depth=4,
                          max_children=8,
                          p_stop_expand=0.40)

    evo_cfg = EvoConfig(generations=4,
                        #generations=20,
                        elite_size=5,
                        p_topology_mut=0.15,
                        p_numeric_mut=0.50,
                        p_device_mut=0.35,
                        # p_topology_mut=0.55,
                        # p_numeric_mut=0.30,
                        # p_device_mut=0.15,
                        enable_cache=True)

    # 根据经验进行种群初值
    builders = {
        0: build_case_0,
        1: build_case_1,
        2: build_case_2,
        3: build_case_3,
        4: build_case_4,
        5: build_case_5,
        6: build_case_6,
        7: build_case_7,
        8: build_case_8,
        9: build_case_9,
    }
    pop_seed_indexes = [4]
    # pop_seed_indexes = []
    pop_seed_roots = []
    for i in pop_seed_indexes:
        if i not in builders:
            continue
        root, _leaves = builders[i](args.req_type_num, model_cfg.layer_num)
        pop_seed_roots.append(root)


    # DSE 主函数
    best, pop = evolve(
        init_cfg,
        evo_cfg,
        req_type_num=args.req_type_num,
        devices=devices,
        root_init=root_init,
        fitness_fn=fitness_fn,
        with_pop_seeds=True,
        pop_seed_roots=pop_seed_roots,
        attach_hardware_leaves=True,
        random_seed=42,
    )

    print("\n Best")
    # print("Best fitness:", best.fitness)
    # print("Best uid:", best.uid)
    # print(best.topology.nodes)
    print("dp_attr:", best.attrs.dp_attr)
    print("pp_attr:", best.attrs.pp_attr)
    print("tp_attr:", best.attrs.tp_attr)
    print("xp_attr:", best.attrs.xp_attr)
    # print(best.device_assign.leaf_to_devices)
    print("\n")

    # print example
    print_individual(best)

    # io Example
    # save_individual_json(best, "best_individual.json")
    # best2 = load_individual_json("best_individual.json")
    # print("\n--- Reloaded ---\n")
    # print_individual(best2)


if __name__ == "__main__":
    main()
