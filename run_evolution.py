# run_evolution_system_v3.py
from __future__ import annotations
import ast
import json
import os
from typing import Any, List, Sequence

from utils.parse_args import parse_args

from system.config import SystemConfig, ModelConfigs

from exploration.decoder import RootInit
from exploration.evolution_pareto import InitConfig, EvoConfig, evolve
from exploration.fitness_adapter import make_fitness_fn, default_result_to_fitness
from exploration.ind_io import print_individual, save_individual_json, load_individual_json

from parallelism.pcase import (
    build_case_0, build_case_1, build_case_2, build_case_3, build_case_4,
    build_case_5, build_case_6, build_case_7, build_case_8, build_case_9,
    build_case_10, build_case_11, build_case_12, build_case_13, build_case_14,
    build_case_15
)

def result_to_fitness(sim_results: List[Any]) -> float:
    return default_result_to_fitness(sim_results)

def main() -> None:
    args = parse_args()
    sys_cfg = SystemConfig(
        hcase_index=args.hcase_index,
        pcase_index=args.pcase_index,
        req_type_num=args.req_type_num,
        req_dist=ast.literal_eval(args.req_dist),
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
    model_cfg = ModelConfigs[args.model_index]

    devices: Sequence[int] = list(range(8))

    # req_prob = [0.8, 0.2]
    req_prob = ast.literal_eval(args.req_dist)

    root_init = RootInit(
        dp_attr=[[0.0, 1.0] for _ in range(args.req_type_num)],
        pp_attr=[0, model_cfg.layer_num - 1],
        tp_attr=[0.0, 1.0],
    )

    fitness_fn = make_fitness_fn(
        sys_cfg,
        model_cfg,
        pareto_mode=True,
        req_prob=req_prob,
        hcase_idx=0,
        base_case_idx_for_init=3,
        result_to_fitness=result_to_fitness,
    )

    init_cfg = InitConfig(population_size=30,
                          # population_size=30,
                          max_depth=4,
                          max_children=8,
                          p_stop_expand=0.40,
                          # batch_size_choices=(256,))  # model0
                          batch_size_choices=(1,2,4,8,16,32,64,128,196,256,384,512))  # model0
                          # batch_size_choices=(1,2,4,8,16,32,48,64,96,128,160,196)) # model1
                          # batch_size_choices=(1,2,4,8,16,32,48,64,96,128,160,196)) # model2
                          # batch_size_choices=(1,2,4,8,16,32,48,64,96,112,128)) # model3

    evo_cfg = EvoConfig(generations=5,
                        #generations=10,
                        elite_size=8,
                        # elite_size=5,
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
        10: build_case_10,
        11: build_case_11,
        12: build_case_12,
        13: build_case_13,
        14: build_case_14,
        15: build_case_15,
    }

    # pop_seed_indexes = [1,2,3,4]
    # pop_seed_indexes = [3,4]
    pop_seed_indexes = [12]
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

    # Dump Pareto front (rank-0) objectives to jsonl if requested
    if args.out:
        if os.path.exists(args.out):
            os.remove(args.out)

        pareto_front = [ind for ind in pop if getattr(ind, "pareto_rank", None) == 0]
        for ind in pareto_front:
            if getattr(ind, "objectives", None) is None:
                continue
            T, L = float(ind.objectives[0]), float(ind.objectives[1])
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"T": T, 
                                    "L": L}, ensure_ascii=False) + "\n")

    # print("\n Best")
    # print("Best fitness:", best.fitness)
    # print("Best Throughput:", best.throughput)
    # print("Best Latency:", best.latency)
    # print("Best uid:", best.uid)
    # print(best.topology.nodes)
    # print("dp_attr:", best.attrs.dp_attr)
    # print("pp_attr:", best.attrs.pp_attr)
    # print("tp_attr:", best.attrs.tp_attr)
    # print("xp_attr:", best.attrs.xp_attr)
    # print(best.device_assign.leaf_to_devices)
    # print("\n")

    # print example
    print_individual(best)

    # io Example
    # save_individual_json(best, "best_individual.json")
    # best2 = load_individual_json("best_individual.json")
    # print("\n--- Reloaded ---\n")
    # print_individual(best2)


if __name__ == "__main__":
    main()