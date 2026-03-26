# run_evolution_system_v3.py
from __future__ import annotations
import ast
import json
import os
from typing import Any, List, Sequence

from utils.parse_args import parse_args

from system.config import SystemConfig, ModelConfigs
from hardware.htree import HardwareTree

from exploration.decoder import RootInit
from exploration.evolution_pareto import InitConfig, EvoConfig, evolve
from exploration.fitness_adapter import make_fitness_fn, default_result_to_fitness
from exploration.ind_io import print_individual,log_individual_json, save_individual_json, load_individual_json

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

    htree = HardwareTree(args.hcase_index)
    devices: Sequence[int] = [int(d.idx) for d in htree.devices]
    device_type_by_id = {int(d.idx): str(d.meta.get("type", d.name)) for d in htree.devices}

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
        pcase_idx_for_init=3,
        result_to_fitness=result_to_fitness,
    )

    init_cfg = InitConfig(population_size=80,
                          # population_size=30,
                          max_depth=4,
                          max_children=8,
                          p_stop_expand=0.40,

                          p_pattern_seed_init = 0.5,
                          p_stratified_init = 0.3,
                          p_random_init = 0.2,

                          # batch_size_choices=(256,))  # model0
                          batch_size_choices=(1,2,4,8,16,32,64,128,160,196,256,320,352,384,416,448,480,512))  # model0
                          # batch_size_choices=(1,2,4,8,16,32,48,64,96,128,160,196)) # model1
                          # batch_size_choices=(1,2,4,8,16,32,48,64,96,128,160,196)) # model2
                          # batch_size_choices=(1,2,4,8,16,32,48,64,96,112,128)) # model3


    evo_cfg = EvoConfig(generations=3,
                        elite_size=15,

                        p_rewrite_mut=0.5,
                        p_numeric_mut=0.5,
                        p_mapping_refine_mut=0.00,

                        p_skeleton_expand=0.25,
                        p_local_refine=0.30,
                        p_relabel=0.15,
                        p_repartition=0.20,
                        p_rollback=0.10,
                        rewrite_max_steps=4,

                        enable_cache=True,
                        # enable_subgraph_batch_mut=True,
                        enable_subgraph_batch_mut=False,

                        subgraph_batch_max_mutated=1,
                        numeric_mutation_max_targets=2,
                        )

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

    # pop_seed_indexes = [3,4]
    # pop_seed_indexes = [0,1,4,12]
    pop_seed_indexes = []
    pop_seed_roots = []
    for i in pop_seed_indexes:
        if i not in builders:
            continue
        root, _leaves = builders[i](args.req_type_num, model_cfg.layer_num)
        pop_seed_roots.append(root)

    # dse scatter point
    # if args.dse_out:
    #     if os.path.exists(args.dse_out):
    #         os.remove(args.dse_out)

    # DSE 主函数
    best, pop = evolve(
        init_cfg,
        evo_cfg,
        req_type_num=args.req_type_num,
        devices=devices,
        root_init=root_init,
        device_type_by_id=device_type_by_id,
        fitness_fn=fitness_fn,
        with_pop_seeds=True,
        pop_seed_roots=pop_seed_roots,
        attach_hardware_leaves=True,
        random_seed=42,
        dse_out=args.dse_out,
    )

    # Dump Pareto front (rank-0) objectives to jsonl if requested
    if args.out:
        # if os.path.exists(args.out):
        #     os.remove(args.out)
        pareto_front = [ind for ind in pop if getattr(ind, "pareto_rank", None) == 0]
        for ind in pareto_front:
            if getattr(ind, "objectives", None) is None:
                continue
            log_individual_json(ind, args.out)



    # print example
    print_individual(best)

    # io Example
    # save_individual_json(best, "best_individual.json")
    # best2 = load_individual_json("best_individual.json")
    # print("\n--- Reloaded ---\n")
    # print_individual(best2)


if __name__ == "__main__":
    main()