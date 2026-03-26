from __future__ import annotations

import argparse
import ast
import pathlib
import random
from typing import Any, List, Sequence

from hardware.htree import HardwareTree
from system.config import ModelConfigs, SystemConfig

from exploration.decoder import RootInit
from exploration.fitness_adapter import default_result_to_fitness, make_fitness_fn
from exploration.ind_io import load_individual_json
from exploration.rewrite_debugger import (
    debug_rewrite_candidates,
    debug_rewrite_multistep,
    dump_candidate_individuals,
    dump_multistep_individuals,
    format_multistep_report,
    format_report,
    save_multistep_report_json,
    save_report_json,
)
from exploration.rewrite_mechanism import RewriteFamily


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dedicated rewrite debugger for Individual -> match/rewrite/evaluate.")

    p.add_argument("--individual-json", type=str, required=True, help="Path to the input individual json.")
    p.add_argument("--model-index", type=int, required=True)
    p.add_argument("--hcase-index", type=int, required=True)
    p.add_argument("--pcase-index", type=int, required=True, help="Only used to initialize simulator scaffolding.")

    p.add_argument("--t-end", type=float, required=True)
    p.add_argument("--req-type-num", type=int, required=True)
    p.add_argument("--req-dist", type=str, required=True)
    p.add_argument("--lam", type=float, default=100.0)
    p.add_argument("--priority-ratio", type=float, default=0.0)
    p.add_argument("--mode", type=str, choices=["preempt", "reserve"], default="preempt")
    p.add_argument("--max-batch-lo", type=int, default=512)
    p.add_argument("--max-batch-hi", type=int, default=32)
    p.add_argument("--reserve-hi", type=int, default=32)
    p.add_argument("--max-wait-ms", type=float, default=0.0)
    p.add_argument("--max-wait-hi-ms", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--debug-mode", type=str, choices=["single", "multistep"], default="single")
    p.add_argument("--rewrite-max-steps", type=int, default=4, help="Only used by multi-step debug mode.")
    p.add_argument("--family", type=str, default=None, choices=[x.value for x in RewriteFamily])
    p.add_argument("--pattern", action="append", default=[], help="Limit to one or more exact pattern names. Only used by single-step debug.")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--include-individual-text", action="store_true")
    p.add_argument("--save-dir", type=str, default=None, help="Optional directory to dump candidate individuals.")
    p.add_argument("--improved-only", action="store_true", help="Only dump candidates/steps that dominate the baseline.")
    p.add_argument("--report-json", type=str, default=None, help="Optional JSON report output path.")

    p.add_argument("--p-skeleton-expand", type=float, default=0.25, help="Multi-step family weight.")
    p.add_argument("--p-local-refine", type=float, default=0.30, help="Multi-step family weight.")
    p.add_argument("--p-relabel", type=float, default=0.15, help="Multi-step family weight.")
    p.add_argument("--p-repartition", type=float, default=0.20, help="Multi-step family weight.")
    p.add_argument("--p-rollback", type=float, default=0.10, help="Multi-step family weight.")
    return p.parse_args()


def result_to_fitness(sim_results: List[Any]) -> float:
    return default_result_to_fitness(sim_results)


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

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
        verbose=False,
    )
    model_cfg = ModelConfigs[args.model_index]

    htree = HardwareTree(args.hcase_index)
    devices: Sequence[int] = [int(d.idx) for d in htree.devices]
    device_type_by_id = {int(d.idx): str(d.meta.get("type", d.name)) for d in htree.devices}
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
        hcase_idx=args.hcase_index,
        pcase_idx_for_init=args.pcase_index,
        result_to_fitness=result_to_fitness,
    )

    ind = load_individual_json(args.individual_json)
    ind.devices = list(devices)

    chosen_family = RewriteFamily(args.family) if args.family else None

    if args.debug_mode == "single":
        report = debug_rewrite_candidates(
            ind,
            fitness_fn=fitness_fn,
            root_init=root_init,
            device_type_by_id=device_type_by_id,
            family=chosen_family,
            pattern_names=args.pattern or None,
            attach_hardware_leaves=True,
        )

        print(format_report(report, topk=args.topk, include_individual_text=args.include_individual_text))

        if args.report_json:
            save_report_json(report, pathlib.Path(args.report_json))
            print(f"\n[report_json] {args.report_json}")

        if args.save_dir:
            paths = dump_candidate_individuals(
                report,
                pathlib.Path(args.save_dir),
                topk=args.topk,
                improved_only=args.improved_only,
            )
            print(f"[saved_candidates] {len(paths)}")
            for p in paths:
                print(f"  - {p}")
        return

    family_weights = {
        RewriteFamily.SKELETON_EXPANSION: float(args.p_skeleton_expand),
        RewriteFamily.LOCAL_REFINEMENT: float(args.p_local_refine),
        RewriteFamily.RELABEL: float(args.p_relabel),
        RewriteFamily.REPARTITION: float(args.p_repartition),
        RewriteFamily.ROLLBACK: float(args.p_rollback),
    }

    report = debug_rewrite_multistep(
        ind,
        fitness_fn=fitness_fn,
        root_init=root_init,
        device_type_by_id=device_type_by_id,
        rewrite_max_steps=args.rewrite_max_steps,
        family=chosen_family,
        family_weights=family_weights,
        seed=args.seed,
        attach_hardware_leaves=True,
    )

    print(format_multistep_report(report, include_individual_text=args.include_individual_text))

    if args.report_json:
        save_multistep_report_json(report, pathlib.Path(args.report_json))
        print(f"\n[report_json] {args.report_json}")

    if args.save_dir:
        paths = dump_multistep_individuals(
            report,
            pathlib.Path(args.save_dir),
            improved_only=args.improved_only,
        )
        print(f"[saved_candidates] {len(paths)}")
        for p in paths:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
