from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import json
import pathlib
import random

from exploration.decoder import RootInit, decode_to_root
from exploration.ind_io import format_individual, save_individual_json
from exploration.individual import Individual
from exploration.rewrite_mechanism import (
    PatternSpec,
    RewriteFamily,
    SymbolicNode,
    _get_node_by_path,
    _replace_node_by_path,
    default_patterns,
    has_open_nodes,
    individual_to_symbolic,
    is_materializable,
    symbolic_to_individual,
)


@dataclass
class BaselineEvaluation:
    throughput: Optional[float]
    latency: Optional[float]
    objectives: Optional[Tuple[float, float]]
    decoded: bool
    error: Optional[str] = None


@dataclass
class RewriteMatch:
    pattern: PatternSpec
    path: Tuple[int, ...]
    before_node_pretty: str
    before_root_pretty: str

    @property
    def family(self) -> RewriteFamily:
        return self.pattern.family

    @property
    def path_str(self) -> str:
        return "root" if not self.path else "root." + ".".join(str(x) for x in self.path)


@dataclass
class RewriteCandidateResult:
    match: RewriteMatch
    matched: bool = False
    applied: bool = False
    materializable: bool = False
    legal: bool = False
    decoded: bool = False
    throughput: Optional[float] = None
    latency: Optional[float] = None
    objectives: Optional[Tuple[float, float]] = None
    error: Optional[str] = None
    dominates_baseline: Optional[bool] = None
    dominated_by_baseline: Optional[bool] = None
    equal_objectives: Optional[bool] = None
    throughput_delta: Optional[float] = None
    latency_delta: Optional[float] = None
    after_node_pretty: Optional[str] = None
    after_root_pretty: Optional[str] = None
    candidate_individual: Optional[Individual] = None


@dataclass
class RewriteDebugReport:
    baseline: BaselineEvaluation
    matches: List[RewriteMatch] = field(default_factory=list)
    candidates: List[RewriteCandidateResult] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "mode": "single",
            "baseline": {
                "decoded": self.baseline.decoded,
                "throughput": self.baseline.throughput,
                "latency": self.baseline.latency,
                "objectives": list(self.baseline.objectives) if self.baseline.objectives is not None else None,
                "error": self.baseline.error,
            },
            "match_count": len(self.matches),
            "candidate_count": len(self.candidates),
            "improved_count": sum(1 for c in self.candidates if c.dominates_baseline),
            "candidates": [
                {
                    "pattern": c.match.pattern.name,
                    "family": c.match.family.value,
                    "path": list(c.match.path),
                    "path_str": c.match.path_str,
                    "matched": c.matched,
                    "applied": c.applied,
                    "materializable": c.materializable,
                    "legal": c.legal,
                    "decoded": c.decoded,
                    "throughput": c.throughput,
                    "latency": c.latency,
                    "objectives": list(c.objectives) if c.objectives is not None else None,
                    "throughput_delta": c.throughput_delta,
                    "latency_delta": c.latency_delta,
                    "dominates_baseline": c.dominates_baseline,
                    "dominated_by_baseline": c.dominated_by_baseline,
                    "equal_objectives": c.equal_objectives,
                    "error": c.error,
                    "before_node": c.match.before_node_pretty,
                    "after_node": c.after_node_pretty,
                    "before_root": c.match.before_root_pretty,
                    "after_root": c.after_root_pretty,
                }
                for c in self.candidates
            ],
        }


@dataclass
class MultiStepAttribution:
    throughput: Optional[float] = None
    latency: Optional[float] = None
    objectives: Optional[Tuple[float, float]] = None
    throughput_delta_from_prev: Optional[float] = None
    latency_delta_from_prev: Optional[float] = None
    throughput_delta_from_baseline: Optional[float] = None
    latency_delta_from_baseline: Optional[float] = None
    dominates_prev: Optional[bool] = None
    dominates_baseline: Optional[bool] = None
    dominated_by_prev: Optional[bool] = None
    dominated_by_baseline: Optional[bool] = None


@dataclass
class MultiStepTrace:
    step: int
    requested_family: Optional[RewriteFamily]
    effective_family: Optional[RewriteFamily]
    used_family_fallback: bool = False
    applied: bool = False
    pattern_name: Optional[str] = None
    path: Optional[Tuple[int, ...]] = None
    before_node_pretty: Optional[str] = None
    after_node_pretty: Optional[str] = None
    before_root_pretty: Optional[str] = None
    after_root_pretty: Optional[str] = None
    materializable: bool = False
    legal: bool = False
    decoded: bool = False
    attribution: Optional[MultiStepAttribution] = None
    error: Optional[str] = None
    candidate_individual: Optional[Individual] = None

    @property
    def path_str(self) -> Optional[str]:
        if self.path is None:
            return None
        return "root" if not self.path else "root." + ".".join(str(x) for x in self.path)


@dataclass
class MultiStepRewriteReport:
    baseline: BaselineEvaluation
    rewrite_max_steps: int
    traces: List[MultiStepTrace] = field(default_factory=list)
    final_materializable: bool = False
    final_legal: bool = False
    final_decoded: bool = False
    final_objectives: Optional[Tuple[float, float]] = None
    final_individual: Optional[Individual] = None
    stopped_reason: Optional[str] = None

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "mode": "multistep",
            "rewrite_max_steps": self.rewrite_max_steps,
            "baseline": {
                "decoded": self.baseline.decoded,
                "throughput": self.baseline.throughput,
                "latency": self.baseline.latency,
                "objectives": list(self.baseline.objectives) if self.baseline.objectives is not None else None,
                "error": self.baseline.error,
            },
            "trace_count": len(self.traces),
            "final_materializable": self.final_materializable,
            "final_legal": self.final_legal,
            "final_decoded": self.final_decoded,
            "final_objectives": list(self.final_objectives) if self.final_objectives is not None else None,
            "stopped_reason": self.stopped_reason,
            "traces": [
                {
                    "step": t.step,
                    "requested_family": t.requested_family.value if t.requested_family is not None else None,
                    "effective_family": t.effective_family.value if t.effective_family is not None else None,
                    "used_family_fallback": t.used_family_fallback,
                    "applied": t.applied,
                    "pattern": t.pattern_name,
                    "path": list(t.path) if t.path is not None else None,
                    "path_str": t.path_str,
                    "before_node": t.before_node_pretty,
                    "after_node": t.after_node_pretty,
                    "before_root": t.before_root_pretty,
                    "after_root": t.after_root_pretty,
                    "materializable": t.materializable,
                    "legal": t.legal,
                    "decoded": t.decoded,
                    "attribution": None if t.attribution is None else {
                        "throughput": t.attribution.throughput,
                        "latency": t.attribution.latency,
                        "objectives": list(t.attribution.objectives) if t.attribution.objectives is not None else None,
                        "throughput_delta_from_prev": t.attribution.throughput_delta_from_prev,
                        "latency_delta_from_prev": t.attribution.latency_delta_from_prev,
                        "throughput_delta_from_baseline": t.attribution.throughput_delta_from_baseline,
                        "latency_delta_from_baseline": t.attribution.latency_delta_from_baseline,
                        "dominates_prev": t.attribution.dominates_prev,
                        "dominates_baseline": t.attribution.dominates_baseline,
                        "dominated_by_prev": t.attribution.dominated_by_prev,
                        "dominated_by_baseline": t.attribution.dominated_by_baseline,
                    },
                    "error": t.error,
                }
                for t in self.traces
            ],
        }


DEFAULT_FAMILY_WEIGHTS: Dict[RewriteFamily, float] = {
    RewriteFamily.SKELETON_EXPANSION: 0.25,
    RewriteFamily.LOCAL_REFINEMENT: 0.30,
    RewriteFamily.RELABEL: 0.15,
    RewriteFamily.REPARTITION: 0.20,
    RewriteFamily.ROLLBACK: 0.10,
}


def _build_device_type_to_ids(devices: Sequence[int], device_type_by_id: Optional[Dict[int, str]]) -> Dict[str, List[int]]:
    dtype_map = device_type_by_id or {int(d): str(int(d)) for d in devices}
    out: Dict[str, List[int]] = {}
    for did in devices:
        out.setdefault(str(dtype_map[int(did)]), []).append(int(did))
    return out


def _as_objectives(v: Any) -> Tuple[float, float]:
    if v is None:
        raise ValueError("fitness_result is None")

    latency_penalty = 1e5

    if isinstance(v, (tuple, list)) and len(v) >= 1:
        throughput = float(v[0])
        latency = None
        for item in list(v)[1:]:
            if isinstance(item, (int, float)):
                latency = float(item)
                break
        if latency is None:
            latency = latency_penalty
        return throughput, latency

    if isinstance(v, dict):
        lower = {str(k).lower(): val for k, val in v.items()}

        def pick(*names: str) -> Any:
            for name in names:
                if name in lower:
                    return lower[name]
            return None

        throughput = pick("t", "throughput", "tps", "qps", "req_s", "req_per_s", "requests_per_s")
        latency = pick("l", "latency", "p95_latency", "p99_latency", "p90_latency", "tail_latency", "p95", "p99", "lat_ms", "lat_s")
        if throughput is None:
            raise ValueError(f"Cannot parse throughput from dict keys={sorted(lower.keys())}")
        return float(throughput), float(latency_penalty if latency is None else latency)

    if isinstance(v, (int, float)):
        return float(v), latency_penalty

    raise ValueError(f"Unsupported fitness_result type: {type(v)}")


def _dominates(lhs: Tuple[float, float], rhs: Tuple[float, float]) -> bool:
    lhs_t, lhs_l = float(lhs[0]), float(lhs[1])
    rhs_t, rhs_l = float(rhs[0]), float(rhs[1])
    return (lhs_t >= rhs_t and lhs_l <= rhs_l) and (lhs_t > rhs_t or lhs_l < rhs_l)


def evaluate_individual(
    ind: Individual,
    *,
    fitness_fn: Callable[[Any, int], Any],
    root_init: RootInit,
    attach_hardware_leaves: bool = True,
) -> BaselineEvaluation:
    try:
        root = decode_to_root(ind, root_init, attach_hardware_leaves=attach_hardware_leaves)
        obj = _as_objectives(fitness_fn(root, int(getattr(ind, "batch_size", 1))))
        return BaselineEvaluation(
            throughput=float(obj[0]),
            latency=float(obj[1]),
            objectives=obj,
            decoded=True,
            error=None,
        )
    except Exception as e:
        return BaselineEvaluation(
            throughput=None,
            latency=None,
            objectives=None,
            decoded=False,
            error=str(e),
        )


def enumerate_rewrite_matches(
    ind: Individual,
    *,
    device_type_by_id: Optional[Dict[int, str]] = None,
    patterns: Optional[Sequence[PatternSpec]] = None,
    family: Optional[RewriteFamily] = None,
    pattern_names: Optional[Sequence[str]] = None,
) -> List[RewriteMatch]:
    pats = list(patterns or default_patterns())
    if family is not None:
        pats = [p for p in pats if p.family == family]
    if pattern_names:
        wanted = set(pattern_names)
        pats = [p for p in pats if p.name in wanted]

    sym_root = individual_to_symbolic(
        ind,
        device_type_by_id=device_type_by_id or {int(d): str(int(d)) for d in ind.devices},
    )

    out: List[RewriteMatch] = []

    def dfs(node: SymbolicNode, path: List[int]) -> None:
        for pat in pats:
            if pat.matches(node):
                out.append(
                    RewriteMatch(
                        pattern=pat,
                        path=tuple(path),
                        before_node_pretty=node.pretty(),
                        before_root_pretty=sym_root.pretty(),
                    )
                )
        for idx, child in enumerate(node.children):
            path.append(idx)
            dfs(child, path)
            path.pop()

    dfs(sym_root, [])
    return out


def apply_rewrite_match(
    ind: Individual,
    match: RewriteMatch,
    *,
    device_type_by_id: Optional[Dict[int, str]] = None,
) -> RewriteCandidateResult:
    result = RewriteCandidateResult(match=match, matched=True)
    dtype_by_id = device_type_by_id or {int(d): str(int(d)) for d in ind.devices}
    device_type_to_ids = _build_device_type_to_ids(ind.devices, dtype_by_id)

    try:
        sym_root = individual_to_symbolic(ind, device_type_by_id=dtype_by_id)
        target = _get_node_by_path(sym_root, match.path)
        replacement = match.pattern.apply(target.clone())
        if replacement is None:
            result.error = "Pattern matched during enumeration, but apply() returned None."
            return result

        result.applied = True
        result.after_node_pretty = replacement.pretty()
        _replace_node_by_path(sym_root, match.path, replacement)
        sym_root.recompute_counts()
        result.after_root_pretty = sym_root.pretty()
        result.materializable = is_materializable(sym_root)
        if not result.materializable:
            result.error = "Rewritten symbolic tree is not materializable."
            return result

        cand = symbolic_to_individual(
            sym_root,
            device_type_to_ids=device_type_to_ids,
            req_type_num=ind.req_type_num,
            batch_size=int(getattr(ind, "batch_size", 1)),
            devices=list(ind.devices),
            sub_graph_batch_sizes=dict(getattr(ind, "sub_graph_batch_sizes", {})),
        )
        cand.check_legality()
        result.legal = True
        result.candidate_individual = cand
        return result
    except Exception as e:
        result.error = str(e)
        return result


def debug_rewrite_candidates(
    ind: Individual,
    *,
    fitness_fn: Callable[[Any, int], Any],
    root_init: RootInit,
    device_type_by_id: Optional[Dict[int, str]] = None,
    patterns: Optional[Sequence[PatternSpec]] = None,
    family: Optional[RewriteFamily] = None,
    pattern_names: Optional[Sequence[str]] = None,
    attach_hardware_leaves: bool = True,
) -> RewriteDebugReport:
    baseline = evaluate_individual(
        ind,
        fitness_fn=fitness_fn,
        root_init=root_init,
        attach_hardware_leaves=attach_hardware_leaves,
    )
    matches = enumerate_rewrite_matches(
        ind,
        device_type_by_id=device_type_by_id,
        patterns=patterns,
        family=family,
        pattern_names=pattern_names,
    )
    report = RewriteDebugReport(baseline=baseline, matches=matches)

    for match in matches:
        candidate = apply_rewrite_match(ind, match, device_type_by_id=device_type_by_id)
        if candidate.legal and candidate.candidate_individual is not None:
            ev = evaluate_individual(
                candidate.candidate_individual,
                fitness_fn=fitness_fn,
                root_init=root_init,
                attach_hardware_leaves=attach_hardware_leaves,
            )
            candidate.decoded = ev.decoded
            candidate.throughput = ev.throughput
            candidate.latency = ev.latency
            candidate.objectives = ev.objectives
            if not ev.decoded:
                candidate.error = ev.error

            if baseline.objectives is not None and ev.objectives is not None:
                candidate.throughput_delta = float(ev.objectives[0] - baseline.objectives[0])
                candidate.latency_delta = float(ev.objectives[1] - baseline.objectives[1])
                candidate.dominates_baseline = _dominates(ev.objectives, baseline.objectives)
                candidate.dominated_by_baseline = _dominates(baseline.objectives, ev.objectives)
                candidate.equal_objectives = (
                    float(ev.objectives[0]) == float(baseline.objectives[0])
                    and float(ev.objectives[1]) == float(baseline.objectives[1])
                )
        report.candidates.append(candidate)
    return report


def _enumerate_symbolic_matches(
    root: SymbolicNode,
    patterns: Sequence[PatternSpec],
    *,
    family: Optional[RewriteFamily] = None,
) -> List[Tuple[Tuple[int, ...], PatternSpec]]:
    matches: List[Tuple[Tuple[int, ...], PatternSpec]] = []

    def dfs(node: SymbolicNode, path: List[int]) -> None:
        for pat in patterns:
            if family is not None and pat.family != family:
                continue
            if pat.matches(node):
                matches.append((tuple(path), pat))
        for idx, child in enumerate(node.children):
            path.append(idx)
            dfs(child, path)
            path.pop()

    dfs(root, [])
    return matches


def _choose_family(
    *,
    rng: random.Random,
    family_weights: Optional[Dict[RewriteFamily, float]] = None,
) -> RewriteFamily:
    weights = dict(DEFAULT_FAMILY_WEIGHTS)
    if family_weights:
        weights.update({k: float(v) for k, v in family_weights.items()})

    families = [
        RewriteFamily.SKELETON_EXPANSION,
        RewriteFamily.LOCAL_REFINEMENT,
        RewriteFamily.RELABEL,
        RewriteFamily.REPARTITION,
        RewriteFamily.ROLLBACK,
    ]
    probs = [max(1e-9, float(weights.get(fam, DEFAULT_FAMILY_WEIGHTS[fam]))) for fam in families]
    return rng.choices(families, weights=probs, k=1)[0]


def _choose_weighted_match(
    matches: Sequence[Tuple[Tuple[int, ...], PatternSpec]],
    *,
    rng: random.Random,
) -> Tuple[Tuple[int, ...], PatternSpec]:
    weights = [max(1e-9, float(pat.weight)) for _, pat in matches]
    return rng.choices(list(matches), weights=weights, k=1)[0]


def debug_rewrite_multistep(
    ind: Individual,
    *,
    fitness_fn: Callable[[Any, int], Any],
    root_init: RootInit,
    device_type_by_id: Optional[Dict[int, str]] = None,
    rewrite_max_steps: int = 4,
    family: Optional[RewriteFamily] = None,
    family_weights: Optional[Dict[RewriteFamily, float]] = None,
    seed: Optional[int] = None,
    attach_hardware_leaves: bool = True,
) -> MultiStepRewriteReport:
    baseline = evaluate_individual(
        ind,
        fitness_fn=fitness_fn,
        root_init=root_init,
        attach_hardware_leaves=attach_hardware_leaves,
    )
    report = MultiStepRewriteReport(
        baseline=baseline,
        rewrite_max_steps=max(1, int(rewrite_max_steps)),
    )

    dtype_by_id = device_type_by_id or {int(d): str(int(d)) for d in ind.devices}
    device_type_to_ids = _build_device_type_to_ids(ind.devices, dtype_by_id)
    sym_root = individual_to_symbolic(ind, device_type_by_id=dtype_by_id)
    patterns = list(default_patterns())
    rng = random.Random(seed)

    prev_objectives = baseline.objectives
    changed_any = False

    for step in range(report.rewrite_max_steps):
        before_root = sym_root.pretty()

        if has_open_nodes(sym_root):
            requested_family = RewriteFamily.LOCAL_REFINEMENT
            matches = _enumerate_symbolic_matches(sym_root, patterns, family=requested_family)
            used_fallback = False
        else:
            requested_family = family if family is not None else _choose_family(rng=rng, family_weights=family_weights)
            matches = _enumerate_symbolic_matches(sym_root, patterns, family=requested_family)
            used_fallback = False
            if not matches and family is None:
                matches = _enumerate_symbolic_matches(sym_root, patterns, family=None)
                used_fallback = bool(matches)

        if not matches:
            trace = MultiStepTrace(
                step=step,
                requested_family=requested_family,
                effective_family=None,
                used_family_fallback=used_fallback,
                applied=False,
                before_root_pretty=before_root,
                after_root_pretty=before_root,
                materializable=is_materializable(sym_root),
                error="No matching rewrite rule for the requested step.",
            )
            report.traces.append(trace)
            report.stopped_reason = "No matching rewrite rule for the requested step."
            break

        path, pat = _choose_weighted_match(matches, rng=rng)
        target = _get_node_by_path(sym_root, path)
        before_node = target.pretty()
        replacement = pat.apply(target.clone())

        if replacement is None:
            trace = MultiStepTrace(
                step=step,
                requested_family=requested_family,
                effective_family=pat.family,
                used_family_fallback=used_fallback,
                applied=False,
                pattern_name=pat.name,
                path=path,
                before_node_pretty=before_node,
                before_root_pretty=before_root,
                after_root_pretty=before_root,
                materializable=is_materializable(sym_root),
                error="Pattern matched but apply() returned None.",
            )
            report.traces.append(trace)
            report.stopped_reason = "Pattern matched but apply() returned None."
            break

        changed_any = True
        _replace_node_by_path(sym_root, path, replacement)
        sym_root.recompute_counts()
        after_root = sym_root.pretty()
        trace = MultiStepTrace(
            step=step,
            requested_family=requested_family,
            effective_family=pat.family,
            used_family_fallback=used_fallback and pat.family != requested_family,
            applied=True,
            pattern_name=pat.name,
            path=path,
            before_node_pretty=before_node,
            after_node_pretty=replacement.pretty(),
            before_root_pretty=before_root,
            after_root_pretty=after_root,
            materializable=is_materializable(sym_root),
        )

        if trace.materializable:
            try:
                cand = symbolic_to_individual(
                    sym_root,
                    device_type_to_ids=device_type_to_ids,
                    req_type_num=ind.req_type_num,
                    batch_size=int(getattr(ind, "batch_size", 1)),
                    devices=list(ind.devices),
                    sub_graph_batch_sizes=dict(getattr(ind, "sub_graph_batch_sizes", {})),
                )
                cand.check_legality()
                trace.legal = True
                trace.candidate_individual = cand

                ev = evaluate_individual(
                    cand,
                    fitness_fn=fitness_fn,
                    root_init=root_init,
                    attach_hardware_leaves=attach_hardware_leaves,
                )
                trace.decoded = ev.decoded
                if ev.objectives is not None:
                    dominates_prev = _dominates(ev.objectives, prev_objectives) if prev_objectives is not None else None
                    dominated_by_prev = _dominates(prev_objectives, ev.objectives) if prev_objectives is not None else None
                    dominates_baseline = _dominates(ev.objectives, baseline.objectives) if baseline.objectives is not None else None
                    dominated_by_baseline = _dominates(baseline.objectives, ev.objectives) if baseline.objectives is not None else None
                    trace.attribution = MultiStepAttribution(
                        throughput=float(ev.objectives[0]),
                        latency=float(ev.objectives[1]),
                        objectives=ev.objectives,
                        throughput_delta_from_prev=None if prev_objectives is None else float(ev.objectives[0] - prev_objectives[0]),
                        latency_delta_from_prev=None if prev_objectives is None else float(ev.objectives[1] - prev_objectives[1]),
                        throughput_delta_from_baseline=None if baseline.objectives is None else float(ev.objectives[0] - baseline.objectives[0]),
                        latency_delta_from_baseline=None if baseline.objectives is None else float(ev.objectives[1] - baseline.objectives[1]),
                        dominates_prev=dominates_prev,
                        dominates_baseline=dominates_baseline,
                        dominated_by_prev=dominated_by_prev,
                        dominated_by_baseline=dominated_by_baseline,
                    )
                    prev_objectives = ev.objectives
                    report.final_objectives = ev.objectives
                    report.final_decoded = ev.decoded
                    report.final_individual = cand
                elif not ev.decoded:
                    trace.error = ev.error
                report.final_materializable = True
                report.final_legal = True
            except Exception as e:
                trace.error = str(e)

        report.traces.append(trace)

        if trace.materializable:
            report.stopped_reason = "Reached materializable closure."
            break

    if not report.traces:
        report.stopped_reason = "No rewrite step executed."
    elif (not report.final_materializable) and report.stopped_reason is None:
        if changed_any:
            report.stopped_reason = "Reached step budget without materializable closure."
        else:
            report.stopped_reason = "No rewrite step executed."

    return report


def sort_candidates(report: RewriteDebugReport) -> List[RewriteCandidateResult]:
    def key(c: RewriteCandidateResult) -> Tuple[int, int, float, float, str]:
        improved = 1 if c.dominates_baseline else 0
        legal = 1 if c.legal else 0
        t = float("-inf") if c.throughput is None else float(c.throughput)
        l = float("inf") if c.latency is None else float(c.latency)
        return (-improved, -legal, -t, l, c.match.pattern.name)

    return sorted(report.candidates, key=key)


def format_report(
    report: RewriteDebugReport,
    *,
    topk: Optional[int] = None,
    include_individual_text: bool = False,
) -> str:
    lines: List[str] = []
    lines.append("=== Rewrite Debug Report ===")
    if report.baseline.objectives is not None:
        lines.append(
            f"Baseline: "
            f"throughput={report.baseline.objectives[0]:.6f}, "
            f"latency={report.baseline.objectives[1]:.6f}"
        )
    else:
        lines.append(f"Baseline: decode/eval failed: {report.baseline.error}")
    lines.append(f"Matched rules: {len(report.matches)}")
    lines.append("")

    items = sort_candidates(report)
    if topk is not None:
        items = items[: max(0, int(topk))]

    if not items:
        lines.append("No rewrite candidates.")
        return "\n".join(lines)

    for idx, c in enumerate(items, start=1):
        lines.append(f"[{idx}] pattern={c.match.pattern.name} family={c.match.family.value} path={c.match.path_str}")
        lines.append(f"    before_node: {c.match.before_node_pretty}")
        lines.append(f"    after_node : {c.after_node_pretty}")
        lines.append(f"    applied={c.applied} materializable={c.materializable} legal={c.legal} decoded={c.decoded}")
        if c.objectives is not None:
            dt = "None" if c.throughput_delta is None else f"{c.throughput_delta:+.6f}"
            dl = "None" if c.latency_delta is None else f"{c.latency_delta:+.6f}"
            lines.append(
                "    objectives: "
                f"throughput={c.objectives[0]:.6f}, latency={c.objectives[1]:.6f}, "
                f"delta_T={dt}, delta_L={dl}, "
                f"dominates_baseline={c.dominates_baseline}, dominated_by_baseline={c.dominated_by_baseline}"
            )
        if c.error:
            lines.append(f"    error: {c.error}")
        if include_individual_text and c.candidate_individual is not None:
            for ln in format_individual(c.candidate_individual).splitlines():
                lines.append(f"    {ln}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_multistep_report(
    report: MultiStepRewriteReport,
    *,
    include_individual_text: bool = False,
) -> str:
    lines: List[str] = []
    lines.append("=== Multi-step Rewrite Debug Report ===")
    if report.baseline.objectives is not None:
        lines.append(
            f"Baseline: throughput={report.baseline.objectives[0]:.6f}, "
            f"latency={report.baseline.objectives[1]:.6f}"
        )
    else:
        lines.append(f"Baseline: decode/eval failed: {report.baseline.error}")
    lines.append(f"rewrite_max_steps={report.rewrite_max_steps}")
    if report.stopped_reason:
        lines.append(f"stopped_reason={report.stopped_reason}")
    lines.append("")

    if not report.traces:
        lines.append("No rewrite steps executed.")
        return "\n".join(lines)

    best_t = None
    best_l = None
    for trace in report.traces:
        lines.append(
            f"[step {trace.step}] requested_family="
            f"{None if trace.requested_family is None else trace.requested_family.value} "
            f"effective_family={None if trace.effective_family is None else trace.effective_family.value} "
            f"fallback={trace.used_family_fallback} applied={trace.applied}"
        )
        if trace.pattern_name is not None:
            lines.append(f"    pattern={trace.pattern_name} path={trace.path_str}")
        if trace.before_node_pretty is not None:
            lines.append(f"    before_node: {trace.before_node_pretty}")
        if trace.after_node_pretty is not None:
            lines.append(f"    after_node : {trace.after_node_pretty}")
        if trace.before_root_pretty is not None:
            lines.append(f"    before_root: {trace.before_root_pretty}")
        if trace.after_root_pretty is not None:
            lines.append(f"    after_root : {trace.after_root_pretty}")
        lines.append(
            f"    materializable={trace.materializable} legal={trace.legal} decoded={trace.decoded}"
        )
        if trace.attribution is not None and trace.attribution.objectives is not None:
            attr = trace.attribution
            dt_prev = "None" if attr.throughput_delta_from_prev is None else f"{attr.throughput_delta_from_prev:+.6f}"
            dl_prev = "None" if attr.latency_delta_from_prev is None else f"{attr.latency_delta_from_prev:+.6f}"
            dt_base = "None" if attr.throughput_delta_from_baseline is None else f"{attr.throughput_delta_from_baseline:+.6f}"
            dl_base = "None" if attr.latency_delta_from_baseline is None else f"{attr.latency_delta_from_baseline:+.6f}"
            lines.append(
                "    objectives: "
                f"throughput={attr.objectives[0]:.6f}, latency={attr.objectives[1]:.6f}, "
                f"delta_prev_T={dt_prev}, delta_prev_L={dl_prev}, "
                f"delta_base_T={dt_base}, delta_base_L={dl_base}"
            )
            lines.append(
                "    attribution: "
                f"dominates_prev={attr.dominates_prev}, dominated_by_prev={attr.dominated_by_prev}, "
                f"dominates_baseline={attr.dominates_baseline}, dominated_by_baseline={attr.dominated_by_baseline}"
            )
            if attr.throughput_delta_from_prev is not None:
                if best_t is None or attr.throughput_delta_from_prev > best_t[1]:
                    best_t = (trace.step, attr.throughput_delta_from_prev)
            if attr.latency_delta_from_prev is not None:
                if best_l is None or attr.latency_delta_from_prev < best_l[1]:
                    best_l = (trace.step, attr.latency_delta_from_prev)
        if trace.error:
            lines.append(f"    error: {trace.error}")
        if include_individual_text and trace.candidate_individual is not None:
            for ln in format_individual(trace.candidate_individual).splitlines():
                lines.append(f"    {ln}")
        lines.append("")

    if report.final_objectives is not None:
        lines.append(
            f"Final: throughput={report.final_objectives[0]:.6f}, "
            f"latency={report.final_objectives[1]:.6f}"
        )
    lines.append(
        f"Final state: materializable={report.final_materializable} "
        f"legal={report.final_legal} decoded={report.final_decoded}"
    )
    if best_t is not None:
        lines.append(f"Best throughput gain step: step={best_t[0]} delta={best_t[1]:+.6f}")
    if best_l is not None:
        lines.append(f"Best latency gain step: step={best_l[0]} delta={best_l[1]:+.6f}")

    return "\n".join(lines).rstrip()


def save_report_json(report: RewriteDebugReport, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.summary_dict(), f, ensure_ascii=False, indent=2)


def save_multistep_report_json(report: MultiStepRewriteReport, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.summary_dict(), f, ensure_ascii=False, indent=2)


def dump_candidate_individuals(
    report: RewriteDebugReport,
    out_dir: pathlib.Path,
    *,
    topk: Optional[int] = None,
    improved_only: bool = False,
) -> List[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[pathlib.Path] = []
    items = sort_candidates(report)
    if improved_only:
        items = [c for c in items if c.dominates_baseline]
    if topk is not None:
        items = items[: max(0, int(topk))]

    for idx, c in enumerate(items, start=1):
        if c.candidate_individual is None:
            continue
        fname = f"candidate_{idx:03d}_{c.match.pattern.name}.json"
        path = out_dir / fname
        save_individual_json(c.candidate_individual, path)
        written.append(path)
    return written


def dump_multistep_individuals(
    report: MultiStepRewriteReport,
    out_dir: pathlib.Path,
    *,
    improved_only: bool = False,
) -> List[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[pathlib.Path] = []

    for trace in report.traces:
        if trace.candidate_individual is None:
            continue
        if improved_only:
            if trace.attribution is None or not trace.attribution.dominates_baseline:
                continue
        stem = trace.pattern_name or "step"
        fname = f"step_{trace.step:03d}_{stem}.json"
        path = out_dir / fname
        save_individual_json(trace.candidate_individual, path)
        written.append(path)

    if report.final_individual is not None:
        final_path = out_dir / "final_individual.json"
        save_individual_json(report.final_individual, final_path)
        written.append(final_path)

    return written
