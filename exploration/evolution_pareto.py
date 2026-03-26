# evolution_pareto.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import random
import math
import copy
import uuid
import hashlib
from collections import deque
import json

from parallelism.pnode import Parallelism, XpTag

from exploration.individual import (
    Attrs,
    DeviceAssign,
    Individual,
    Topology,
    TopologyNodeGene,
)

from exploration.decoder import RootInit, try_decode_to_root
from exploration.seed_from_pcase import individual_from_pcase_root
from exploration.ind_io import (
    log_individual_json,
    print_individual,
    format_topology,
    save_individual_json
)
from exploration.rewrite_mechanism import (
    InitPatternSpec,
    RewriteEngine,
    RewriteFamily,
    default_init_patterns,
    default_patterns,
    has_open_nodes,
    individual_to_symbolic,
    is_materializable,
    symbolic_to_individual,
)


@dataclass
class InitConfig:
    population_size: int = 50
    max_depth: int = 4
    max_children: int = 4
    p_stop_expand: float = 0.35

    # XP MUST be exactly one ATTENTION and one LINEAR
    xp_swap_prob: float = 0.5

    shuffle_devices_per_individual: bool = True

    # Mixed initialization ratios
    p_pattern_seed_init: float = 0.50
    p_stratified_init: float = 0.45
    p_random_init: float = 0.05

    # Per-individual batch size search space (resampled on every mutation)
    batch_size_choices: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

@dataclass
class EvoConfig:
    generations: int = 50
    elite_size: int = 5
    offspring_size: Optional[int] = None

    # New mutation taxonomy
    p_rewrite_mut: float = 0.60
    p_numeric_mut: float = 0.35
    p_mapping_refine_mut: float = 0.05

    # Backward-compatible aliases.
    p_topology_mut: Optional[float] = None
    p_device_mut: Optional[float] = None

    # Rewrite family weights
    p_skeleton_expand: float = 0.25
    p_local_refine: float = 0.30
    p_relabel: float = 0.15
    p_repartition: float = 0.20
    p_rollback: float = 0.10
    rewrite_max_steps: int = 4

    weight_noise_sigma: float = 0.25
    max_attempt_factor: int = 20

    tournament_k: int = 3
    enable_cache: bool = True
    enable_subgraph_batch_mut: bool = False

    def __post_init__(self) -> None:
        if self.p_topology_mut is not None or self.p_device_mut is not None:
            top = float(self.p_topology_mut or 0.0)
            dev = float(self.p_device_mut or 0.0)
            num = float(self.p_numeric_mut)
            total = top + dev + num
            if total > 0:
                self.p_rewrite_mut = (top + dev) / total
                self.p_numeric_mut = num / total
                self.p_mapping_refine_mut = 0.0

        total = self.p_rewrite_mut + self.p_numeric_mut + self.p_mapping_refine_mut
        if total <= 0:
            self.p_rewrite_mut, self.p_numeric_mut, self.p_mapping_refine_mut = 0.6, 0.35, 0.05
        else:
            self.p_rewrite_mut /= total
            self.p_numeric_mut /= total
            self.p_mapping_refine_mut /= total


def canonical_key(ind: Individual, *, round_digits: int = 3) -> str:
    topo = ind.topology
    topo_items = sorted(
        [(g.node_id, g.parent_id, int(g.ptype.value), g.child_slot) for g in topo.nodes],
        key=lambda x: (x[1], x[3], x[0]),
    )

    def round_list(xs):
        return [round(float(x), round_digits) for x in xs]

    attrs_items = []
    for nid in topo.iter_dfs():
        t = topo.gene(nid).ptype
        if t == Parallelism.DP:
            v = [[round(float(x), round_digits) for x in row] for row in ind.attrs.dp_attr.get(nid, [])]
            attrs_items.append(("DP", nid, v))
        elif t == Parallelism.PP:
            attrs_items.append(("PP", nid, round_list(ind.attrs.pp_attr.get(nid, []))))
        elif t == Parallelism.TP:
            attrs_items.append(("TP", nid, round_list(ind.attrs.tp_attr.get(nid, []))))
        elif t == Parallelism.XP:
            attrs_items.append(("XP", nid, [int(x.value) for x in ind.attrs.xp_attr.get(nid, [])]))

    leaves = sorted(topo.leaf_parallel_nodes())
    dev_items = [(nid, ind.device_assign.leaf_to_devices.get(nid, [])) for nid in leaves]

    sub_graph_batch_items = sorted((int(k), int(v)) for k, v in getattr(ind, "sub_graph_batch_sizes", {}).items())
    s = repr((topo_items, attrs_items, dev_items, int(getattr(ind, "batch_size", 1)), sub_graph_batch_items)).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


def _allowed_types(seen_pp: bool, seen_tp: bool) -> List[Parallelism]:
    allowed = [Parallelism.XP, Parallelism.TP, Parallelism.PP, Parallelism.DP]
    if seen_pp or seen_tp:
        allowed = [t for t in allowed if t != Parallelism.DP]
    if seen_tp:
        allowed = [t for t in allowed if t != Parallelism.PP]
    return allowed


def random_topology(cfg: InitConfig) -> Topology:
    nodes: List[TopologyNodeGene] = []
    root_type = random.choice(_allowed_types(False, False))
    nodes.append(TopologyNodeGene(node_id=0, parent_id=-1, ptype=root_type, child_slot=0))
    next_id = 1

    q: deque[Tuple[int, int, bool, bool]] = deque()
    q.append((0, 0, False, False))
    id2ptype: Dict[int, Parallelism] = {0: root_type}

    while q:
        nid, depth, seen_pp, seen_tp = q.popleft()
        t = id2ptype[nid]
        npp = seen_pp or (t == Parallelism.PP)
        ntp = seen_tp or (t == Parallelism.TP)

        if depth >= cfg.max_depth or random.random() < cfg.p_stop_expand:
            continue

        k = 2 if t == Parallelism.XP else random.randint(2, cfg.max_children)
        for slot in range(k):
            ctype = random.choice(_allowed_types(npp, ntp))
            nodes.append(TopologyNodeGene(node_id=next_id, parent_id=nid, ptype=ctype, child_slot=slot))
            id2ptype[next_id] = ctype
            q.append((next_id, depth + 1, npp, ntp))
            next_id += 1

    return Topology(nodes=nodes)


def _rand_positive_weights(k: int) -> List[float]:
    return [random.lognormvariate(0.0, 0.8) for _ in range(k)]


def _random_partition(n: int, k: int) -> List[int]:
    if k > n:
        raise ValueError("leaf_count > device_count")
    cuts = sorted(random.sample(range(1, n), k - 1)) if k > 1 else []
    parts = []
    prev = 0
    for c in cuts + [n]:
        parts.append(c - prev)
        prev = c
    return parts


def sample_device_assign(topo: Topology, devices: Sequence[int], *, shuffle: bool = True) -> DeviceAssign:
    # 最终目的：把 devices 随机切分到各个 leaf 上，每个 leaf 至少一个
    # 取 leaf 列表
    leaves = sorted(topo.leaf_parallel_nodes())
    if not leaves:
        raise ValueError("Topology has no leaves.")
    dev_list = list(devices)
    if shuffle:
        random.shuffle(dev_list)
    L = len(leaves)
    N = len(dev_list)
    # 生成一个随机 partition：把 N 分成 L 个正整数（每个 ≥1）
    sizes = _random_partition(N, L)  # each leaf >=1 device

    da = DeviceAssign()
    idx = 0
    for leaf_id, sz in zip(leaves, sizes):
        da.leaf_to_devices[leaf_id] = dev_list[idx: idx + sz]
        idx += sz
    return da

def _detect_begin_node_ids(topo: Topology) -> List[int]:
    """Mirror ptraversal.detect_begin_nodes() on topology node ids."""
    out: List[int] = []

    def dfs(nid: int) -> None:
        if topo.gene(nid).ptype == Parallelism.DP:
            for cid in topo.children_of(nid):
                dfs(cid)
        else:
            out.append(nid)

    dfs(topo.root_id)
    return out


def _sample_sub_graph_batch_sizes_for_topology(topo: Topology,
                                               max_batch_lo: int,
                                               enable_subgraph_batch_mut: bool) -> Dict[int, int]:
    """
    Resample sub-graph batch sizes with a mixed strategy:
      - 70%: reset every begin node to the global upper bound
      - 30%: true random resampling in [1, max_batch_lo]
    """
    begin_ids = _detect_begin_node_ids(topo)
    upper = max(1, int(max_batch_lo))

    if enable_subgraph_batch_mut:
        if random.random() < 0.50:
            return {int(nid): int(upper) for nid in begin_ids}
        return {int(nid): int(random.randint(1, upper)) for nid in begin_ids}
    else:
        return {int(nid): int(upper) for nid in begin_ids}


def _tweak_sub_graph_batch_sizes(parent_map: Dict[int, int], child_begin_ids: Sequence[int], max_batch_lo: int) -> Dict[int, int]:
    """Micro-tune sub-graph batch sizes only around the parent's assignments."""
    upper = max(1, int(max_batch_lo))
    if not child_begin_ids:
        return {}

    child_map: Dict[int, int] = {}
    touched = False
    for nid in child_begin_ids:
        base = int(parent_map.get(int(nid), upper))
        base = max(1, min(upper, base))
        if random.random() < 0.6:
            delta = random.choice([-2, -1, 1, 2])
            base = max(1, min(upper, base + delta))
            touched = True
        child_map[int(nid)] = int(base)

    if (not touched) and child_begin_ids:
        nid = int(random.choice(list(child_begin_ids)))
        base = child_map[nid]
        delta = random.choice([-2, -1, 1, 2])
        child_map[nid] = max(1, min(upper, base + delta))
    return child_map


def _mutate_sub_graph_batch_sizes(
    parent: Individual,
    child: Individual,
    mutation_kind: str,
    enable_subgraph_batch_mut: bool
) -> Dict[int, int]:
    child_begin_ids = _detect_begin_node_ids(child.topology)
    upper = max(1, int(getattr(child, "batch_size", 1)))

    # 不做 subgraph 级搜索时，所有 begin-node 直接绑定到 global batch size
    if not enable_subgraph_batch_mut:
        return {int(nid): int(upper) for nid in child_begin_ids}

    if mutation_kind == "topology":
        return _sample_sub_graph_batch_sizes_for_topology(child.topology, upper, enable_subgraph_batch_mut)

    parent_map = {int(k): int(v) for k, v in getattr(parent, "sub_graph_batch_sizes", {}).items()}
    if (not parent_map) or (set(parent_map.keys()) != set(int(x) for x in child_begin_ids)):
        return _sample_sub_graph_batch_sizes_for_topology(child.topology, upper, enable_subgraph_batch_mut)

    r = random.random()
    if r < 0.60:
        return {int(nid): max(1, min(upper, int(parent_map[int(nid)]))) for nid in child_begin_ids}
    if r < 0.80:
        return _tweak_sub_graph_batch_sizes(parent_map, child_begin_ids, upper)
    return _sample_sub_graph_batch_sizes_for_topology(child.topology, upper, enable_subgraph_batch_mut)


def sample_attrs(topo: Topology, device_assign: DeviceAssign, req_type_num: int, init_cfg: InitConfig) -> Attrs:
    """
    v3: leaf arity is device_group_size; XP tags are exactly [ATTENTION, LINEAR] (maybe swapped).
    """
    attrs = Attrs()
    for nid in topo.iter_dfs():
        t = topo.gene(nid).ptype

        parallel_k = len(topo.children_of(nid))
        if parallel_k == 0:
            k = len(device_assign.leaf_to_devices[nid])
        else:
            k = parallel_k

        if t == Parallelism.DP:
            attrs.dp_attr[nid] = [_rand_positive_weights(k) for _ in range(req_type_num)]
        elif t == Parallelism.PP:
            attrs.pp_attr[nid] = _rand_positive_weights(k)
        elif t == Parallelism.TP:
            attrs.tp_attr[nid] = _rand_positive_weights(k)
        elif t == Parallelism.XP:
            tags = [XpTag.ATTENTION, XpTag.LINEAR]
            if random.random() < init_cfg.xp_swap_prob:
                tags.reverse()
            attrs.xp_attr[nid] = tags
        else:
            raise ValueError(f"Unexpected type: {t}")
    return attrs



def _normalized_init_mix(init_cfg: InitConfig) -> Tuple[float, float, float]:
    mix = [
        max(0.0, float(getattr(init_cfg, "p_pattern_seed_init", 0.50))),
        max(0.0, float(getattr(init_cfg, "p_stratified_init", 0.45))),
        max(0.0, float(getattr(init_cfg, "p_random_init", 0.05))),
    ]
    total = sum(mix)
    if total <= 0:
        return 0.50, 0.45, 0.05
    return mix[0] / total, mix[1] / total, mix[2] / total


def _plan_init_counts(total: int, init_cfg: InitConfig) -> Tuple[int, int, int]:
    p_pat, p_strat, p_rand = _normalized_init_mix(init_cfg)
    raw = [total * p_pat, total * p_strat, total * p_rand]
    counts = [int(math.floor(x)) for x in raw]
    remain = int(total - sum(counts))
    frac_order = sorted(
        range(3),
        key=lambda i: (raw[i] - counts[i], i),
        reverse=True,
    )
    for i in frac_order:
        if remain <= 0:
            break
        counts[i] += 1
        remain -= 1
    return max(0, counts[0]), max(0, counts[1]), max(0, counts[2])


def _build_device_type_to_ids(
    devices: Sequence[int],
    device_type_by_id: Optional[Dict[int, str]],
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for did in devices:
        dtype = str((device_type_by_id or {}).get(int(did), str(int(did))))
        out.setdefault(dtype, []).append(int(did))
    return out


def _shuffle_device_type_to_ids(device_type_to_ids: Dict[str, List[int]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for dtype, ids in device_type_to_ids.items():
        vals = list(ids)
        random.shuffle(vals)
        out[str(dtype)] = vals
    return out


def _evaluate_individual(
    ind: Individual,
    *,
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
) -> bool:
    try:
        ind.check_legality()
    except Exception:
        return False

    local_root_init = copy.deepcopy(root_init)
    root = try_decode_to_root(ind, local_root_init, attach_hardware_leaves=attach_hardware_leaves)
    if root is None:
        return False

    try:
        res = fitness_fn(root, ind.batch_size)
        thr, lat = _parse_objectives(res)
        if (not math.isfinite(thr)) or (not math.isfinite(lat)):
            return False
        ind.throughput = thr
        ind.latency = lat
        ind.objectives = (thr, lat)
    except Exception:
        return False
    return True


def _try_register_individual(
    pop: List[Individual],
    seen: set[str],
    ind: Individual,
    *,
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
) -> bool:

    uid = canonical_key(ind)
    ind.uid = uid
    if uid in seen:
        return False

    if not _evaluate_individual(
        ind,
        root_init=root_init,
        fitness_fn=fitness_fn,
        attach_hardware_leaves=attach_hardware_leaves,
    ):
        return False

    seen.add(uid)
    pop.append(ind)
    return True


def _make_individual_from_symbolic_seed(
    sym_root,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    req_type_num: int,
    devices: Sequence[int],
    device_type_to_ids: Dict[str, List[int]],
) -> Optional[Individual]:
    batch_size = int(random.choice(init_cfg.batch_size_choices))
    try:
        ind = symbolic_to_individual(
            sym_root,
            device_type_to_ids=_shuffle_device_type_to_ids(device_type_to_ids),
            req_type_num=req_type_num,
            batch_size=batch_size,
            devices=list(devices),
            sub_graph_batch_sizes={},
        )
    except Exception:
        return None

    ind.sub_graph_batch_sizes = _sample_sub_graph_batch_sizes_for_topology(
        ind.topology,
        batch_size,
        evo_cfg.enable_subgraph_batch_mut,
    )
    return ind


def _fill_pattern_seeded_population(
    pop: List[Individual],
    seen: set[str],
    target_count: int,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    req_type_num: int,
    devices: Sequence[int],
    device_type_by_id: Optional[Dict[int, str]],
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
) -> None:
    if target_count <= 0:
        return

    start_len = len(pop)
    device_type_to_ids = _build_device_type_to_ids(devices, device_type_by_id)
    patterns = default_init_patterns(device_type_to_ids)
    if not patterns:
        return

    weights = [max(1e-9, float(p.weight)) for p in patterns]
    attempts = 0
    max_attempts = max(20, target_count * 40)

    while len(pop) - start_len < target_count and attempts < max_attempts:
        attempts += 1
        pat = random.choices(patterns, weights=weights, k=1)[0]
        ind = _make_individual_from_symbolic_seed(
            pat.instantiate(),
            init_cfg=init_cfg,
            evo_cfg=evo_cfg,
            req_type_num=req_type_num,
            devices=devices,
            device_type_to_ids=device_type_to_ids,
        )
        if ind is None:
            continue

        # print(ind.batch_size, format_topology(ind, True, True),"\n")

        _try_register_individual(
            pop,
            seen,
            ind,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
        )


def _fill_stratified_population(
    pop: List[Individual],
    seen: set[str],
    target_count: int,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    req_type_num: int,
    devices: Sequence[int],
    device_type_by_id: Optional[Dict[int, str]],
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
) -> None:
    if target_count <= 0:
        return

    start_len = len(pop)
    device_type_to_ids = _build_device_type_to_ids(devices, device_type_by_id)
    patterns = default_init_patterns(device_type_to_ids)
    if not patterns:
        return

    by_stratum: Dict[str, List[InitPatternSpec]] = {}
    for pat in patterns:
        by_stratum.setdefault(str(pat.stratum), []).append(pat)
    strata = list(sorted(by_stratum.keys()))
    if not strata:
        return

    attempts = 0
    max_attempts = max(30, target_count * 50)
    idx = 0

    while len(pop) - start_len < target_count and attempts < max_attempts:
        attempts += 1
        stratum = strata[idx % len(strata)]
        idx += 1
        plist = by_stratum[stratum]
        weights = [max(1e-9, float(p.weight)) for p in plist]
        pat = random.choices(plist, weights=weights, k=1)[0]
        ind = _make_individual_from_symbolic_seed(
            pat.instantiate(),
            init_cfg=init_cfg,
            evo_cfg=evo_cfg,
            req_type_num=req_type_num,
            devices=devices,
            device_type_to_ids=device_type_to_ids,
        )
        if ind is None:
            continue
        _try_register_individual(
            pop,
            seen,
            ind,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
        )


def _fill_random_population(
    pop: List[Individual],
    seen: set[str],
    target_count: int,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    req_type_num: int,
    devices: Sequence[int],
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
) -> None:
    if target_count <= 0:
        return

    start_len = len(pop)
    attempts = 0
    max_attempts = max(40, target_count * 80)
    devices_list = list(devices)

    while len(pop) - start_len < target_count and attempts < max_attempts:
        attempts += 1
        topo = random_topology(init_cfg)

        try:
            da = sample_device_assign(
                topo,
                devices_list,
                shuffle=init_cfg.shuffle_devices_per_individual,
            )
        except Exception:
            continue

        attrs = sample_attrs(topo, da, req_type_num=req_type_num, init_cfg=init_cfg)
        batch_size = int(random.choice(init_cfg.batch_size_choices))
        ind = Individual(
            topology=topo,
            device_assign=da,
            attrs=attrs,
            devices=devices_list,
            req_type_num=req_type_num,
            batch_size=batch_size,
            sub_graph_batch_sizes=_sample_sub_graph_batch_sizes_for_topology(
                topo,
                batch_size,
                evo_cfg.enable_subgraph_batch_mut,
            ),
        )
        _try_register_individual(
            pop,
            seen,
            ind,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
        )


def _fill_batch_variant_population(
    pop: List[Individual],
    seen: set[str],
    target_count: int,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
) -> None:
    """Final fallback: keep the structure/device assignment, resample batch knobs.

    This is intentionally aligned with the user's goal of broad batch exploration.
    It should only be used after structural seed pools and pure-random structure
    generation have already been tried.
    """
    if target_count <= 0 or not pop:
        return

    start_len = len(pop)
    base_pool = list(pop)
    attempts = 0
    max_attempts = max(60, target_count * 120)

    while len(pop) - start_len < target_count and attempts < max_attempts:
        attempts += 1
        donor = copy.deepcopy(random.choice(base_pool))

        batch_size = int(random.choice(init_cfg.batch_size_choices))
        donor.batch_size = batch_size
        donor.sub_graph_batch_sizes = _sample_sub_graph_batch_sizes_for_topology(
            donor.topology,
            batch_size,
            evo_cfg.enable_subgraph_batch_mut,
        )

        _try_register_individual(
            pop,
            seen,
            donor,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
        )


def _fill_numeric_variant_population(
    pop: List[Individual],
    seen: set[str],
    target_count: int,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
) -> None:
    """Final fallback after batch-only variants: keep validated structures/devices,
    but resample attrs and batch knobs to unlock additional unique seeds.
    """
    if target_count <= 0 or not pop:
        return

    start_len = len(pop)
    base_pool = list(pop)
    attempts = 0
    max_attempts = max(80, target_count * 160)

    while len(pop) - start_len < target_count and attempts < max_attempts:
        attempts += 1
        donor = copy.deepcopy(random.choice(base_pool))
        donor.attrs = sample_attrs(donor.topology, donor.device_assign, req_type_num=donor.req_type_num, init_cfg=init_cfg)
        batch_size = int(random.choice(init_cfg.batch_size_choices))
        donor.batch_size = batch_size
        donor.sub_graph_batch_sizes =_sample_sub_graph_batch_sizes_for_topology(
            donor.topology,
            batch_size,
            evo_cfg.enable_subgraph_batch_mut
        )

        _try_register_individual(
            pop,
            seen,
            donor,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
        )


def _fill_population_mixed(
    pop: List[Individual],
    seen: set[str],
    target_count: int,
    *,
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    req_type_num: int,
    devices: Sequence[int],
    device_type_by_id: Optional[Dict[int, str]],
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool,
) -> None:
    if target_count <= 0:
        return

    start_len = len(pop)

    # Phase 1: honour the configured mixture once.
    c_pat, c_strat, c_rand = _plan_init_counts(target_count, init_cfg)
    print(c_pat, c_strat, c_rand)

    _fill_pattern_seeded_population(
        pop, seen, c_pat,
        init_cfg=init_cfg, evo_cfg=evo_cfg, req_type_num=req_type_num,
        devices=devices, device_type_by_id=device_type_by_id,
        root_init=root_init, fitness_fn=fitness_fn,
        attach_hardware_leaves=attach_hardware_leaves,
    )
    _fill_stratified_population(
        pop, seen, c_strat,
        init_cfg=init_cfg, evo_cfg=evo_cfg, req_type_num=req_type_num,
        devices=devices, device_type_by_id=device_type_by_id,
        root_init=root_init, fitness_fn=fitness_fn,
        attach_hardware_leaves=attach_hardware_leaves,
    )
    _fill_random_population(
        pop, seen, c_rand,
        init_cfg=init_cfg, evo_cfg=evo_cfg, req_type_num=req_type_num,
        devices=devices, root_init=root_init, fitness_fn=fitness_fn,
        attach_hardware_leaves=attach_hardware_leaves,
    )

    # Phase 2: adaptive backfill. If high-quality seed pools are exhausted,
    # progressively lean harder on random legal sampling instead of failing
    # immediately after one no-progress round.
    stagnation_rounds = 0
    while len(pop) - start_len < target_count and stagnation_rounds < 4:
        before = len(pop)
        remaining = target_count - (len(pop) - start_len)

        # keep some seed pressure, but do not let a tiny seed library starve init
        seeded_quota = max(0, int(math.ceil(remaining * 0.40)))
        strat_quota = max(0, int(math.ceil(remaining * 0.30)))
        rand_quota = max(remaining, int(math.ceil(remaining * 1.50)))

        _fill_pattern_seeded_population(
            pop, seen, seeded_quota,
            init_cfg=init_cfg, evo_cfg=evo_cfg, req_type_num=req_type_num,
            devices=devices, device_type_by_id=device_type_by_id,
            root_init=root_init, fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
        )
        if len(pop) - start_len < target_count:
            _fill_stratified_population(
                pop, seen, strat_quota,
                init_cfg=init_cfg, evo_cfg=evo_cfg, req_type_num=req_type_num,
                devices=devices, device_type_by_id=device_type_by_id,
                root_init=root_init, fitness_fn=fitness_fn,
                attach_hardware_leaves=attach_hardware_leaves,
            )
        if len(pop) - start_len < target_count:
            _fill_random_population(
                pop, seen, rand_quota,
                init_cfg=init_cfg, evo_cfg=evo_cfg, req_type_num=req_type_num,
                devices=devices, root_init=root_init, fitness_fn=fitness_fn,
                attach_hardware_leaves=attach_hardware_leaves,
            )

        if len(pop) == before:
            # Last-resort top-up: exploit the already-valid structures by widening
            # batch/sub-graph batch exploration.
            remaining = target_count - (len(pop) - start_len)
            _fill_batch_variant_population(
                pop, seen, max(remaining * 2, remaining),
                init_cfg=init_cfg, evo_cfg=evo_cfg,
                root_init=root_init, fitness_fn=fitness_fn,
                attach_hardware_leaves=attach_hardware_leaves,
            )
            if len(pop) == before:
                _fill_numeric_variant_population(
                    pop, seen, max(remaining * 3, remaining),
                    init_cfg=init_cfg, evo_cfg=evo_cfg,
                    root_init=root_init, fitness_fn=fitness_fn,
                    attach_hardware_leaves=attach_hardware_leaves,
                )
            if len(pop) == before:
                stagnation_rounds += 1
            else:
                stagnation_rounds = 0
        else:
            stagnation_rounds = 0


def initialize_population(
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    *,
    req_type_num: int,
    devices: Sequence[int],
    root_init: RootInit,
    device_type_by_id: Optional[Dict[int, str]] = None,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool = True,
) -> List[Individual]:
    """Pattern/device-abstraction aware mixed initialization with global dedup."""
    pop: List[Individual] = []
    seen: set[str] = set()

    _fill_population_mixed(
        pop,
        seen,
        init_cfg.population_size,
        init_cfg=init_cfg,
        evo_cfg=evo_cfg,
        req_type_num=req_type_num,
        devices=list(devices),
        device_type_by_id=device_type_by_id,
        root_init=root_init,
        fitness_fn=fitness_fn,
        attach_hardware_leaves=attach_hardware_leaves,
    )

    if len(pop) < init_cfg.population_size:
        raise RuntimeError(
            f"Init failed: {len(pop)}/{init_cfg.population_size}. "
            f"pattern/stratified/random mix could not produce enough unique valid individuals."
        )

    fronts = fast_nondominated_sort(pop)
    for f in fronts:
        crowding_distance(f)
    pop.sort(key=lambda x: (10**9 if x.pareto_rank is None else x.pareto_rank, -x.crowding))
    return pop


def initialize_population_with_seeds(
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    *,
    req_type_num: int,
    devices: Sequence[int],
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    seed_roots: Optional[Sequence[Any]] = None,
    seed_individuals: Optional[Sequence[Individual]] = None,
    attach_hardware_leaves: bool = True,
    strict_device_partition: bool = True,
    device_type_by_id: Optional[Dict[int, str]] = None,
) -> List[Individual]:
    """Seed-aware initialization: external seeds first, then mixed-fill the remainder with global dedup."""
    pop: List[Individual] = []
    seen: set[str] = set()
    devices_list = list(devices)

    def try_add(ind: Individual) -> None:
        if not getattr(ind, "sub_graph_batch_sizes", None):
            ind.sub_graph_batch_sizes = _sample_sub_graph_batch_sizes_for_topology(
                ind.topology,
                int(getattr(ind, "batch_size", 1)),
                evo_cfg.enable_subgraph_batch_mut,
            )
        _try_register_individual(
            pop,
            seen,
            ind,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
        )

    if seed_individuals:
        for ind in seed_individuals:
            try_add(copy.deepcopy(ind))

    if seed_roots:
        for root in seed_roots:
            try:
                batch_size = int(random.choice(init_cfg.batch_size_choices))
                local_root = copy.deepcopy(root)
                ind = individual_from_pcase_root(
                    local_root,
                    devices=devices_list,
                    req_type_num=req_type_num,
                    batch_size=batch_size,
                    strict_device_partition=strict_device_partition,
                )
                ind.sub_graph_batch_sizes = _sample_sub_graph_batch_sizes_for_topology(
                    ind.topology,
                    batch_size,
                    evo_cfg.enable_subgraph_batch_mut,
                )
            except Exception:
                continue
            try_add(ind)

    if len(pop) > init_cfg.population_size:
        return nsga2_environmental_select(pop, init_cfg.population_size)

    remaining = init_cfg.population_size - len(pop)
    if remaining > 0:
        _fill_population_mixed(
            pop,
            seen,
            remaining,
            init_cfg=init_cfg,
            evo_cfg=evo_cfg,
            req_type_num=req_type_num,
            devices=devices_list,
            device_type_by_id=device_type_by_id,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
        )

    if len(pop) < init_cfg.population_size:
        raise RuntimeError(
            f"Init with seeds failed: {len(pop)}/{init_cfg.population_size}. "
            f"Not enough unique valid individuals after seed ingestion and mixed initialization."
        )

    fronts = fast_nondominated_sort(pop)
    for f in fronts:
        crowding_distance(f)
    pop.sort(key=lambda x: (10**9 if x.pareto_rank is None else x.pareto_rank, -x.crowding))
    return pop

def _parse_objectives(fitness_result: Any) -> Tuple[float, float]:
    """Parse evaluator output into (throughput, latency).

    In this codebase, pareto-mode simulator outputs are often shaped like:
      [throughput, p99_tpot_or_summary, finished_dist, processed_dist]
    where the second item may be a numeric latency *or* a status string such as
    ``"[Finished] finished=0\n"`` when no request has completed yet.

    We therefore parse more defensively:
      - first numeric field -> throughput
      - first later numeric scalar -> latency
      - if no later numeric scalar exists, assign a large finite latency penalty
        instead of rejecting the individual outright
    """
    if fitness_result is None:
        raise ValueError("fitness_result is None")

    latency_penalty = 1e5

    # tuple/list
    if isinstance(fitness_result, (tuple, list)) and len(fitness_result) >= 1:
        thr = float(fitness_result[0])
        lat = None
        for item in list(fitness_result)[1:]:
            if isinstance(item, (int, float)):
                lat = float(item)
                break
        if lat is None:
            lat = latency_penalty
        return thr, lat

    # dict
    if isinstance(fitness_result, dict):
        lower = {str(k).lower(): v for k, v in fitness_result.items()}

        def pick(*names):
            for n in names:
                if n in lower:
                    return lower[n]
            return None

        thr = pick("T", "throughput", "tps", "qps", "req_s", "req_per_s", "requests_per_s")
        lat = pick("L","latency", "p95_latency", "p99_latency", "p90_latency", "tail_latency", "p95", "p99", "lat_ms", "lat_s")

        if thr is None:
            raise ValueError(f"Cannot parse throughput from dict keys={sorted(lower.keys())}")
        return float(thr), float(latency_penalty if lat is None else lat)

    # single number: old behavior (maximize this)
    if isinstance(fitness_result, (int, float)):
        return float(fitness_result), latency_penalty

    raise ValueError(f"Unsupported fitness_result type: {type(fitness_result)}")


def dominates(a: Individual, b: Individual) -> bool:
    """Return True if a Pareto-dominates b (maximize throughput, minimize latency)."""
    if a.objectives is None or b.objectives is None:
        return False
    a_thr, a_lat = a.objectives
    b_thr, b_lat = b.objectives
    return (a_thr >= b_thr and a_lat <= b_lat) and (a_thr > b_thr or a_lat < b_lat)


def fast_nondominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    """NSGA-II fast non-dominated sorting. Assigns pareto_rank."""
    S: Dict[str, List[Individual]] = {}
    n: Dict[str, int] = {}
    fronts: List[List[Individual]] = [[]]

    for p in pop:
        S[p.uid] = []
        n[p.uid] = 0
        for q in pop:
            if p is q:
                continue
            if dominates(p, q):
                S[p.uid].append(q)
            elif dominates(q, p):
                n[p.uid] += 1
        if n[p.uid] == 0:
            p.pareto_rank = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[Individual] = []
        for p in fronts[i]:
            for q in S[p.uid]:
                n[q.uid] -= 1
                if n[q.uid] == 0:
                    q.pareto_rank = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def crowding_distance(front: List[Individual]) -> None:
    """Compute NSGA-II crowding distance in-place for a front."""
    if not front:
        return
    for p in front:
        p.crowding = 0.0

    # Objective 0: throughput (max)
    # Objective 1: latency (min)  -> treat as negative for sorting so "larger is better" is consistent? 
    # We'll implement separately with correct normalization.
    thr_vals = [p.objectives[0] for p in front]  # type: ignore[index]
    lat_vals = [p.objectives[1] for p in front]  # type: ignore[index]

    # Throughput (ascending for distance calc)
    order = sorted(range(len(front)), key=lambda i: thr_vals[i])
    front[order[0]].crowding = float("inf")
    front[order[-1]].crowding = float("inf")
    thr_min, thr_max = thr_vals[order[0]], thr_vals[order[-1]]
    denom = (thr_max - thr_min) if thr_max != thr_min else 1.0
    for j in range(1, len(order) - 1):
        i_prev, i_next, i_cur = order[j - 1], order[j + 1], order[j]
        front[i_cur].crowding += (thr_vals[i_next] - thr_vals[i_prev]) / denom

    # Latency (ascending: smaller is better)
    order = sorted(range(len(front)), key=lambda i: lat_vals[i])
    front[order[0]].crowding = float("inf")
    front[order[-1]].crowding = float("inf")
    lat_min, lat_max = lat_vals[order[0]], lat_vals[order[-1]]
    denom = (lat_max - lat_min) if lat_max != lat_min else 1.0
    for j in range(1, len(order) - 1):
        i_prev, i_next, i_cur = order[j - 1], order[j + 1], order[j]
        # For latency, larger normalized gap means more isolated => bigger crowding (still good)
        front[i_cur].crowding += (lat_vals[i_next] - lat_vals[i_prev]) / denom


def nsga2_environmental_select(pop: List[Individual], N: int) -> List[Individual]:
    """Select next generation of size N from pop (already evaluated)."""
    fronts = fast_nondominated_sort(pop)
    next_pop: List[Individual] = []
    for f in fronts:
        crowding_distance(f)
        if len(next_pop) + len(f) <= N:
            next_pop.extend(f)
        else:
            # sort by crowding descending
            f_sorted = sorted(f, key=lambda x: x.crowding, reverse=True)
            next_pop.extend(f_sorted[: max(0, N - len(next_pop))])
            break
    return next_pop


def tournament_select(pop: List[Individual], k: int) -> Individual:
    """Binary tournament on (pareto_rank asc, crowding desc)."""
    cand = random.sample(pop, k=min(k, len(pop)))
    # Ensure ranks/crowding exist
    return min(
        cand,
        key=lambda x: (
            10**9 if x.pareto_rank is None else x.pareto_rank,
            -float("inf") if x.crowding is None else -x.crowding,
        ),
    )


def _choose_rewrite_family(cfg: EvoConfig) -> RewriteFamily:
    families = [
        RewriteFamily.SKELETON_EXPANSION,
        RewriteFamily.LOCAL_REFINEMENT,
        RewriteFamily.RELABEL,
        RewriteFamily.REPARTITION,
        RewriteFamily.ROLLBACK,
    ]
    weights = [
        max(1e-9, float(cfg.p_skeleton_expand)),
        max(1e-9, float(cfg.p_local_refine)),
        max(1e-9, float(cfg.p_relabel)),
        max(1e-9, float(cfg.p_repartition)),
        max(1e-9, float(cfg.p_rollback)),
    ]
    return random.choices(families, weights=weights, k=1)[0]


def rewrite_mutation(
    ind: Individual,
    cfg: EvoConfig,
    init_cfg: InitConfig,
    *,
    devices: Sequence[int],
    device_type_by_id: Optional[Dict[int, str]] = None,
) -> Individual:
    """Bounded multi-step closure rewrite.

    In one mutation call, perform up to `cfg.rewrite_max_steps` rewrite steps.
    If the symbolic tree contains unclosed nodes, force LOCAL_REFINEMENT first.
    Only materialize back to Individual when the symbolic tree becomes fully
    materializable within the step budget; otherwise fall back to the parent.
    """
    del init_cfg
    sym_root = individual_to_symbolic(
        ind,
        device_type_by_id=device_type_by_id or {int(d): str(int(d)) for d in devices},
    )
    engine = RewriteEngine(default_patterns(), rng=random)

    max_steps = max(1, int(getattr(cfg, "rewrite_max_steps", 1)))
    changed_any = False

    for _ in range(max_steps):
        if has_open_nodes(sym_root):
            changed = engine.rewrite_random(sym_root, family=RewriteFamily.LOCAL_REFINEMENT)
            if not changed:
                break
        else:
            family = _choose_rewrite_family(cfg)
            changed = engine.rewrite_random(sym_root, family=family)
            if not changed:
                changed = engine.rewrite_random(sym_root, family=None)
            if not changed:
                break

        changed_any = True

        if is_materializable(sym_root):
            break

    if (not changed_any) or (not is_materializable(sym_root)):
        return copy.deepcopy(ind)

    try:
        return symbolic_to_individual(
            sym_root,
            device_type_to_ids=_build_device_type_to_ids(devices, device_type_by_id),
            req_type_num=ind.req_type_num,
            batch_size=int(getattr(ind, "batch_size", 1)),
            devices=list(devices),
            sub_graph_batch_sizes={},
        )
    except Exception:
        return copy.deepcopy(ind)


def mapping_refinement(ind: Individual, *, device_type_by_id: Optional[Dict[int, str]] = None) -> Individual:
    """Keep symbolic semantics fixed and only adjust concrete same-type mapping."""
    topo = ind.topology
    leaves = sorted(topo.leaf_parallel_nodes())
    if len(leaves) <= 1:
        return ind

    new_da = DeviceAssign(leaf_to_devices={lid: grp[:] for lid, grp in ind.device_assign.leaf_to_devices.items()})
    dtype_map = device_type_by_id or {int(d): str(int(d)) for d in ind.devices}

    by_type: Dict[str, List[Tuple[int, int]]] = {}
    for lid in leaves:
        grp = new_da.leaf_to_devices.get(lid, [])
        for idx, did in enumerate(grp):
            by_type.setdefault(str(dtype_map[int(did)]), []).append((lid, idx))

    candidate_types = [t for t, pos in by_type.items() if len(pos) >= 2]
    if not candidate_types:
        return ind

    dtype = random.choice(candidate_types)
    (l1, i1), (l2, i2) = random.sample(by_type[dtype], 2)
    if l1 == l2:
        grp = new_da.leaf_to_devices[l1]
        random.shuffle(grp)
    else:
        g1 = new_da.leaf_to_devices[l1]
        g2 = new_da.leaf_to_devices[l2]
        g1[i1], g2[i2] = g2[i2], g1[i1]

    return Individual(
        topology=copy.deepcopy(ind.topology),
        device_assign=copy.deepcopy(new_da),
        attrs=copy.deepcopy(ind.attrs),
        devices=list(ind.devices),
        req_type_num=ind.req_type_num,
        batch_size=int(getattr(ind, "batch_size", 1)),
        sub_graph_batch_sizes=dict(getattr(ind, "sub_graph_batch_sizes", {})),
    )


def numeric_mutation(ind: Individual, cfg: EvoConfig, init_cfg: InitConfig) -> Individual:
    # 目的：不改变拓扑、不改变设备分配，只在同一棵树结构上“微调参数”，让 fitness 能更平滑地改进。
    # topology：完全不动
    # device_assign：完全不动（因此 leaf 的 device_group_size 也不变）
    # attrs：对某个随机节点的 parallel_attr 做扰动（或改 XP tag）
    topo = ind.topology
    da = ind.device_assign  # keep device partition for numeric mutation

    new_attrs = Attrs(
        dp_attr={nid: [row[:] for row in mat] for nid, mat in ind.attrs.dp_attr.items()},
        pp_attr={nid: vec[:] for nid, vec in ind.attrs.pp_attr.items()},
        tp_attr={nid: vec[:] for nid, vec in ind.attrs.tp_attr.items()},
        xp_attr={nid: tags[:] for nid, tags in ind.attrs.xp_attr.items()},
    )

    candidates = [nid for nid in topo.iter_dfs()]
    nid = random.choice(candidates)
    t = topo.gene(nid).ptype

    sigma = cfg.weight_noise_sigma

    def perturb(x: float) -> float:
        return float(x) * math.exp(random.gauss(0.0, sigma))

    if t == Parallelism.DP:
        mat = new_attrs.dp_attr[nid]
        for r in range(len(mat)):
            mat[r] = [max(1e-9, perturb(x)) for x in mat[r]]
    elif t == Parallelism.PP:
        vec = new_attrs.pp_attr[nid]
        new_attrs.pp_attr[nid] = [max(1e-9, perturb(x)) for x in vec]
    elif t == Parallelism.TP:
        vec = new_attrs.tp_attr[nid]
        new_attrs.tp_attr[nid] = [max(1e-9, perturb(x)) for x in vec]
    elif t == Parallelism.XP:
        # FIX: XP tags must be exactly one ATTENTION and one LINEAR.
        # The only legal mutation is swapping their order.
        tags = new_attrs.xp_attr[nid]
        new_attrs.xp_attr[nid] = [tags[1], tags[0]]

    return Individual(
        copy.deepcopy(ind.topology),
        copy.deepcopy(ind.device_assign),
        new_attrs,
        devices=ind.devices,
        req_type_num=ind.req_type_num,
        batch_size=int(getattr(ind, "batch_size", 1)),
        sub_graph_batch_sizes=dict(getattr(ind, "sub_graph_batch_sizes", {})),
    )


def topology_mutation(ind: Individual, cfg: EvoConfig, init_cfg: InitConfig, *, devices: Sequence[int], device_type_by_id: Optional[Dict[int, str]] = None) -> Individual:
    """Backward-compatible alias for the new rewrite-based mutation."""
    return rewrite_mutation(ind, cfg, init_cfg, devices=devices, device_type_by_id=device_type_by_id)

def device_mutation(ind: Individual, *, p_swap=0.6, p_move=0.3, p_shuffle=0.1, device_type_by_id: Optional[Dict[int, str]] = None) -> Individual:
    """Backward-compatible alias. Device mutation is absorbed into rewrite mutation; this now only refines concrete mapping."""
    del p_swap, p_move, p_shuffle
    return mapping_refinement(ind, device_type_by_id=device_type_by_id)

def evolve(
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    *,
    req_type_num: int,
    devices: Sequence[int],
    root_init: RootInit,
    device_type_by_id: Optional[Dict[int, str]] = None,
    fitness_fn: Callable[[Any, int], float],
    # 使用经验进行种群初始化
    with_pop_seeds: bool = False,
    pop_seed_roots: Optional[Sequence[Any]] = None,
    pop_seed_individuals: Optional[Sequence[Individual]] = None,
    # decoder 是否把 HW 节点挂在 leaf 下（一般 True）
    attach_hardware_leaves: bool = True,
    # 随机数种子
    random_seed: Optional[int] = None,
    dse_out: Optional[str] = None,

) -> Tuple[Individual, List[Individual]]:
    '''
    # 该函数的关键行为/隐含假设
    # 1. 丢弃策略：任何非法/解码失败/仿真失败就扔掉（不会修复）
    # 2. v3的变异：rewrite-based mutation，直接在 symbolic 上改，允许多步，允许不完全 materialize（但必须 fully decode 成 topology），只要最后能 materialize 就行。相比 v2 的“单步、完全 materialize、失败回退”更灵活，能探索更大空间。
    # 3. 缓存：相同 uid 的个体不会重复跑仿真，能省很多时间（仿真很贵，这点很关键）
    # 4. attempt_limit：防止“丢弃太多导致卡死”
    '''

    if random_seed is not None:
        random.seed(random_seed)

    devices_list = list(devices)

    if not with_pop_seeds:
        pop = initialize_population(
            init_cfg,
            evo_cfg,
            req_type_num=req_type_num,
            devices=devices_list,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
            device_type_by_id=device_type_by_id,
        )
    else:
        pop = initialize_population_with_seeds(
            init_cfg,
            evo_cfg,
            req_type_num=req_type_num,
            devices=devices_list,
            root_init=root_init,
            fitness_fn=fitness_fn,
            seed_roots=pop_seed_roots,
            seed_individuals=pop_seed_individuals,
            attach_hardware_leaves=attach_hardware_leaves,
            strict_device_partition=True,
            device_type_by_id=device_type_by_id,
        )

    for init_ind in pop:
        # dse scatter point
        if dse_out:
            log_individual_json(init_ind, dse_out)
            print("\n [init ind] ", init_ind.uid, init_ind.batch_size, init_ind.throughput, init_ind.latency)
            print(format_topology(init_ind, True, True))
            print("\n")
            save_individual_json(init_ind, f"./debug/individuals/init_{init_ind.uid}.json")

    cache: Dict[str, Tuple[float, float]] = {}

    def eval_ind(ind: Individual) -> Optional[Individual]:
        try:
            # 合法性检查共检查3部分：
            # 1. topology 合法（DP/PP/TP 顺序约束、XP 子数=2）
            # 2. device partition 完整、无重复、覆盖全部 devices
            # 3. attrs shape 匹配“有效 arity”（含 leaf=设备组大小）
            ind.check_legality()
        except Exception:
            return None
        key = canonical_key(ind)
        ind.uid = key
        if evo_cfg.enable_cache and key in cache:
            thr, lat = cache[key]
            ind.throughput = thr
            ind.latency = lat
            ind.objectives = (thr, lat)
            return ind
        local_root_init = copy.deepcopy(root_init)
        root = try_decode_to_root(ind, local_root_init, attach_hardware_leaves=attach_hardware_leaves)
        if root is None:
            return None
        try:
            res = fitness_fn(root, ind.batch_size)
            thr, lat = _parse_objectives(res)
            if (not math.isfinite(thr)) or (not math.isfinite(lat)):
                return None
            ind.throughput = thr
            ind.latency = lat
            ind.objectives = (thr, lat)

            # dse scatter point
            if dse_out:
                log_individual_json(ind, dse_out)
                print("\n [eval ind] ", ind.uid, ind.batch_size, ind.throughput, ind.latency)
                print(format_topology(ind, True, True))
                save_individual_json(ind, f"./debug/individuals/eval_{ind.uid}.json")
                print("\n")


        except Exception as e:
            print("evaluate/write failed:", repr(e))
            # fitness 阶段报错（含 ValueError），视为 invalid 个体丢弃
            return None

        if evo_cfg.enable_cache:
            cache[key] = (float(ind.throughput), float(ind.latency))
        return ind

    # 默认每代新产生 population_size - elite_size 个 offspring
    offspring_target = evo_cfg.offspring_size or init_cfg.population_size
    attempt_limit = evo_cfg.max_attempt_factor * max(1, offspring_target)

    # 每一代主要做四件事：排序 → 精英保留 → 生成 offspring → 更新种群
    for _gen in range(evo_cfg.generations):
        print("gen:", _gen)

        # 1) Assign Pareto rank + crowding distance for current population
        fronts = fast_nondominated_sort(pop)
        for f in fronts:
            crowding_distance(f)

        if fronts and fronts[0]:
            # Print a quick snapshot of the current Pareto front (top few by throughput)
            snap = sorted(fronts[0], key=lambda x: float('-inf') if x.throughput is None else x.throughput, reverse=True)[:5]
            print("  front0 size:", len(fronts[0]), " snapshot:", [(x.throughput, x.latency) for x in snap])

        # 2) Create offspring
        offspring: List[Individual] = []
        seen: set[str] = set()
        attempts = 0

        # 循环生成 offspring
        while len(offspring) < offspring_target and attempts < attempt_limit:
            attempts += 1
            # 选择父代，锦标赛选择：从 pop 随机抽 k 个，取 fitness 最大的那个。k 越大选择压力越强。
            parent = tournament_select(pop, evo_cfg.tournament_k)

            r = random.random()
            if r < evo_cfg.p_rewrite_mut:
                child = rewrite_mutation(
                    parent,
                    evo_cfg,
                    init_cfg,
                    devices=devices_list,
                    device_type_by_id=device_type_by_id,
                )
                mutation_kind = "topology"
            elif r < evo_cfg.p_rewrite_mut + evo_cfg.p_numeric_mut:
                child = numeric_mutation(parent, evo_cfg, init_cfg)
                mutation_kind = "numeric"
            else:
                child = mapping_refinement(parent, device_type_by_id=device_type_by_id)
                mutation_kind = "mapping_refine"

            # Always resample batch_size on every mutation, then mutate sub-graph batch sizes.
            # NOTE: some mutation ops may return the parent object; avoid in-place edits.
            child_batch_size = int(random.choice(init_cfg.batch_size_choices))
            child = copy.deepcopy(child)
            child.batch_size = child_batch_size
            child.sub_graph_batch_sizes=_mutate_sub_graph_batch_sizes(
                parent,
                Individual(
                    topology=copy.deepcopy(child.topology),
                    device_assign=copy.deepcopy(child.device_assign),
                    attrs=copy.deepcopy(child.attrs),
                    devices=list(child.devices),
                    req_type_num=child.req_type_num,
                    batch_size=child_batch_size,
                    sub_graph_batch_sizes=dict(getattr(child, "sub_graph_batch_sizes", {})),
                ),
                mutation_kind,
                evo_cfg.enable_subgraph_batch_mut,
            )

            child2 = eval_ind(child)
            if child2 is None or child2.uid is None or child2.objectives is None:
                continue
            # 去重（按 uid）
            if child2.uid in seen:
                continue
            if child2.throughput is None or child2.latency is None:
                continue
            if math.isnan(child2.throughput) or math.isnan(child2.latency):
                continue

            seen.add(child2.uid)
            offspring.append(child2)

        # If offspring not enough, pad with random individuals from current pop
        # offspring 不够时，用旧种群补齐。即，如果因为丢弃太多导致 offspring 产量不足，就从上一代非精英里随机抽一些填充，避免种群规模缩水。
        if len(offspring) < offspring_target:
            pool = pop
            if pool:
                offspring.extend(random.sample(pool, k=min(offspring_target - len(offspring), len(pool))))

        # 3) Environmental selection (NSGA-II): parents + offspring -> next generation
        pop = nsga2_environmental_select(pop + offspring, init_cfg.population_size)
    # Final: compute Pareto info and return a representative "best" + the final population
    fronts = fast_nondominated_sort(pop)
    for f in fronts:
        crowding_distance(f)

    # Choose representative best: highest-throughput solution on the first Pareto front.
    if fronts and fronts[0]:
        best = max(fronts[0], key=lambda x: float('-inf') if x.throughput is None else x.throughput)
    else:
        # fallback
        best = max(pop, key=lambda x: float('-inf') if x.throughput is None else x.throughput)

    # Sort population for readability (rank asc, then throughput desc)
    pop.sort(key=lambda x: (
        10**9 if x.pareto_rank is None else x.pareto_rank,
        float('-inf') if x.throughput is None else -x.throughput,
    ))
    return best, pop