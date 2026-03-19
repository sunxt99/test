# evolution_v3.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import random
import math
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
from exploration.ind_io import print_individual

@dataclass
class InitConfig:
    population_size: int = 50
    max_depth: int = 4
    max_children: int = 4
    p_stop_expand: float = 0.35

    # XP MUST be exactly one ATTENTION and one LINEAR
    xp_swap_prob: float = 0.5

    shuffle_devices_per_individual: bool = True

    # Per-individual batch size search space (resampled on every mutation)
    batch_size_choices: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

@dataclass
class EvoConfig:
    generations: int = 50
    elite_size: int = 5
    offspring_size: Optional[int] = None

    p_topology_mut: float = 0.15
    p_numeric_mut: float = 0.50
    p_device_mut: float = 0.35

    weight_noise_sigma: float = 0.25

    p_add_child: float = 0.5
    max_attempt_factor: int = 20

    tournament_k: int = 3
    enable_cache: bool = True


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

    s = repr((topo_items, attrs_items, dev_items, int(getattr(ind, "batch_size", 1)))).encode("utf-8")
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


def initialize_population(
    init_cfg: InitConfig,
    *,
    req_type_num: int,
    devices: Sequence[int],
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    attach_hardware_leaves: bool = True,
) -> List[Individual]:
    pop: List[Individual] = []
    attempts = 0
    max_attempts = init_cfg.population_size * 120

    devices_list = list(devices)

    while len(pop) < init_cfg.population_size and attempts < max_attempts:
        attempts += 1
        topo = random_topology(init_cfg)

        try:
            da = sample_device_assign(topo, devices_list, shuffle=init_cfg.shuffle_devices_per_individual)
        except Exception:
            continue

        attrs = sample_attrs(topo, da, req_type_num=req_type_num, init_cfg=init_cfg)

        ind = Individual(
            topology=topo,
            device_assign=da,
            attrs=attrs,
            devices=devices_list,
            req_type_num=req_type_num,
            batch_size=int(random.choice(init_cfg.batch_size_choices)),
        )

        try:
            ind.check_legality()
        except Exception:
            continue

        root = try_decode_to_root(ind, root_init, attach_hardware_leaves=attach_hardware_leaves)
        if root is None:
            continue

        try:
            res = fitness_fn(root, ind.batch_size)
            thr, lat = _parse_objectives(res)
            ind.throughput = thr
            ind.latency = lat
            ind.objectives = (thr, lat)
        except Exception:
            continue

        ind.uid = canonical_key(ind)
        pop.append(ind)

    if len(pop) < init_cfg.population_size:
        raise RuntimeError(f"Init failed: {len(pop)}/{init_cfg.population_size} after {attempts} attempts.")
    return pop

def initialize_population_with_seeds(
    init_cfg: InitConfig,
    *,
    req_type_num: int,
    devices: Sequence[int],
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    seed_roots: Optional[Sequence[Any]] = None,  # pcase roots (BasicNode)
    seed_individuals: Optional[Sequence[Individual]] = None,
    attach_hardware_leaves: bool = True,
    strict_device_partition: bool = True,
) -> List[Individual]:
    """
    Initialize population with optional seed strategies (from pcase or pre-built Individuals),
    then fill the remainder by random initialization (same as v3).

    Rules:
      - Seeds are evaluated and deduplicated by uid.
      - Any seed that fails legality/decoding/fitness is discarded (consistent with your 'discard invalid' rule).
      - If seeds exceed population_size, we keep the top-N by fitness.
    """
    pop: List[Individual] = []
    seen: set[str] = set()
    devices_list = list(devices)

    def try_add(ind: Individual) -> None:
        nonlocal pop, seen
        try:
            root = try_decode_to_root(ind, root_init, attach_hardware_leaves=attach_hardware_leaves)
            if root is None:
                return
            try:
                res = fitness_fn(root, ind.batch_size)
                thr, lat = _parse_objectives(res)
                ind.throughput = thr
                ind.latency = lat
                ind.objectives = (thr, lat)
            except Exception:
                return
            ind.uid = canonical_key(ind)
            if ind.uid in seen:
                return
            seen.add(ind.uid)
            pop.append(ind)
            print("seed:", ind.uid, ind.throughput, ind.latency)
        except Exception:
            return

    # 1) seed_individuals directly
    if seed_individuals:
        for ind in seed_individuals:
            try_add(ind)

    # 2) seed_roots from pcase
    if seed_roots:
        for r in seed_roots:
            try:
                ind = individual_from_pcase_root(
                    r,
                    devices=devices_list,
                    req_type_num=req_type_num,
                    batch_size=int(random.choice(init_cfg.batch_size_choices)),
                    strict_device_partition=strict_device_partition,
                )
            except Exception:
                continue
            try_add(ind)
            print("------ seed ------")
            print_individual(ind)
            print("\n")


    # If too many seeds, keep best N (Pareto / NSGA-II environmental selection)
    if len(pop) > init_cfg.population_size:
        # compute ranks/crowding then select
        pop = nsga2_environmental_select(pop, init_cfg.population_size)
        return pop

    # 3) Fill remaining using original random initializer
    need = init_cfg.population_size - len(pop)
    if need > 0:
        # reuse initialize_population to get `need` individuals, but it returns full size population;
        # we call it with a temporary config and then merge/dedup.
        tmp_cfg = InitConfig(
            population_size=need,
            max_depth=init_cfg.max_depth,
            max_children=init_cfg.max_children,
            p_stop_expand=init_cfg.p_stop_expand,
            xp_swap_prob=init_cfg.xp_swap_prob,
            shuffle_devices_per_individual=init_cfg.shuffle_devices_per_individual,
            batch_size_choices=init_cfg.batch_size_choices,
        )
        rand_pop = initialize_population(
            tmp_cfg,
            req_type_num=req_type_num,
            devices=devices_list,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
        )
        for ind in rand_pop:
            if ind.uid is None:
                ind.uid = canonical_key(ind)
            if ind.uid in seen:
                continue
            seen.add(ind.uid)
            pop.append(ind)
            if len(pop) >= init_cfg.population_size:
                break

    # final sort (optional): by Pareto rank then crowding (NSGA-II)
    fronts = fast_nondominated_sort(pop)
    for f in fronts:
        crowding_distance(f)
    pop.sort(key=lambda x: (10**9 if x.pareto_rank is None else x.pareto_rank, -x.crowding))
    return pop

def _parse_objectives(fitness_result: Any) -> Tuple[float, float]:
    """Parse evaluator output into (throughput, latency).

    Supported formats:
      - (throughput, latency)
      - {"throughput": ..., "latency": ...} (case-insensitive keys are also accepted)
      - {"tps": ..., "p95": ...} etc. (best-effort)
      - a single float -> treated as throughput, latency=+inf (keeps backward compatibility)
    """
    if fitness_result is None:
        raise ValueError("fitness_result is None")

    # tuple/list
    if isinstance(fitness_result, (tuple, list)) and len(fitness_result) >= 2:
        thr = float(fitness_result[0])
        lat = float(fitness_result[1])
        return thr, lat

    # dict
    if isinstance(fitness_result, dict):
        lower = {str(k).lower(): v for k, v in fitness_result.items()}

        def pick(*names):
            for n in names:
                if n in lower:
                    return lower[n]
            return None

        thr = pick("throughput", "tps", "qps", "req_s", "req_per_s", "requests_per_s")
        lat = pick("latency", "p95_latency", "p99_latency", "p90_latency", "tail_latency", "p95", "p99", "lat_ms", "lat_s")

        if thr is None or lat is None:
            raise ValueError(f"Cannot parse objectives from dict keys={sorted(lower.keys())}")
        return float(thr), float(lat)

    # single number: old behavior (maximize this)
    if isinstance(fitness_result, (int, float)):
        return float(fitness_result), float("inf")

    raise ValueError(f"Unsupported fitness_result type: {type(fitness_result)}")


def dominates(a: Individual, b: Individual) -> bool:
    """Return True if a Pareto-dominates b (maximize throughput, minimize latency)."""
    if a.objectives is None or b.objectives is None:
        return False
    athr, alat = a.objectives
    bthr, blat = b.objectives
    return (athr >= bthr and alat <= blat) and (athr > bthr or alat < blat)


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

    return Individual(topo, da, new_attrs, devices=ind.devices, req_type_num=ind.req_type_num, batch_size=int(getattr(ind, "batch_size", 1)))

def _path_seen_flags(topo: Topology, node_id: int) -> Tuple[bool, bool]:
    seen_pp = False
    seen_tp = False
    cur = node_id
    while cur != -1:
        t = topo.gene(cur).ptype
        if t == Parallelism.PP:
            seen_pp = True
        if t == Parallelism.TP:
            seen_tp = True
        cur = topo.parent_of(cur)
    return seen_pp, seen_tp


def _collect_subtree_nodes(nodes: List[TopologyNodeGene], root_id: int) -> set[int]:
    child_map: Dict[int, List[int]] = {}
    for g in nodes:
        child_map.setdefault(g.parent_id, []).append(g.node_id)
    to_remove: set[int] = set()
    stack = [root_id]
    while stack:
        u = stack.pop()
        if u in to_remove:
            continue
        to_remove.add(u)
        for v in child_map.get(u, []):
            stack.append(v)
    return to_remove


def _reindex_child_slots(nodes: List[TopologyNodeGene]) -> List[TopologyNodeGene]:
    id2gene = {g.node_id: g for g in nodes}
    parent2children: Dict[int, List[int]] = {}
    for g in nodes:
        parent2children.setdefault(g.parent_id, []).append(g.node_id)

    out: Dict[int, TopologyNodeGene] = dict(id2gene)
    for pid, ch in parent2children.items():
        ch.sort(key=lambda cid: out[cid].child_slot)
        for s, cid in enumerate(ch):
            gg = out[cid]
            if gg.child_slot != s:
                out[cid] = TopologyNodeGene(node_id=cid, parent_id=pid, ptype=gg.ptype, child_slot=s)
    return list(out.values())


def _append_random_subtree(
    base_nodes: List[TopologyNodeGene],
    parent_id: int,
    child_slot: int,
    *,
    start_id: int,
    depth_left: int,
    init_cfg: InitConfig,
    seen_pp: bool,
    seen_tp: bool,
) -> Tuple[List[TopologyNodeGene], int]:
    """Create a new child under parent_id (at child_slot) and optionally expand it.

    Returns (new_nodes, next_free_id).
    """
    nid = start_id
    ctype = random.choice(_allowed_types(seen_pp, seen_tp))
    base_nodes.append(TopologyNodeGene(node_id=nid, parent_id=parent_id, ptype=ctype, child_slot=child_slot))
    nid += 1

    # Force XP to have children.
    q: deque[Tuple[int, int, bool, bool]] = deque()
    q.append((start_id, depth_left - 1, seen_pp or (ctype == Parallelism.PP), seen_tp or (ctype == Parallelism.TP)))

    id2ptype: Dict[int, Parallelism] = {start_id: ctype}

    while q:
        cur, dl, spp, stp = q.popleft()
        t = id2ptype[cur]

        # stop condition: allow leaf for non-XP
        if dl <= 0 or random.random() < init_cfg.p_stop_expand:
            if t == Parallelism.XP:
                pass
            else:
                continue

        max_k = max(2, init_cfg.max_children)
        k = 2 if t == Parallelism.XP else random.randint(2, max_k)
        for slot in range(k):
            c2 = random.choice(_allowed_types(spp, stp))
            base_nodes.append(TopologyNodeGene(node_id=nid, parent_id=cur, ptype=c2, child_slot=slot))
            id2ptype[nid] = c2
            q.append((nid, dl - 1, spp or (c2 == Parallelism.PP), stp or (c2 == Parallelism.TP)))
            nid += 1

    return base_nodes, nid


def topology_mutation(ind: Individual, cfg: EvoConfig, init_cfg: InitConfig, *, devices: Sequence[int]) -> Individual:
    """Stronger topology mutation (v2).

    Operators mixed:
      - add/remove a child under a random non-XP node (original behavior)
      - random subtree deletion (safe constraints)
      - subtree replacement (replace a subtree with a freshly sampled subtree)
      - node type mutation (change DP/PP/TP/XP with local repairs)

    Any failure returns the original individual.
    """
    topo = ind.topology
    nodes0: List[TopologyNodeGene] = list(topo.nodes)

    # choose operator
    r = random.random()
    if r < 0.25:
        op = "add_remove"
    elif r < 0.50:
        op = "subtree_delete"
    elif r < 0.75:
        op = "subtree_replace"
    else:
        op = "type_mutate"

    # convenience maps
    id2gene0: Dict[int, TopologyNodeGene] = {g.node_id: g for g in nodes0}
    parent2children0: Dict[int, List[int]] = {g.node_id: [] for g in nodes0}
    for g in nodes0:
        if g.parent_id != -1:
            parent2children0.setdefault(g.parent_id, []).append(g.node_id)
    for pid, ch in parent2children0.items():
        ch.sort(key=lambda cid: id2gene0[cid].child_slot)

    next_id = max(id2gene0.keys()) + 1
    new_nodes: List[TopologyNodeGene] = list(nodes0)

    def _finalize(nodes: List[TopologyNodeGene]) -> Optional[Individual]:
        try:
            nodes = _reindex_child_slots(nodes)
            new_topo = Topology(nodes=nodes)
            new_topo.check_legality()
        except Exception:
            return None

        try:
            new_da = sample_device_assign(new_topo, list(devices), shuffle=init_cfg.shuffle_devices_per_individual)
        except Exception:
            return None
        new_attrs = sample_attrs(new_topo, new_da, req_type_num=ind.req_type_num, init_cfg=init_cfg)
        return Individual(new_topo, new_da, new_attrs, devices=list(devices), req_type_num=ind.req_type_num, batch_size=int(getattr(ind, "batch_size", 1)))

    # -------- operator: add/remove (original) --------
    if op == "add_remove":
        candidates = [nid for nid in topo.iter_dfs() if topo.gene(nid).ptype != Parallelism.XP]
        if not candidates:
            return ind
        target = random.choice(candidates)

        # current children
        ch = list(parent2children0.get(target, []))
        k = len(ch)
        do_add = (random.random() < cfg.p_add_child)

        if do_add:
            if k >= init_cfg.max_children:
                return ind
            seen_pp, seen_tp = _path_seen_flags(topo, target)
            new_type = random.choice(_allowed_types(seen_pp, seen_tp))
            new_nodes.append(TopologyNodeGene(node_id=next_id, parent_id=target, ptype=new_type, child_slot=k))
            out = _finalize(new_nodes)
            return out if out is not None else ind

        # remove a random child subtree (but keep >=2 children)
        if k <= 2:
            return ind
        remove_cid = random.choice(ch)
        to_remove = _collect_subtree_nodes(new_nodes, remove_cid)
        new_nodes = [g for g in new_nodes if g.node_id not in to_remove]
        out = _finalize(new_nodes)
        return out if out is not None else ind

    # -------- operator: safe subtree deletion --------
    if op == "subtree_delete":
        # pick a node that is not root
        cand = [nid for nid in topo.iter_dfs() if nid != topo.root_id]
        if not cand:
            return ind
        u = random.choice(cand)
        parent = topo.parent_of(u)
        if parent == -1:
            return ind

        # cannot delete if parent is XP (must keep exactly 2 children)
        if topo.gene(parent).ptype == Parallelism.XP:
            return ind

        siblings = list(parent2children0.get(parent, []))
        if len(siblings) <= 2:
            return ind  # would violate >=2 child rule for non-XP nodes in this design

        to_remove = _collect_subtree_nodes(new_nodes, u)
        new_nodes = [g for g in new_nodes if g.node_id not in to_remove]
        out = _finalize(new_nodes)
        return out if out is not None else ind

    # -------- operator: subtree replacement --------
    if op == "subtree_replace":
        cand = [nid for nid in topo.iter_dfs() if nid != topo.root_id]
        if not cand:
            return ind
        u = random.choice(cand)
        parent = topo.parent_of(u)
        if parent == -1:
            return ind

        # determine the slot of u under parent
        u_slot = topo.gene(u).child_slot

        # remove u subtree
        to_remove = _collect_subtree_nodes(new_nodes, u)
        new_nodes = [g for g in new_nodes if g.node_id not in to_remove]

        # add a new subtree root under parent at the same slot
        seen_pp, seen_tp = _path_seen_flags(topo, parent)
        try:
            new_nodes, next_id2 = _append_random_subtree(
                new_nodes,
                parent_id=parent,
                child_slot=u_slot,
                start_id=next_id,
                depth_left=max(1, init_cfg.max_depth // 2),
                init_cfg=init_cfg,
                seen_pp=seen_pp,
                seen_tp=seen_tp,
            )
        except Exception:
            return ind

        out = _finalize(new_nodes)
        return out if out is not None else ind

    # -------- operator: node type mutation --------
    if op == "type_mutate":
        cand = [nid for nid in topo.iter_dfs()]
        if not cand:
            return ind
        u = random.choice(cand)

        # don't mutate root to keep search stable? allow it but repair carefully
        parent = topo.parent_of(u)

        # allowed new types based on path
        seen_pp, seen_tp = _path_seen_flags(topo, u)
        # remove current node type from choices if possible
        cur_t = topo.gene(u).ptype
        choices = [t for t in _allowed_types(seen_pp, seen_tp) if t != cur_t]
        if not choices:
            return ind
        new_t = random.choice(choices)

        # update this node's ptype
        new_nodes2: List[TopologyNodeGene] = []
        for g in new_nodes:
            if g.node_id == u:
                new_nodes2.append(TopologyNodeGene(node_id=u, parent_id=g.parent_id, ptype=new_t, child_slot=g.child_slot))
            else:
                new_nodes2.append(g)
        new_nodes = new_nodes2

        # repair arity constraints for XP
        # build fresh child list for u
        id2g = {g.node_id: g for g in new_nodes}
        p2c: Dict[int, List[int]] = {}
        for g in new_nodes:
            p2c.setdefault(g.parent_id, []).append(g.node_id)
        for pid, ch in p2c.items():
            ch.sort(key=lambda cid: id2g[cid].child_slot)

        children_u = list(p2c.get(u, []))
        if new_t == Parallelism.XP:
            # must have exactly 2 children
            if len(children_u) > 2:
                # trim extra subtrees
                extra = children_u[2:]
                to_remove = set()
                for cid in extra:
                    to_remove |= _collect_subtree_nodes(new_nodes, cid)
                new_nodes = [g for g in new_nodes if g.node_id not in to_remove]
            elif len(children_u) < 2:
                # append new children subtrees
                spp, stp = _path_seen_flags(topo, u)
                base_slot = len(children_u)
                for add_i in range(2 - len(children_u)):
                    new_nodes, next_id = _append_random_subtree(
                        new_nodes,
                        parent_id=u,
                        child_slot=base_slot + add_i,
                        start_id=next_id,
                        depth_left=max(1, init_cfg.max_depth // 2),
                        init_cfg=init_cfg,
                        seen_pp=spp,
                        seen_tp=stp,
                    )
        # if changing from XP to others, keeping 2 children is fine.

        out = _finalize(new_nodes)
        return out if out is not None else ind

    return ind


def _rand_pos():
    return random.lognormvariate(0.0, 0.8)

def device_mutation(ind: Individual,
                    *,
                    p_swap=0.6,
                    p_move=0.3,
                    p_shuffle=0.1) -> Individual:
    """
    Mutate DeviceAssign while keeping:
      - exact partition of devices
      - every leaf non-empty

    Also updates leaf-node attrs when group sizes/order change.
    """
    topo = ind.topology
    leaves = sorted(topo.leaf_parallel_nodes())
    if len(leaves) <= 1:
        return ind

    # deep copy device assign
    new_da = DeviceAssign(
        leaf_to_devices={lid: grp[:] for lid, grp in ind.device_assign.leaf_to_devices.items()}
    )

    # deep copy attrs
    new_attrs = Attrs(
        dp_attr={nid: [row[:] for row in mat] for nid, mat in ind.attrs.dp_attr.items()},
        pp_attr={nid: vec[:] for nid, vec in ind.attrs.pp_attr.items()},
        tp_attr={nid: vec[:] for nid, vec in ind.attrs.tp_attr.items()},
        xp_attr={nid: tags[:] for nid, tags in ind.attrs.xp_attr.items()},
    )

    r = random.random()
    op = "swap" if r < p_swap else ("move" if r < p_swap + p_move else "shuffle")

    if op == "swap":
        a, b = random.sample(leaves, 2)
        ga, gb = new_da.leaf_to_devices[a], new_da.leaf_to_devices[b]
        ia = random.randrange(len(ga))
        ib = random.randrange(len(gb))
        ga[ia], gb[ib] = gb[ib], ga[ia]
        # sizes unchanged -> attrs shapes unchanged, no need to modify

    elif op == "move":
        # pick donor with size>=2
        donors = [lid for lid in leaves if len(new_da.leaf_to_devices[lid]) >= 2]
        if not donors:
            return ind
        donor = random.choice(donors)
        receiver = random.choice([lid for lid in leaves if lid != donor])

        gd, gr = new_da.leaf_to_devices[donor], new_da.leaf_to_devices[receiver]
        i = random.randrange(len(gd))
        moved_dev = gd.pop(i)
        # insert to receiver at random position
        j = random.randrange(len(gr) + 1)
        gr.insert(j, moved_dev)

        # update donor/receiver leaf attrs to match new sizes
        def adjust_leaf_attr(nid: int, remove_index: int | None = None, insert_index: int | None = None):
            t = topo.gene(nid).ptype
            if t == Parallelism.DP:
                mat = new_attrs.dp_attr[nid]
                for r in range(len(mat)):
                    if remove_index is not None:
                        mat[r].pop(remove_index)
                    if insert_index is not None:
                        mat[r].insert(insert_index, _rand_pos())
            elif t == Parallelism.PP:
                vec = new_attrs.pp_attr[nid]
                if remove_index is not None:
                    vec.pop(remove_index)
                if insert_index is not None:
                    vec.insert(insert_index, _rand_pos())
            elif t == Parallelism.TP:
                vec = new_attrs.tp_attr[nid]
                if remove_index is not None:
                    vec.pop(remove_index)
                if insert_index is not None:
                    vec.insert(insert_index, _rand_pos())
            else:
                # XP 不应该成为 leaf（通常 leaf->device 的节点多是 DP/PP/TP）
                pass

        # donor removed at i
        adjust_leaf_attr(donor, remove_index=i, insert_index=None)
        # receiver inserted at j
        adjust_leaf_attr(receiver, remove_index=None, insert_index=j)

    else:  # shuffle
        lid = random.choice(leaves)
        grp = new_da.leaf_to_devices[lid]
        if len(grp) <= 1:
            return ind
        perm = list(range(len(grp)))
        random.shuffle(perm)
        new_da.leaf_to_devices[lid] = [grp[i] for i in perm]

        # permute corresponding leaf attrs to keep device<->weight alignment
        t = topo.gene(lid).ptype
        if t == Parallelism.DP:
            mat = new_attrs.dp_attr[lid]
            for r in range(len(mat)):
                mat[r] = [mat[r][i] for i in perm]
        elif t == Parallelism.PP:
            vec = new_attrs.pp_attr[lid]
            new_attrs.pp_attr[lid] = [vec[i] for i in perm]
        elif t == Parallelism.TP:
            vec = new_attrs.tp_attr[lid]
            new_attrs.tp_attr[lid] = [vec[i] for i in perm]

    child = Individual(
        topology=topo,
        device_assign=new_da,
        attrs=new_attrs,
        devices=ind.devices,
        req_type_num=ind.req_type_num,
        batch_size=int(getattr(ind, "batch_size", 1)),
    )
    return child

def evolve(
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    *,
    req_type_num: int,
    devices: Sequence[int],
    root_init: RootInit,
    fitness_fn: Callable[[Any, int], float],
    # 使用经验进行种群初始化
    with_pop_seeds: bool = False,
    pop_seed_roots: Optional[Sequence[Any]] = None,
    pop_seed_individuals: Optional[Sequence[Individual]] = None,
    # decoder 是否把 HW 节点挂在 leaf 下（一般 True）
    attach_hardware_leaves: bool = True,
    # 随机数种子
    random_seed: Optional[int] = None,

) -> Tuple[Individual, List[Individual]]:
    '''
    # 该函数的关键行为/隐含假设
    # 1. 丢弃策略：任何非法/解码失败/仿真失败就扔掉（不会修复）
    # 2. v3 的关键约束落点在：
    #   sample_device_assign()：devices 被 partition 给 leaf
    #   sample_attrs()：leaf 的 arity = device_group_size
    #   check_legality()：保证 attrs shape 与这个 arity 一致
    #   这样 ptraversal 在 leaf→HW 边上调用 derive_child_info 不会越界
    # 3. 缓存：相同 uid 的个体不会重复跑仿真，能省很多时间（仿真很贵，这点很关键）
    # 4. attempt_limit：防止“丢弃太多导致卡死”
    '''


    if random_seed is not None:
        random.seed(random_seed)

    devices_list = list(devices)

    if not with_pop_seeds:
        pop = initialize_population(
            init_cfg,
            req_type_num=req_type_num,
            devices=devices_list,
            root_init=root_init,
            fitness_fn=fitness_fn,
            attach_hardware_leaves=attach_hardware_leaves,
        )
    else:
        pop = initialize_population_with_seeds(
            init_cfg,
            req_type_num=req_type_num,
            devices=devices_list,
            root_init=root_init,
            fitness_fn=fitness_fn,
            seed_roots=pop_seed_roots,
            seed_individuals=pop_seed_individuals,
            attach_hardware_leaves=attach_hardware_leaves,
            strict_device_partition=True,
        )

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
        root = try_decode_to_root(ind, root_init, attach_hardware_leaves=attach_hardware_leaves)
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

            with open("./result/DSE.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"T": thr, "L": lat}, ensure_ascii=False) + "\n")

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
            if r < evo_cfg.p_topology_mut:
                # topology_mutation：改树结构（加/删某个节点的一个并行子树），然后必须：
                # 1. 重新做 device partition（因为 leaf 集合可能变了）
                # 2. 重新生成 attrs（因为 leaf 有效 arity 变了）
                child = topology_mutation(parent, evo_cfg, init_cfg, devices=devices_list)
            # else:
            #     numeric_mutation：不改结构，只对 attrs 做“乘性噪声”扰动（DP/PP/TP）或 XP tag 交换等
                # child = numeric_mutation(parent, evo_cfg, init_cfg)
            elif r < evo_cfg.p_topology_mut + evo_cfg.p_device_mut:
                child = device_mutation(parent)
            else:
                child = numeric_mutation(parent, evo_cfg, init_cfg)

            # Always resample batch_size on every mutation.
            # NOTE: some mutation ops may return the parent object; avoid in-place edits.
            child = Individual(
                topology=child.topology,
                device_assign=child.device_assign,
                attrs=child.attrs,
                devices=list(child.devices),
                req_type_num=child.req_type_num,
                batch_size=int(random.choice(init_cfg.batch_size_choices)),
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