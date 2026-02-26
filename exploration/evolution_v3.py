# evolution_v3.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import random
import math
import hashlib
from collections import deque

from parallelism.pnode import Parallelism, XpTag

from exploration.individual_v3 import (
    Attrs,
    DeviceAssign,
    Individual,
    Topology,
    TopologyNodeGene,
)
from exploration.decoder_v3 import RootInit, try_decode_to_root
from exploration.seed_from_pcase_v3 import individual_from_pcase_root
from exploration.ind_io_v3 import print_individual


@dataclass
class InitConfig:
    population_size: int = 50
    max_depth: int = 4
    max_children: int = 4
    p_stop_expand: float = 0.35

    # XP MUST be exactly one ATTENTION and one LINEAR
    xp_swap_prob: float = 0.5

    shuffle_devices_per_individual: bool = True


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

    s = repr((topo_items, attrs_items, dev_items)).encode("utf-8")
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
    fitness_fn: Callable[[Any], float],
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
        )

        try:
            ind.check_legality()
        except Exception:
            continue

        root = try_decode_to_root(ind, root_init, attach_hardware_leaves=attach_hardware_leaves)
        if root is None:
            continue

        # ind.fitness = fitness_fn(root)
        try:
            ind.fitness = fitness_fn(root)
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
    fitness_fn: Callable[[Any], float],
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
                ind.fitness = fitness_fn(root)
            except Exception:
                return
            ind.uid = canonical_key(ind)
            if ind.uid in seen:
                return
            seen.add(ind.uid)
            pop.append(ind)
            print("seed:", ind.uid, ind.fitness)
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
                    strict_device_partition=strict_device_partition,
                )
            except Exception:
                continue
            try_add(ind)
            print("------ seed ------")
            print_individual(ind)
            print("\n")


    # If too many seeds, keep best N
    if len(pop) > init_cfg.population_size:
        pop.sort(key=lambda x: float("-inf") if x.fitness is None else x.fitness, reverse=True)
        pop = pop[: init_cfg.population_size]
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

    # final sort (optional)
    pop.sort(key=lambda x: float("-inf") if x.fitness is None else x.fitness, reverse=True)
    return pop

def tournament_select(pop: List[Individual], k: int) -> Individual:
    cand = random.sample(pop, k=min(k, len(pop)))
    return max(cand, key=lambda x: float('-inf') if x.fitness is None else x.fitness)


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

    return Individual(topo, da, new_attrs, devices=ind.devices, req_type_num=ind.req_type_num)


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


def topology_mutation(ind: Individual, cfg: EvoConfig, init_cfg: InitConfig, *, devices: Sequence[int]) -> Individual:
    topo = ind.topology
    candidates = [nid for nid in topo.iter_dfs() if topo.gene(nid).ptype != Parallelism.XP]
    if not candidates:
        return ind
    target = random.choice(candidates)

    # editable nodes
    id2gene: Dict[int, TopologyNodeGene] = {g.node_id: g for g in topo.nodes}
    parent2children: Dict[int, List[int]] = {g.node_id: [] for g in topo.nodes}
    for g in topo.nodes:
        if g.parent_id != -1:
            parent2children[g.parent_id].append(g.node_id)

    for pid, ch in parent2children.items():
        ch.sort(key=lambda cid: id2gene[cid].child_slot)
        for s, cid in enumerate(ch):
            gg = id2gene[cid]
            if gg.child_slot != s:
                id2gene[cid] = TopologyNodeGene(node_id=cid, parent_id=pid, ptype=gg.ptype, child_slot=s)

    ch = parent2children.get(target, [])
    k = len(ch)
    do_add = (random.random() < cfg.p_add_child)

    next_id = max(id2gene.keys()) + 1
    new_nodes: List[TopologyNodeGene] = list(id2gene.values())

    if do_add:
        if k >= init_cfg.max_children:
            return ind
        seen_pp, seen_tp = _path_seen_flags(topo, target)
        new_type = random.choice(_allowed_types(seen_pp, seen_tp))
        new_nodes.append(TopologyNodeGene(node_id=next_id, parent_id=target, ptype=new_type, child_slot=k))
    else:
        if k <= 2:
            return ind
        remove_cid = random.choice(ch)

        child_map: Dict[int, List[int]] = {}
        for g in new_nodes:
            child_map.setdefault(g.parent_id, []).append(g.node_id)

        to_remove = set()
        stack = [remove_cid]
        while stack:
            u = stack.pop()
            if u in to_remove:
                continue
            to_remove.add(u)
            for v in child_map.get(u, []):
                stack.append(v)

        new_nodes = [g for g in new_nodes if g.node_id not in to_remove]

        remain = [cid for cid in ch if cid not in to_remove]
        remain.sort(key=lambda cid: id2gene[cid].child_slot)

        tmp = {g.node_id: g for g in new_nodes}
        for s, cid in enumerate(remain):
            gg = tmp[cid]
            tmp[cid] = TopologyNodeGene(node_id=cid, parent_id=target, ptype=gg.ptype, child_slot=s)
        new_nodes = list(tmp.values())

    try:
        new_topo = Topology(nodes=new_nodes)
        new_topo.check_legality()
    except Exception:
        return ind

    # resample device partition and attrs (because leaf arities change)
    try:
        new_da = sample_device_assign(new_topo, list(devices), shuffle=init_cfg.shuffle_devices_per_individual)
    except Exception:
        return ind
    new_attrs = sample_attrs(new_topo, new_da, req_type_num=ind.req_type_num, init_cfg=init_cfg)

    return Individual(new_topo, new_da, new_attrs, devices=list(devices), req_type_num=ind.req_type_num)


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
    )
    return child

def evolve(
    init_cfg: InitConfig,
    evo_cfg: EvoConfig,
    *,
    req_type_num: int,
    devices: Sequence[int],
    root_init: RootInit,
    fitness_fn: Callable[[Any], float],
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
    # 1. fitness 越大越好（排序是 reverse=True）
    # 2. 丢弃策略：任何非法/解码失败/仿真失败就扔掉（不会修复）
    # 3. v3 的关键约束落点在：
    #   sample_device_assign()：devices 被 partition 给 leaf
    #   sample_attrs()：leaf 的 arity = device_group_size
    #   check_legality()：保证 attrs shape 与这个 arity 一致
    #   这样 ptraversal 在 leaf→HW 边上调用 derive_child_info 不会越界
    # 4. 缓存：相同 uid 的个体不会重复跑仿真，能省很多时间（仿真很贵，这点很关键）
    # 5. attempt_limit：防止“丢弃太多导致卡死”
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

    cache: Dict[str, float] = {}

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
            ind.fitness = cache[key]
            return ind
        root = try_decode_to_root(ind, root_init, attach_hardware_leaves=attach_hardware_leaves)
        if root is None:
            return None

        # ind.fitness = fitness_fn(root)
        try:
            ind.fitness = fitness_fn(root)
        except Exception:
            # fitness 阶段报错（含 ValueError），视为 invalid 个体丢弃
            return None

        if evo_cfg.enable_cache:
            cache[key] = ind.fitness
        return ind

    # 默认每代新产生 population_size - elite_size 个 offspring
    offspring_target = evo_cfg.offspring_size or (init_cfg.population_size - evo_cfg.elite_size)
    attempt_limit = evo_cfg.max_attempt_factor * max(1, offspring_target)

    # 每一代主要做四件事：排序 → 精英保留 → 生成 offspring → 更新种群
    for _gen in range(evo_cfg.generations):
        print("gen:", _gen)
        # fitness 越大越好；精英直接原样进入下一代
        pop.sort(key=lambda x: float('-inf') if x.fitness is None else x.fitness, reverse=True)
        elites = pop[:evo_cfg.elite_size]

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
            # 评估 child（合法性 + decode + fitness）
            child2 = eval_ind(child)
            if child2 is None or child2.uid is None:
                continue
            # 去重（按 uid）
            if child2.uid in seen:
                continue
            if math.isnan(child2.fitness):
                continue

            seen.add(child2.uid)
            offspring.append(child2)

            print(child2.uid, child2.fitness)

        # offspring 不够时，用旧种群补齐
        # 即，如果因为丢弃太多导致 offspring 产量不足，就从上一代非精英里随机抽一些填充，避免种群规模缩水。
        if len(offspring) < offspring_target:
            pool = pop[evo_cfg.elite_size:]
            if pool:
                offspring.extend(random.sample(pool, k=min(offspring_target - len(offspring), len(pool))))

        pop = elites + offspring

    pop.sort(key=lambda x: float('-inf') if x.fitness is None else x.fitness, reverse=True)
    return pop[0], pop
