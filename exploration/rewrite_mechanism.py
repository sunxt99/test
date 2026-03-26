from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import random

from parallelism.pnode import Parallelism, XpTag
from exploration.individual import Attrs, DeviceAssign, Individual, Topology, TopologyNodeGene


class RewriteFamily(str, Enum):
    SKELETON_EXPANSION = "skeleton_expansion"
    LOCAL_REFINEMENT = "local_refinement"
    RELABEL = "relabel"
    REPARTITION = "repartition"
    ROLLBACK = "rollback"


@dataclass
class SymbolicNode:
    op: Parallelism
    device_counts: Dict[str, int] = field(default_factory=dict)
    children: List["SymbolicNode"] = field(default_factory=list)
    hint: Dict[str, Any] = field(default_factory=dict)
    closed: bool = True

    def clone(self) -> "SymbolicNode":
        return copy.deepcopy(self)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def total_devices(self) -> int:
        return sum(int(v) for v in self.device_counts.values())

    def normalized_device_counts(self) -> Dict[str, int]:
        return {str(k): int(v) for k, v in sorted(self.device_counts.items()) if int(v) > 0}

    def recompute_counts(self) -> Dict[str, int]:
        if self.children:
            merged: Dict[str, int] = {}
            for c in self.children:
                c.recompute_counts()
                for t, cnt in c.device_counts.items():
                    merged[t] = merged.get(t, 0) + int(cnt)
            self.device_counts = merged
        else:
            self.device_counts = self.normalized_device_counts()
        return self.device_counts

    def walk(self) -> Iterable["SymbolicNode"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def pretty(self) -> str:
        if self.children:
            body = ", ".join(c.pretty() for c in self.children)
            return f"{self.op.name}({body})"
        if self.device_counts:
            dc = ",".join(f"{k}*{v}" for k, v in sorted(self.device_counts.items()))
            return f"{self.op.name}[{dc}]"
        return f"{self.op.name}[]"


@dataclass
class PatternSpec:
    name: str
    family: RewriteFamily
    match: Dict[str, Any]
    rewrite: Dict[str, Any]
    weight: float = 1.0

    def matches(self, node: SymbolicNode) -> bool:
        return _match_symbolic_node(node, self.match)

    def apply(self, node: SymbolicNode) -> Optional[SymbolicNode]:
        if not self.matches(node):
            return None
        spec = copy.deepcopy(self.rewrite)
        out = _build_from_spec(spec)
        out.recompute_counts()
        return out



@dataclass
class NumericAttrCandidate:
    attrs: Dict[str, Any]
    weight: float = 1.0


@dataclass
class NumericPatternSpec:
    name: str
    match: Dict[str, Any]
    candidates: List[NumericAttrCandidate]
    weight: float = 1.0

    def matches(self, node: SymbolicNode) -> bool:
        return _match_symbolic_node(node, self.match)

    def choose_candidate(self, *, rng: Optional[random.Random] = None) -> Optional[Dict[str, Any]]:
        if not self.candidates:
            return None
        local_rng = rng or random.Random()
        weights = [max(1e-9, float(c.weight)) for c in self.candidates]
        cand = local_rng.choices(self.candidates, weights=weights, k=1)[0]
        return copy.deepcopy(cand.attrs)


@dataclass
class InitPatternSpec:
    name: str
    stratum: str
    root: Dict[str, Any]
    weight: float = 1.0

    def instantiate(self) -> "SymbolicNode":
        spec = copy.deepcopy(self.root)
        out = _build_from_spec(spec)
        out.recompute_counts()
        _ensure_materializable(out)
        return out


class RewriteEngine:
    def __init__(self, patterns: Sequence[PatternSpec], *, rng: Optional[random.Random] = None):
        self.patterns = list(patterns)
        self.rng = rng or random.Random()

    def rewrite_random(self, root: SymbolicNode, *, family: Optional[RewriteFamily] = None) -> bool:
        matches: List[Tuple[List[int], PatternSpec]] = []

        def dfs(node: SymbolicNode, path: List[int]) -> None:
            for p in self.patterns:
                if family is not None and p.family != family:
                    continue
                if p.matches(node):
                    matches.append((path[:], p))
            for i, c in enumerate(node.children):
                path.append(i)
                dfs(c, path)
                path.pop()

        dfs(root, [])
        if not matches:
            return False

        weights = [max(1e-9, float(p.weight)) for _, p in matches]
        path, pat = self.rng.choices(matches, weights=weights, k=1)[0]
        replacement = pat.apply(_get_node_by_path(root, path).clone())
        if replacement is None:
            return False
        _replace_node_by_path(root, path, replacement)
        root.recompute_counts()
        return True


# -----------------------------
# Manual, declarative rewrite DSL
# -----------------------------

def _normalize_counts(counts: Dict[str, int]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counts.items()) if int(v) > 0}


def _parallelism_from_any(v: Any) -> Parallelism:
    if isinstance(v, Parallelism):
        return v
    return Parallelism[str(v)]


def _effective_arity(node: SymbolicNode) -> int:
    return len(node.children) if node.children else node.total_devices()


def _match_symbolic_node(node: SymbolicNode, match: Dict[str, Any]) -> bool:
    op = match.get("op")
    if op is not None:
        want = op if isinstance(op, Parallelism) else Parallelism[str(op)]
        if node.op != want:
            return False

    leaf = match.get("leaf")
    if leaf is not None and bool(leaf) != node.is_leaf():
        return False

    closed = match.get("closed")
    if closed is not None and bool(closed) != bool(node.closed):
        return False

    child_count = match.get("child_count")
    if child_count is not None and len(node.children) != int(child_count):
        return False

    arity = match.get("arity")
    if arity is not None and _effective_arity(node) != int(arity):
        return False

    min_arity = match.get("min_arity")
    if min_arity is not None and _effective_arity(node) < int(min_arity):
        return False

    max_arity = match.get("max_arity")
    if max_arity is not None and _effective_arity(node) > int(max_arity):
        return False

    min_devices = match.get("min_total_devices")
    if min_devices is not None and node.total_devices() < int(min_devices):
        return False

    max_devices = match.get("max_total_devices")
    if max_devices is not None and node.total_devices() > int(max_devices):
        return False

    require_types = match.get("require_types")
    if require_types is not None:
        types = set(node.device_counts.keys())
        if not set(str(x) for x in require_types).issubset(types):
            return False

    exact_counts = match.get("device_counts")
    if exact_counts is not None and _normalize_counts(exact_counts) != _normalize_counts(node.device_counts):
        return False

    return True



def _build_from_spec(spec: Dict[str, Any]) -> SymbolicNode:
    children = [_build_from_spec(c) for c in spec.get("children", [])]
    op = _parallelism_from_any(spec["op"])
    node = SymbolicNode(
        op=op,
        device_counts=_normalize_counts(spec.get("device_counts", {})),
        children=children,
        hint=copy.deepcopy(spec.get("hint", {})),
        closed=bool(spec.get("closed", len(children) > 0 or op != Parallelism.XP)),
    )
    node.recompute_counts()
    return node


def _get_node_by_path(root: SymbolicNode, path: Sequence[int]) -> SymbolicNode:
    cur = root
    for idx in path:
        cur = cur.children[idx]
    return cur


def _replace_node_by_path(root: SymbolicNode, path: Sequence[int], node: SymbolicNode) -> None:
    if not path:
        root.op = node.op
        root.device_counts = node.device_counts
        root.children = node.children
        root.hint = node.hint
        root.closed = node.closed
        return
    parent = _get_node_by_path(root, path[:-1])
    parent.children[path[-1]] = node


# -----------------------------
# Individual <-> Symbolic conversion
# -----------------------------

def individual_to_symbolic(ind: Individual, *, device_type_by_id: Dict[int, str]) -> SymbolicNode:
    topo = ind.topology

    def build(nid: int) -> SymbolicNode:
        ptype = topo.gene(nid).ptype
        children = [build(cid) for cid in topo.children_of(nid)]
        hint: Dict[str, Any] = {}
        if ptype == Parallelism.DP and nid in ind.attrs.dp_attr:
            hint["dp_attr"] = copy.deepcopy(ind.attrs.dp_attr[nid])
        elif ptype == Parallelism.PP and nid in ind.attrs.pp_attr:
            hint["pp_attr"] = copy.deepcopy(ind.attrs.pp_attr[nid])
        elif ptype == Parallelism.TP and nid in ind.attrs.tp_attr:
            hint["tp_attr"] = copy.deepcopy(ind.attrs.tp_attr[nid])
        elif ptype == Parallelism.XP and nid in ind.attrs.xp_attr:
            hint["xp_attr"] = copy.deepcopy(ind.attrs.xp_attr[nid])

        if children:
            node = SymbolicNode(op=ptype, children=children, hint=hint, closed=True)
            node.recompute_counts()
            return node

        counts: Dict[str, int] = {}
        for did in ind.device_assign.leaf_to_devices.get(nid, []):
            dtype = str(device_type_by_id[int(did)])
            counts[dtype] = counts.get(dtype, 0) + 1
        leaf_closed = True
        if ptype == Parallelism.XP:
            leaf_closed = sum(int(v) for v in counts.values()) == 2
        return SymbolicNode(op=ptype, device_counts=counts, hint=hint, closed=leaf_closed)

    root = build(topo.root_id)
    root.recompute_counts()
    return root


def symbolic_to_individual(
    root: SymbolicNode,
    *,
    device_type_to_ids: Dict[str, List[int]],
    req_type_num: int,
    batch_size: int,
    devices: Sequence[int],
    sub_graph_batch_sizes: Optional[Dict[int, int]] = None,
) -> Individual:
    _ensure_materializable(root)
    root = root.clone()
    root.recompute_counts()

    pools: Dict[str, List[int]] = {str(k): list(v) for k, v in device_type_to_ids.items()}
    nodes: List[TopologyNodeGene] = []
    leaf_to_devices: Dict[int, List[int]] = {}
    attrs = Attrs()
    next_id = 0

    def build(node: SymbolicNode, parent_id: int, child_slot: int) -> int:
        nonlocal next_id
        nid = next_id
        next_id += 1
        nodes.append(TopologyNodeGene(node_id=nid, parent_id=parent_id, ptype=node.op, child_slot=child_slot))

        if node.children:
            for idx, ch in enumerate(node.children):
                build(ch, nid, idx)
            _materialize_attrs(node, nid, attrs, req_type_num=req_type_num, arity=len(node.children))
            return nid

        assigned: List[int] = []
        for dtype, cnt in sorted(node.device_counts.items()):
            bucket = pools.get(str(dtype), [])
            if len(bucket) < int(cnt):
                raise ValueError(f"Not enough devices for type {dtype}: need {cnt}, have {len(bucket)}")
            take = bucket[: int(cnt)]
            del bucket[: int(cnt)]
            assigned.extend(int(x) for x in take)
        if not assigned:
            raise ValueError(f"Leaf {node.op.name} must receive at least one device.")
        leaf_to_devices[nid] = assigned
        _materialize_attrs(node, nid, attrs, req_type_num=req_type_num, arity=len(assigned))
        return nid

    build(root, -1, 0)
    leftovers = [x for bucket in pools.values() for x in bucket]
    if leftovers:
        raise ValueError(f"Unused devices after materialization: {sorted(leftovers)}")

    topo = Topology(nodes=nodes)
    ind = Individual(
        topology=topo,
        device_assign=DeviceAssign(leaf_to_devices=leaf_to_devices),
        attrs=attrs,
        devices=list(devices),
        req_type_num=req_type_num,
        batch_size=int(batch_size),
        sub_graph_batch_sizes=dict(sub_graph_batch_sizes or {}),
    )
    ind.check_legality()
    return ind


# -----------------------------
# Materialization checks / attr filling
# -----------------------------

def _uniform_dp_attr(req_type_num: int, arity: int) -> List[List[float]]:
    return [[1.0] * arity for _ in range(req_type_num)]


def _materialize_attrs(node: SymbolicNode, node_id: int, attrs: Attrs, *, req_type_num: int, arity: int) -> None:
    hint = node.hint or {}
    if node.op == Parallelism.DP:
        val = copy.deepcopy(hint.get("dp_attr"))
        if val is None or len(val) != req_type_num or any(len(row) != arity for row in val):
            val = _uniform_dp_attr(req_type_num, arity)
        attrs.dp_attr[node_id] = val
    elif node.op == Parallelism.PP:
        val = copy.deepcopy(hint.get("pp_attr"))
        if val is None or len(val) != arity:
            val = [1.0] * arity
        attrs.pp_attr[node_id] = val
    elif node.op == Parallelism.TP:
        val = copy.deepcopy(hint.get("tp_attr"))
        if val is None or len(val) != arity:
            val = [1.0] * arity
        attrs.tp_attr[node_id] = val
    elif node.op == Parallelism.XP:
        val = copy.deepcopy(hint.get("xp_attr"))
        if val is None or len(val) != 2 or set(val) != {XpTag.ATTENTION, XpTag.LINEAR}:
            val = [XpTag.ATTENTION, XpTag.LINEAR]
        attrs.xp_attr[node_id] = val
    else:
        raise ValueError(f"Unexpected op: {node.op}")


def _ensure_materializable(root: SymbolicNode) -> None:
    for node in root.walk():
        if node.op == Parallelism.XP:
            if node.children:
                if len(node.children) != 2:
                    raise ValueError("Internal XP must be materialized as a binary parallel node.")
            else:
                if node.total_devices() != 2:
                    raise ValueError("XP leaf must own exactly 2 devices.")
        if node.children:
            if not node.closed:
                raise ValueError(f"Node {node.pretty()} is not closed.")
        elif node.total_devices() <= 0:
            raise ValueError(f"Leaf {node.pretty()} must own at least one device.")



def is_materializable(root: SymbolicNode) -> bool:
    try:
        _ensure_materializable(root)
        return True
    except Exception:
        return False


def has_open_nodes(root: SymbolicNode) -> bool:
    for node in root.walk():
        if not bool(node.closed):
            return True
        if node.op == Parallelism.XP:
            if node.children:
                if len(node.children) != 2:
                    return True
            elif node.total_devices() != 2:
                return True
    return False


# -----------------------------
# Initialization pattern library
# Fixed, hand-written seed templates driven by device abstraction.
# Every pattern returned here is already CLOSED and directly materializable. XP may be internal-binary or a leaf with exactly 2 devices.
# -----------------------------

def _all_device_counts(device_type_to_ids: Dict[str, List[int]]) -> Dict[str, int]:
    return {
        str(dtype): int(len(ids))
        for dtype, ids in sorted(device_type_to_ids.items())
        if len(ids) > 0
    }


def _split_counts_two_buckets(counts: Dict[str, int]) -> Optional[Tuple[Dict[str, int], Dict[str, int]]]:
    counts = _normalize_counts(counts)
    if sum(counts.values()) < 2:
        return None

    items = [(str(k), int(v)) for k, v in sorted(counts.items()) if int(v) > 0]
    if not items:
        return None

    if len(items) == 1:
        dtype, total = items[0]
        if total < 2:
            return None
        left_n = total // 2
        right_n = total - left_n
        if left_n <= 0 or right_n <= 0:
            return None
        return ({dtype: left_n}, {dtype: right_n})

    total = sum(v for _, v in items)
    target_left = max(1, total // 2)

    left: Dict[str, int] = {}
    right: Dict[str, int] = {k: v for k, v in items}
    left_total = 0

    for dtype, avail in items:
        if left_total >= target_left:
            break
        take = min(avail, target_left - left_total)
        if take > 0:
            left[dtype] = take
            right[dtype] -= take
            left_total += take

    right = _normalize_counts(right)
    if sum(left.values()) == 0 or sum(right.values()) == 0:
        # deterministic fallback: move exactly one full type to the left
        dtype0, avail0 = items[0]
        if len(items) >= 2:
            left = {dtype0: avail0}
            right = _normalize_counts({k: v for k, v in items[1:]})
        else:
            return None

    if sum(left.values()) == 0 or sum(right.values()) == 0:
        return None
    return _normalize_counts(left), _normalize_counts(right)


def _split_counts_for_dp_xp_tail(
    counts: Dict[str, int],
) -> Optional[Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]]:
    counts = _normalize_counts(counts)
    items = [(str(k), int(v)) for k, v in sorted(counts.items()) if int(v) > 0]
    if len(items) < 2:
        return None

    pivot = [(k, v) for k, v in items if v >= 2]
    if len(pivot) < 2:
        return None

    (t0, c0), (t1, c1) = pivot[:2]
    xp_left = {t0: max(1, c0 // 2)}
    xp_right = {t1: max(1, c1 // 2)}

    tail: Dict[str, int] = {k: v for k, v in items}
    tail[t0] -= xp_left[t0]
    tail[t1] -= xp_right[t1]
    tail = _normalize_counts(tail)

    if not tail:
        return None
    return _normalize_counts(xp_left), _normalize_counts(xp_right), tail


def _enumerate_two_way_seed_splits(counts: Dict[str, int]) -> List[Tuple[Dict[str, int], Dict[str, int]]]:
    counts = _normalize_counts(counts)
    items = [(str(k), int(v)) for k, v in sorted(counts.items()) if int(v) > 0]
    out: List[Tuple[Dict[str, int], Dict[str, int]]] = []
    seen: set[Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, int], ...]]] = set()

    def add(left: Dict[str, int], right: Dict[str, int]) -> None:
        l = _normalize_counts(left)
        r = _normalize_counts(right)
        if not l or not r:
            return
        key_lr = (tuple(sorted(l.items())), tuple(sorted(r.items())))
        key_rl = (key_lr[1], key_lr[0])
        key = min(key_lr, key_rl)
        if key in seen:
            return
        seen.add(key)
        out.append((l, r))

    bal = _split_counts_two_buckets(counts)
    if bal is not None:
        add(*bal)

    n = len(items)
    if n >= 2:
        # full-type subset splits
        for mask in range(1, (1 << n) - 1):
            left = {items[i][0]: items[i][1] for i in range(n) if (mask >> i) & 1}
            right = {items[i][0]: items[i][1] for i in range(n) if not ((mask >> i) & 1)}
            add(left, right)
            if len(out) >= 12:
                break

    # single-type partial splits against the remainder
    for dtype, cnt in items:
        if cnt < 2:
            continue
        probe_points = sorted(set([1, cnt // 2, cnt - 1]))
        for take in probe_points:
            if take <= 0 or take >= cnt:
                continue
            left = {dtype: take}
            right = dict(counts)
            right[dtype] -= take
            add(left, right)
            if len(out) >= 18:
                break
        if len(out) >= 18:
            break

    return out


def default_init_patterns(device_type_to_ids: Dict[str, List[int]]) -> List[InitPatternSpec]:
    all_counts = _all_device_counts(device_type_to_ids)
    if not all_counts:
        return []

    patterns: List[InitPatternSpec] = []
    seen_names: set[str] = set()

    def add(name: str, stratum: str, root: Dict[str, Any], weight: float = 1.0) -> None:
        if name in seen_names:
            return
        patterns.append(InitPatternSpec(name=name, stratum=stratum, root=root, weight=weight))
        seen_names.add(name)

    # 1) Single closed leaf seeds. Very stable seeds, good for breadth.
    for op_name in ("DP", "PP", "TP"):
        add(
            f"{op_name.lower()}_single_leaf_all",
            "single_leaf",
            {"op": op_name, "device_counts": all_counts, "closed": True},
            1.0 if op_name == "TP" else 0.9,
        )

    # 2) Split by device type: one TP child per type under DP / PP / TP.
    if len(all_counts) >= 2:
        children = [
            {"op": "TP", "device_counts": {dtype: cnt}, "closed": True}
            for dtype, cnt in sorted(all_counts.items())
        ]
        add("dp_split_by_type", "by_type_split", {"op": "DP", "children": children, "closed": True}, 1.2)
        add("tp_split_by_type", "by_type_split", {"op": "TP", "children": children, "closed": True}, 1.1)
        add("pp_split_by_type", "by_type_split", {"op": "PP", "children": children, "closed": True}, 0.8)

    # 3) Closed two-way partitions across all devices. These are the main source
    #    of robust, diverse initial seeds.
    for idx, (left_counts, right_counts) in enumerate(_enumerate_two_way_seed_splits(all_counts)):
        add(
            f"xp_binary_root_{idx}",
            "xp_root",
            {
                "op": "XP",
                "children": [
                    {"op": "TP", "device_counts": left_counts, "closed": True},
                    {"op": "TP", "device_counts": right_counts, "closed": True},
                ],
                "closed": True,
            },
            1.15,
        )
        add(
            f"dp_binary_split_{idx}",
            "binary_split",
            {
                "op": "DP",
                "children": [
                    {"op": "TP", "device_counts": left_counts, "closed": True},
                    {"op": "TP", "device_counts": right_counts, "closed": True},
                ],
                "closed": True,
            },
            1.05,
        )
        add(
            f"pp_binary_split_{idx}",
            "binary_split",
            {
                "op": "PP",
                "children": [
                    {"op": "TP", "device_counts": left_counts, "closed": True},
                    {"op": "TP", "device_counts": right_counts, "closed": True},
                ],
                "closed": True,
            },
            0.95,
        )

    # 4) XP embedded under DP / PP: closed but still structurally richer than pure binary roots.
    dp_xp_tail = _split_counts_for_dp_xp_tail(all_counts)
    if dp_xp_tail is not None:
        xp_left, xp_right, tail = dp_xp_tail
        add(
            "dp_with_embedded_xp",
            "xp_embedded",
            {
                "op": "DP",
                "children": [
                    {
                        "op": "XP",
                        "children": [
                            {"op": "TP", "device_counts": xp_left, "closed": True},
                            {"op": "TP", "device_counts": xp_right, "closed": True},
                        ],
                        "closed": True,
                    },
                    {"op": "TP", "device_counts": tail, "closed": True},
                ],
                "closed": True,
            },
            1.10,
        )
        add(
            "pp_with_embedded_xp",
            "xp_embedded",
            {
                "op": "PP",
                "children": [
                    {
                        "op": "XP",
                        "children": [
                            {"op": "TP", "device_counts": xp_left, "closed": True},
                            {"op": "TP", "device_counts": xp_right, "closed": True},
                        ],
                        "closed": True,
                    },
                    {"op": "TP", "device_counts": tail, "closed": True},
                ],
                "closed": True,
            },
            0.95,
        )

    return patterns



def default_numeric_patterns() -> List[NumericPatternSpec]:
    return [
        NumericPatternSpec(
            name="xp_any_binary",
            match={"op": Parallelism.XP, "arity": 2},
            candidates=[
                NumericAttrCandidate(attrs={"xp_attr": [XpTag.ATTENTION, XpTag.LINEAR]}, weight=1.0),
                NumericAttrCandidate(attrs={"xp_attr": [XpTag.LINEAR, XpTag.ATTENTION]}, weight=1.0),
            ],
            weight=1.0,
        ),
        NumericPatternSpec(
            name="pp_binary_bias",
            match={"op": Parallelism.PP, "arity": 2},
            candidates=[
                NumericAttrCandidate(attrs={"pp_attr": [1.0, 1.0]}, weight=0.8),
                NumericAttrCandidate(attrs={"pp_attr": [1.5, 0.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"pp_attr": [0.5, 1.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"pp_attr": [2.0, 1.0]}, weight=0.7),
                NumericAttrCandidate(attrs={"pp_attr": [1.0, 2.0]}, weight=0.7),
            ],
            weight=0.9,
        ),
        NumericPatternSpec(
            name="pp_ternary_bias",
            match={"op": Parallelism.PP, "arity": 3},
            candidates=[
                NumericAttrCandidate(attrs={"pp_attr": [1.0, 1.0, 1.0]}, weight=0.8),
                NumericAttrCandidate(attrs={"pp_attr": [1.5, 0.5, 0.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"pp_attr": [0.5, 1.5, 0.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"pp_attr": [0.5, 0.5, 1.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"pp_attr": [1.5, 1.0, 0.5]}, weight=0.7),
                NumericAttrCandidate(attrs={"pp_attr": [0.5, 1.0, 1.5]}, weight=0.7),
            ],
            weight=0.8,
        ),
        NumericPatternSpec(
            name="tp_binary_bias",
            match={"op": Parallelism.TP, "arity": 2},
            candidates=[
                NumericAttrCandidate(attrs={"tp_attr": [1.0, 1.0]}, weight=0.8),
                NumericAttrCandidate(attrs={"tp_attr": [1.5, 0.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"tp_attr": [0.5, 1.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"tp_attr": [2.0, 1.0]}, weight=0.7),
                NumericAttrCandidate(attrs={"tp_attr": [1.0, 2.0]}, weight=0.7),
            ],
            weight=1.0,
        ),
        NumericPatternSpec(
            name="tp_ternary_bias",
            match={"op": Parallelism.TP, "arity": 3},
            candidates=[
                NumericAttrCandidate(attrs={"tp_attr": [1.0, 1.0, 1.0]}, weight=0.8),
                NumericAttrCandidate(attrs={"tp_attr": [1.5, 0.5, 0.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"tp_attr": [0.5, 1.5, 0.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"tp_attr": [0.5, 0.5, 1.5]}, weight=1.0),
                NumericAttrCandidate(attrs={"tp_attr": [1.5, 1.0, 0.5]}, weight=0.7),
                NumericAttrCandidate(attrs={"tp_attr": [0.5, 1.0, 1.5]}, weight=0.7),
            ],
            weight=0.9,
        ),
        NumericPatternSpec(
            name="dp_binary_bias",
            match={"op": Parallelism.DP, "arity": 2},
            candidates=[
                # [dp_attr] 一共有 request_type_num 个 list，每个 list 是该 req_type 在各个 child 上的切分比例
                NumericAttrCandidate(attrs={"dp_attr": [[1.0, 0.5], [0.5, 0.0], [0.0, 1.0]]},weight=1.5),
                NumericAttrCandidate(attrs={"dp_attr": [[0.1, 0.5], [0.5, 1.0], [0.5, 1.0]]}, weight=1.5),
                NumericAttrCandidate(attrs={"dp_attr": [[1.0, 1.5], [0.5, 0.5], [0.5, 1.5]]}, weight=1.5),
            ],
            weight=0.9,
        ),
        NumericPatternSpec(
            name="dp_ternary_bias",
            match={"op": Parallelism.DP, "arity": 3},
            candidates=[
                # [dp_attr] 一共有 request_type_num 个 list，每个 list 是该 req_type 在各个 child 上的切分比例
                NumericAttrCandidate(attrs={"dp_attr": [[1.0, 0.5, 0.0], [0.5, 0.0, 1.0], [0.0, 0.0, 1.0]]}, weight=1.5),
            ],
            weight=0.9,
        ),
    ]



def default_patterns() -> List[PatternSpec]:
    xp_hint = {"xp_attr": [XpTag.LINEAR, XpTag.ATTENTION]}

    return [
        # ------------------------------------------------------------------
        # Skeleton expansion rules that DIRECTLY generate closed XP leaves.
        # This matches the current semantics:
        #   - internal XP => exactly 2 parallel children
        #   - leaf XP     => exactly 2 devices
        # ------------------------------------------------------------------
        PatternSpec(
            name="dp_leaf_npu4_pim4_to_dp_xp_tp_tp",
            family=RewriteFamily.SKELETON_EXPANSION,
            match={
                "op": Parallelism.DP,
                "leaf": True,
                "closed": True,
                "device_counts": {"NPU": 4, "PIM": 4},
            },
            rewrite={
                "op": "DP",
                "children": [
                    {"op": "XP", "device_counts": {"NPU": 1, "PIM": 1}, "hint": xp_hint, "closed": True},
                    {"op": "TP", "device_counts": {"NPU": 3}, "closed": True},
                    {"op": "TP", "device_counts": {"PIM": 3}, "closed": True},
                ],
                # [dp_attr] 一共有 request_type_num 个 list，每个 list 是该 req_type 在各个 child 上的切分比例
                "hint": {"dp_attr": [[1.0, 0.5, 0.0], [0.5, 0.0, 1.0], [0.0, 1.0, 0.5]]},
                "closed": True,
            },
            weight=1.20,
        ),
        PatternSpec(
            name="tp_leaf_npu4_pim4_to_dp_xp_tp_tp",
            family=RewriteFamily.SKELETON_EXPANSION,
            match={
                "op": Parallelism.TP,
                "leaf": True,
                "closed": True,
                "device_counts": {"NPU": 4, "PIM": 4},
            },
            rewrite={
                "op": "DP",
                "children": [
                    {"op": "XP", "device_counts": {"NPU": 2, "PIM": 2}, "hint": xp_hint, "closed": False},
                    {"op": "TP", "device_counts": {"NPU": 2}, "closed": True},
                    {"op": "TP", "device_counts": {"PIM": 2}, "closed": True},
                ],
                # [dp_attr] 一共有 request_type_num 个 list，每个 list 是该 req_type 在各个 child 上的切分比例
                # "hint": {"dp_attr": [[1.0, 0.5, 0.0], [0.5, 0.0, 1.0], [0.0, 1.0, 0.5]]},
                "hint": {"dp_attr": [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]},
                "closed": True,
            },
            weight=1.20,
        ),
        # ------------------------------------------------------------------
        # Repartition / rollback around the new XP-leaf shape.
        # These rules let the search move between:
        #   XP leaf (2 devices)  <->  internal XP with 2 TP children
        # ------------------------------------------------------------------
        PatternSpec(
            name="xp_leaf_npu2_pim2_to_xp_tp_tp",
            family=RewriteFamily.LOCAL_REFINEMENT,
            match={
                "op": Parallelism.XP,
                "leaf": True,
                "closed": False,
                "device_counts": {"NPU": 2, "PIM": 2},
            },
            rewrite={
                "op": "XP",
                "children": [
                    {"op": "TP", "device_counts": {"NPU": 2}, "closed": True},
                    {"op": "TP", "device_counts": {"PIM": 2}, "closed": True},
                ],
                "hint": xp_hint,
                "closed": True,
            },
            weight=0.90,
        ),
        PatternSpec(
            name="xp_leaf_npu1_pim1_to_xp_tp_tp",
            family=RewriteFamily.LOCAL_REFINEMENT,
            match={
                "op": Parallelism.XP,
                "leaf": True,
                "closed": True,
                "device_counts": {"NPU": 1, "PIM": 1},
            },
            rewrite={
                "op": "XP",
                "children": [
                    {"op": "TP", "device_counts": {"NPU": 1}, "closed": True},
                    {"op": "TP", "device_counts": {"PIM": 1}, "closed": True},
                ],
                "hint": xp_hint,
                "closed": True,
            },
            weight=0.90,
        ),
        PatternSpec(
            name="xp_leaf_npu2_to_xp_tp_tp",
            family=RewriteFamily.LOCAL_REFINEMENT,
            match={
                "op": Parallelism.XP,
                "leaf": True,
                "closed": True,
                "device_counts": {"NPU": 2},
            },
            rewrite={
                "op": "XP",
                "children": [
                    {"op": "TP", "device_counts": {"NPU": 1}, "closed": True},
                    {"op": "TP", "device_counts": {"NPU": 1}, "closed": True},
                ],
                "hint": xp_hint,
                "closed": True,
            },
            weight=0.85,
        ),
        PatternSpec(
            name="xp_leaf_pim2_to_xp_tp_tp",
            family=RewriteFamily.LOCAL_REFINEMENT,
            match={
                "op": Parallelism.XP,
                "leaf": True,
                "closed": True,
                "device_counts": {"PIM": 2},
            },
            rewrite={
                "op": "XP",
                "children": [
                    {"op": "TP", "device_counts": {"PIM": 1}, "closed": True},
                    {"op": "TP", "device_counts": {"PIM": 1}, "closed": True},
                ],
                "hint": xp_hint,
                "closed": True,
            },
            weight=0.85,
        )
    ]
