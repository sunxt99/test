from math import floor, ceil
import numpy as np

from parallelism.case import *
from parallelism.ptraversal import *
from parallelism.pperf import peer_to_peer_communication_pattern, all_reduce_communication_pattern

from serving.request import Request
from serving.config import SimulatorConfig, ModelConfig

from hardware.htree import HardwareTree
from hardware.hperf import peer_to_peer_communication_time_cost, all_reduce_communication_time_cost

class ParallelismTree:
    def __init__(self,
                 sim_cfg: SimulatorConfig,
                 model_cfg: ModelConfig,
                 htree: HardwareTree,
                 case_idx: int =2):
        self.root_node = None
        self.begin_nodes = None
        self.leaf_nodes = None
        self.mapper = None
        self.sim_cfg = sim_cfg
        self.model_cfg = model_cfg
        self.htree = htree
        self.build_tree_by_case(case_idx)

    def build_tree_by_case(self, case_idx: int):
        if case_idx == 0:
            self.root_node, self.leaf_nodes = build_case_0(self.sim_cfg.req_type_num, self.model_cfg.layer_num)
        elif case_idx == 1:
            self.root_node, self.leaf_nodes = build_case_1(self.sim_cfg.req_type_num, self.model_cfg.layer_num)
        elif case_idx == 2:
            self.root_node, self.leaf_nodes = build_case_2(self.sim_cfg.req_type_num, self.model_cfg.layer_num)
        elif case_idx == 3:
            self.root_node, self.leaf_nodes = build_case_3(self.sim_cfg.req_type_num, self.model_cfg.layer_num)
        elif case_idx == 4:
            self.root_node, self.leaf_nodes = build_case_4(self.sim_cfg.req_type_num, self.model_cfg.layer_num)
        else:
            raise NotImplementedError

        self.begin_nodes = [i for i in detect_begin_nodes(self.root_node)]
        self.hardware_mapping()

        # for begin_node in self.begin_nodes:
        #     leaf_nodes = derive_from_node(begin_node)
            # for leaf_node in leaf_nodes:
            #     leaf_node.print_info()

    def run(self, active_queue = None):
        req_list_by_type = [[] for _ in range(self.sim_cfg.req_type_num)]
        req_type_distribution = [0] * self.sim_cfg.req_type_num
        if active_queue is not None:
            for r in active_queue:
                req_list_by_type[r.req_type].append(r)
                req_type_distribution[r.req_type] += 1
            # print('req_type_distribution:',req_type_distribution)

        max_run_time_cost_ms = 0.0
        min_run_time_cost_ms = 10000000.0
        for begin_node in self.begin_nodes:
            leaf_nodes = derive_from_node(begin_node)
            sub_graph_time_cost_ms = self.analyse_sub_graph(begin_node, leaf_nodes, req_list_by_type, req_type_distribution)
            max_run_time_cost_ms = max(max_run_time_cost_ms, sub_graph_time_cost_ms)
            min_run_time_cost_ms = min(min_run_time_cost_ms, sub_graph_time_cost_ms)
        return max_run_time_cost_ms
        # return min_run_time_cost_ms

    def analyse_sub_graph(self,
                          root_of_subtree: BasicParallelismNode,
                          leaf_nodes_of_subtree: List[BasicHardwareNode],
                          req_list_by_type: List[List[Request]],
                          req_type_distribution: List[int]) -> float:
        for leaf in leaf_nodes_of_subtree:
            assert leaf.is_leaf()
        previous_active_leaf_indexes = []
        processing_time_cost_ms = 0.0
        result_cache = [dict() for _ in range(4)]
        for layer_idx in range(self.model_cfg.layer_num):
            for module_idx in range(4):  # QKV / ATTN / PROJ / FFN
                if layer_idx == 0 and module_idx == 0:
                    continue
                active_leaf_indexes = trigger_leaf_node(layer_idx, module_idx, leaf_nodes_of_subtree)

                if frozenset(active_leaf_indexes) not in result_cache[module_idx]:
                    computation_time_cost_ms = self.analyse_computation_pattern(root_of_subtree,
                                                                             leaf_nodes_of_subtree,
                                                                             module_idx,
                                                                             active_leaf_indexes,
                                                                             req_list_by_type, req_type_distribution)
                    communication_time_cost_ms = self.analyse_communication_pattern(root_of_subtree,
                                                                                   leaf_nodes_of_subtree,
                                                                                   layer_idx,
                                                                                   module_idx,
                                                                                   previous_active_leaf_indexes,
                                                                                   active_leaf_indexes,
                                                                                   req_type_distribution)
                    # layer_time_cost = max(computation_time_cost_ms, communication_time_cost_ms)
                    layer_time_cost = computation_time_cost_ms + communication_time_cost_ms
                    result_cache[module_idx][frozenset(active_leaf_indexes)] = layer_time_cost
                else:
                    layer_time_cost = result_cache[module_idx][frozenset(active_leaf_indexes)]
                processing_time_cost_ms += layer_time_cost
                previous_active_leaf_indexes = active_leaf_indexes

        return processing_time_cost_ms


    def analyse_computation_pattern(self,
                                    root_of_subtree: BasicParallelismNode,
                                    leaf_of_subtree: List[BasicHardwareNode],
                                    module_idx: int,
                                    active_leaf_indexes: List[int],
                                    req_list_by_type: List[List[Request]],
                                    req_type_distribution: List[int],
                                    print_detail = False):
        max_computation_time_ms = 0.0

        if module_idx == 0:
            if print_detail:
                print("------ Compute QKV")
            for active_leaf_idx in active_leaf_indexes:
                active_leaf = leaf_of_subtree[active_leaf_idx]
                # [M,K] * [K,N] = [M,N]
                M_size = sum([(d[1]-d[0])*n for d,n in zip(active_leaf.dp_attr, req_type_distribution)]) # batch size
                K_size = self.model_cfg.hidden_size # input dimension
                N_size = (active_leaf.tp_attr[1] - active_leaf.tp_attr[0]) * (self.model_cfg.hidden_size + 2 * self.model_cfg.kv_hidden_size) # output dimension
                if M_size > 0 and K_size > 0 and N_size > 0:
                    max_computation_time_ms = max(max_computation_time_ms,
                                            self.mapper[active_leaf.name].compute_gemm_time_cost(M_size, N_size, K_size))
        elif module_idx == 1:
            if print_detail:
                print("------ Compute ATTN")
            for active_leaf_idx in active_leaf_indexes:
                active_leaf = leaf_of_subtree[active_leaf_idx]
                query_head_num = ceil(self.model_cfg.query_head_num * (active_leaf.tp_attr[1] - active_leaf.tp_attr[0]))
                kv_head_num = ceil(self.model_cfg.kv_head_num * (active_leaf.tp_attr[1] - active_leaf.tp_attr[0]))
                computation_time_ms = 0.0
                # gqa_ratio = self.model_cfg.query_head_num // self.model_cfg.kv_head_num
                mem_ops = 0
                comp_ops = 0
                for d,n,r in zip(active_leaf.dp_attr, req_type_distribution, req_list_by_type):
                    start_req_idx = floor(d[0] * n)
                    end_req_idx = ceil(d[1] * n)
                    for req_idx in range(start_req_idx, end_req_idx):
                        sequence_length = r[req_idx].prompt_tokens + r[req_idx].gen_tokens
                        # Q * K^T = S
                        mem_ops += (query_head_num * self.model_cfg.head_dim +
                            kv_head_num * self.model_cfg.head_dim * sequence_length +
                            query_head_num * sequence_length) * 2 # 考虑 float16 类型
                        comp_ops += query_head_num * self.model_cfg.head_dim * sequence_length * 2
                        # S * V = C
                        mem_ops += (query_head_num * self.model_cfg.head_dim +
                            kv_head_num * self.model_cfg.head_dim * sequence_length +
                            query_head_num * sequence_length) * 2 # 考虑 float16 类型
                        comp_ops += query_head_num * self.model_cfg.head_dim * sequence_length * 2

                        # Q * K = S
                        # computation_time_ms += self.mapper[active_leaf.name].compute_gemm_time_cost(gqa_ratio,
                        #                                                                        sequence_length,
                        #                                                                        self.model_cfg.head_dim)
                        # # S * V = C
                        # computation_time_ms += self.mapper[active_leaf.name].compute_gemm_time_cost(gqa_ratio,
                        #                                                                        self.model_cfg.head_dim,
                        #                                                                        sequence_length)
                    # print(req_type_distribution)
                    # print(comp_ops, mem_ops)
                if mem_ops > 0:
                    computation_time_ms += self.mapper[active_leaf.name].compute_gemm_time_cost_by_ops(comp_ops, mem_ops)

                max_computation_time_ms = max(max_computation_time_ms, computation_time_ms)

        elif module_idx == 2:
            if print_detail:
                print("------ Compute PROJ")
            for active_leaf_idx in active_leaf_indexes:
                active_leaf = leaf_of_subtree[active_leaf_idx]
                # [M,K] * [K,N] = [M,N]
                M_size = sum([(d[1]-d[0])*n for d,n in zip(active_leaf.dp_attr, req_type_distribution)])
                K_size =  (active_leaf.tp_attr[1] - active_leaf.tp_attr[0]) * self.model_cfg.hidden_size
                N_size = self.model_cfg.hidden_size
                if M_size > 0 and K_size > 0 and N_size > 0:
                    max_computation_time_ms = max(max_computation_time_ms,
                                                  self.mapper[active_leaf.name].compute_gemm_time_cost(M_size, N_size, K_size))

        else:
            if print_detail:
                print("------ Compute FFN")
            for active_leaf_idx in active_leaf_indexes:
                active_leaf = leaf_of_subtree[active_leaf_idx]
                # 1.[M,K] * [K,N] = [M,N]
                # 2.[M,N] * [N,P] = [M,P]
                M_size = sum([(d[1]-d[0])*n for d,n in zip(active_leaf.dp_attr, req_type_distribution)])
                K_size = self.model_cfg.hidden_size
                N_size = (active_leaf.tp_attr[1] - active_leaf.tp_attr[0]) * self.model_cfg.intermediate_size
                P_size = self.model_cfg.hidden_size
                if M_size > 0 and K_size > 0 and N_size > 0 and P_size > 0:
                    max_computation_time_ms = max(max_computation_time_ms,
                                              self.mapper[active_leaf.name].compute_gemm_time_cost(M_size, N_size, K_size) + \
                                              self.mapper[active_leaf.name].compute_gemm_time_cost(M_size, P_size, N_size))

        return max_computation_time_ms

    def analyse_communication_pattern(self,
                                      root_of_subtree: BasicParallelismNode,
                                      leaf_of_subtree: List[BasicHardwareNode],
                                      layer_idx: int, module_idx: int,
                                      src_leaf_indexes: List[int],
                                      dst_leaf_indexes: List[int],
                                      req_type_distribution: List[int],
                                      print_detail = False):
        max_comm_time_ms = 0.0

        # if layer_idx == 0 and module_idx == 0:
        #     if print_detail:
        #         print("------ Broadcast to QKV")

        if module_idx == 0:  # FFN -> QKV
            if print_detail:
                print("------ FFN -> QKV")
            # 可能启动的通信有 D、P、T
            # 先进行发送端的 AllReduce，然后再进行其他操作。总体上就是后面两个操作的联合体。
            # 1. 如果 src_leaf_indexes == dst_leaf_indexes，则没有启动 P 节点，只可能有 D 和 T。
            # 2. 如果 src_leaf_indexes != dst_leaf_indexes，则一定启动了 P 节点，可能有 D 和 T。
            all_reduce_groups, all_reduce_comm_pattern = all_reduce_communication_pattern(leaf_of_subtree, src_leaf_indexes,
                                                         req_type_distribution, self.model_cfg.hidden_size)
            # all reduce 开销
            max_all_reduce_comm_time_ms = 0.0
            for all_reduce_tuple in all_reduce_comm_pattern: # 每个 all reduce group 具有一个 tuple
                all_reduce_leaf_name_list, all_reduce_data_size = all_reduce_tuple
                all_reduce_hardware_list = [self.mapper[name] for name in all_reduce_leaf_name_list]
                all_reduce_comm_time_ms = all_reduce_communication_time_cost(all_reduce_hardware_list, all_reduce_data_size)
                max_all_reduce_comm_time_ms += all_reduce_comm_time_ms

            # peer to peer 开销
            if src_leaf_indexes != dst_leaf_indexes:
                # print_src_and_dst_info(leaf_nodes, src_leaf_indexes, dst_leaf_indexes)
                for all_reduce_group in all_reduce_groups.values():
                    comm_time_ms = 0.0
                    assert all_reduce_group is not None
                    tmp_src_node_idx = all_reduce_group[0]
                    tmp_src_node = copy.deepcopy(leaf_of_subtree[tmp_src_node_idx])
                    tmp_src_node.tp_attr = [0, 1.0]

                    for dst in dst_leaf_indexes:
                        dst_node = leaf_of_subtree[dst]
                        comm_size_byte = 2 * peer_to_peer_communication_pattern(tmp_src_node,
                                                               dst_node,
                                                               req_type_distribution,
                                                               self.model_cfg.hidden_size)
                        comm_time_ms += peer_to_peer_communication_time_cost(self.mapper[tmp_src_node.name],
                                                                            self.mapper[dst_node.name],
                                                                            comm_size_byte)
                    max_comm_time_ms = max(max_comm_time_ms, comm_time_ms)

            max_comm_time_ms += max_all_reduce_comm_time_ms

        elif module_idx == 1 or module_idx == 2:  # QKV -> ATTN and ATTN -> PROJ
            if print_detail:
                if module_idx == 1:
                    print("------ QKV -> ATTN")
                elif module_idx == 2:
                    print("------ ATTN -> PROJ")

            if src_leaf_indexes == dst_leaf_indexes:
                # 未启动 X，则 src == dst；反之也成立，若 src == dst，则未启动 X。
                pass
            else:
                # 启动 X，则 src != dst，需要进行通信，将 src 节点的数据传递到 dst 节点
                # X 节点将 Graph 划分为 Fan-In 路径和 Fan-Out 路径，所以整个通信路径是 Fan-In -> X -> Fan-Out
                # Fan-In 和 Fan-Out 路径上可能会有D、P、T。根据规定，Fan-In 和 Fan-Out 的路径都遵循 D-P-T 的顺序。
                #   Fan-In 和 Fan-Out 路径上的 P 不发挥作用，可不用分析。
                #   因此通信模式可以直接通过 tp_attr 和 dp_attr 来分析得到。

                # print_src_and_dst_info(leaf_nodes, src_leaf_indexes, dst_leaf_indexes)
                for src_idx in src_leaf_indexes:
                    comm_time_ms = 0.0
                    for dst_idx in dst_leaf_indexes:
                        src_node = leaf_of_subtree[src_idx]
                        dst_node = leaf_of_subtree[dst_idx]
                        comm_size_byte = 2 * peer_to_peer_communication_pattern(src_node,
                                                                                  dst_node,
                                                                                  req_type_distribution,
                                                                                  self.model_cfg.hidden_size)
                        comm_time_ms += peer_to_peer_communication_time_cost(self.mapper[src_node.name],
                                                                            self.mapper[dst_node.name],
                                                                            comm_size_byte)
                    max_comm_time_ms = max(comm_time_ms, max_comm_time_ms)

        elif module_idx == 3:  # PROJ -> FFN
            if print_detail:
                print("------ PROJ -> FFN")
            if src_leaf_indexes == dst_leaf_indexes:  # 启动 AllReduce
                # Active Graph 中肯定有 T，可能有 D，不可能有 P（有 P 的话不会再这个阶段进行通信）。
                # 总体思路就是向上回溯到 level 最高的 TP-node。具有同一个最高 level TP-node 的节点们会进行 All-Reduce。
                _, all_reduce_comm_pattern = all_reduce_communication_pattern(leaf_of_subtree, src_leaf_indexes,
                                                             req_type_distribution, self.model_cfg.hidden_size)
                # all reduce 开销（同前面）
                max_all_reduce_comm_time_ms = 0.0
                for all_reduce_tuple in all_reduce_comm_pattern:
                    all_reduce_leaf_name_list, all_reduce_data_size = all_reduce_tuple
                    all_reduce_hardware_list = [self.mapper[name] for name in all_reduce_leaf_name_list]
                    all_reduce_comm_time_ms = all_reduce_communication_time_cost(all_reduce_hardware_list,
                                                                                 all_reduce_data_size)
                    max_all_reduce_comm_time_ms += all_reduce_comm_time_ms
                    # max_all_reduce_comm_time_ms = max(max_all_reduce_comm_time_ms, all_reduce_comm_time_ms)
                max_comm_time_ms = max(max_comm_time_ms, max_all_reduce_comm_time_ms)
            else:
                raise ValueError
        return max_comm_time_ms

    def hardware_mapping(self):
        # 将 parallel tree 中的每个 leaf 对应到 hardware 中的 leaf
        # self.mapper's key 是 parallel leaf's name
        # self.mapper's value 就是 device
        print("\n---------hardware mapping---------")
        self.mapper = dict()
        for leaf, device in zip(self.leaf_nodes, self.htree.devices):
            print(leaf.name, device.name)
            self.mapper[leaf.name] = device
        for leaf,device in self.mapper.items():
            print("{} : {}".format(leaf, device))
        print("----------------------------------\n")