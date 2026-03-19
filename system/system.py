from copy import deepcopy
import numpy as np

from system.config import SystemConfig, ModelConfig
from serving.simulator import Simulator
from parallelism.ptree import ParallelismTree
from hardware.htree import HardwareTree
from system.metrics import summarize_metrics, summarize_metrics_data

class System:
    def __init__(self, sys_cfg: SystemConfig, model_cfg: ModelConfig):
        self.sys_cfg = sys_cfg
        self.model_cfg = model_cfg

        hcase_idx = sys_cfg.hcase_index
        self.htree = HardwareTree(hcase_idx)

        pcase_idx = sys_cfg.pcase_index
        self.ptree = ParallelismTree(sys_cfg, model_cfg, self.htree, case_idx=pcase_idx)

        # self.req_prob = [0.0, 1.0, 0.0]
        self.req_prob = sys_cfg.req_dist

    def run_system(self):
        # 每个 begin_nodes 都对应一个 simulator
        original_lambda = self.sys_cfg.lam
        simulation_result = []
        for begin_node_idx, begin_node in enumerate(self.ptree.begin_nodes):
            print("begin_node_name:", begin_node.name)
            print("begin_node_dp_attr:", begin_node.dp_attr)
            this_sys_cfg = deepcopy(self.sys_cfg)
            # 指定 PP sub batch num
            sub_batch_num = self.ptree.summarise_layer_info(begin_node)
            this_sys_cfg.sub_batch_num = sub_batch_num
            # this_sys_cfg.use_pp_sub_batch = False
            this_sys_cfg.use_pp_sub_batch = True
            # 指定是否采用 MP sub batch
            # this_sys_cfg.use_mp_sub_batch = False
            this_sys_cfg.use_mp_sub_batch = True
            # 分配 lambda (req rate)
            this_lambda = sum([(dp_attr[1]-dp_attr[0])*prob*original_lambda for dp_attr, prob in zip(begin_node.dp_attr, self.req_prob)])
            if this_lambda <= 0:
                continue
            this_sys_cfg.lam = this_lambda
            print("this_lambda:", this_lambda)

            # TODO: 这里是非常关键的一个优化点，探究不同 sub graph 的 batch size
            # this_sys_cfg.max_batch_lo = min(this_sys_cfg.max_batch_lo, this_lambda * 10)
            # print("this_max_batch=",this_sys_cfg.max_batch_lo)

            # 启动 Simulator
            sample_prob = np.array([(dp_attr[1]-dp_attr[0])*prob for dp_attr, prob in zip(begin_node.dp_attr, self.req_prob)])
            sample_prob = sample_prob/np.sum(sample_prob)
            # print("sample_prob:", sample_prob)
            simulator = Simulator(this_sys_cfg, self.model_cfg, sample_prob, self.ptree)
            single_thread_result = simulator.run(begin_node)
            simulation_result.append(single_thread_result)

            # print("sub graph:", begin_node_idx)
            # print(summarize_metrics([single_thread_result], self.sys_cfg.t_end))
            # print("\n")

        return simulation_result