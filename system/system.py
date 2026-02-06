from copy import deepcopy
from ctypes.wintypes import DOUBLE

from system.config import SystemConfig, ModelConfig
from serving.simulator import Simulator
from parallelism.ptree import ParallelismTree
from hardware.htree import HardwareTree

class System:
    def __init__(self, sys_cfg: SystemConfig, model_cfg: ModelConfig):
        self.sys_cfg = sys_cfg
        self.model_cfg = model_cfg
        # 0 是 baseline；1 是测试例子；2 是简化例子
        hcase_idx = 2
        self.htree = HardwareTree(hcase_idx)
        # 0 1 2 测试例子，复杂但性能差；3 是 baseline；4 已超越 baseline；5 是简化例子
        pcase_idx = 9
        self.ptree = ParallelismTree(sys_cfg, model_cfg, self.htree, case_idx=pcase_idx)
        self.req_prob = [0.9, 0.1]

    def run_system(self):
        # 每个 begin_nodes 都对应一个 simulator
        original_lambda = self.sys_cfg.lam
        simulation_result = []
        for begin_node in self.ptree.begin_nodes:
            sim_cfg = deepcopy(self.sys_cfg)
            # 指定 PP sub batch num
            sub_batch_num = self.ptree.summarise_layer_info(begin_node)
            sim_cfg.sub_batch_num = sub_batch_num
            # sim_cfg.use_pp_sub_batch = False
            sim_cfg.use_pp_sub_batch = True
            # 指定是否采用 MP sub batch
            # sim_cfg.use_mp_sub_batch = False
            sim_cfg.use_mp_sub_batch = True
            # 分配 lambda (req rate)
            this_lambda = sum([(dp_attr[1]-dp_attr[0])*prob*original_lambda for dp_attr, prob in zip(begin_node.dp_attr, self.req_prob)])
            sim_cfg.lam = this_lambda
            # 启动 Simulator
            simulator = Simulator(sim_cfg, self.model_cfg, self.req_prob, self.ptree)
            single_thread_result = simulator.run(begin_node)
            simulation_result.append(single_thread_result)
        return simulation_result