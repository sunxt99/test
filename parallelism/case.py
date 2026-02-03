from parallelism.pnode import *

def build_case_0(request_type_num,
                 total_layer_num):
    # ------------------ CASE 0 --------------------------
    node_0 = DataParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 parallel_attr=[[0.5,0.5],[0.5,0.5]])
    node_1 = TensorParallelismNode(name='p1', parallel_attr=[0.5, 0.5])
    node_2 = ModuleParallelismNode(name='p2', parallel_attr=[XpTag.ATTENTION, XpTag.LINEAR])
    node_3 = TensorParallelismNode(name='p3', parallel_attr=[0.5, 0.5])
    node_4 = ModuleParallelismNode(name='p4', parallel_attr=[XpTag.ATTENTION, XpTag.LINEAR])
    node_5 = TensorParallelismNode(name='p5', parallel_attr=[0.5, 0.5])
    node_6 = TensorParallelismNode(name='p6', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_1.add_child(node_2)
    node_1.add_child(node_3)
    node_0.add_child(node_4)
    node_4.add_child(node_5)
    node_4.add_child(node_6)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')

    node_2.add_child(leaf_0)
    node_2.add_child(leaf_1)
    node_3.add_child(leaf_2)
    node_3.add_child(leaf_3)
    node_5.add_child(leaf_4)
    node_5.add_child(leaf_5)
    node_6.add_child(leaf_6)
    node_6.add_child(leaf_7)

    return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]

def build_case_1(request_type_num,
                 total_layer_num):
    # ------------------ CASE 1 --------------------------
    node_0 = ModuleParallelismNode(name='p0',
                                   dp_attr=[[0.0, 1.0]] * request_type_num,
                                   pp_attr=[0,total_layer_num-1],
                                   tp_attr=[0.0, 1.0],
                                   xp_attr=XpTag.BOTH,
                                   parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_1 = PipelineParallelismNode(name='p1', parallel_attr=[0.4, 0.6])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.5, 0.5])
    node_3 = TensorParallelismNode(name='p3', parallel_attr=[0.5, 0.5])
    node_4 = TensorParallelismNode(name='p4', parallel_attr=[0.5, 0.5])
    node_5 = TensorParallelismNode(name='p5', parallel_attr=[0.5, 0.5])
    node_6 = TensorParallelismNode(name='p6', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_1.add_child(node_2)
    node_1.add_child(node_3)
    node_0.add_child(node_4)
    node_4.add_child(node_5)
    node_4.add_child(node_6)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')

    node_2.add_child(leaf_0)
    node_2.add_child(leaf_1)
    node_3.add_child(leaf_2)
    node_3.add_child(leaf_3)
    node_5.add_child(leaf_4)
    node_5.add_child(leaf_5)
    node_6.add_child(leaf_6)
    node_6.add_child(leaf_7)

    return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]

def build_case_2(request_type_num,
                 total_layer_num):
    # ------------------ CASE 2 --------------------------
    node_0 = ModuleParallelismNode(name='p0',
                                   dp_attr=[[0.0, 1.0]] * request_type_num,
                                   pp_attr=[0,total_layer_num-1],
                                   tp_attr=[0.0, 1.0],
                                   xp_attr=XpTag.BOTH,
                                   parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_1 = PipelineParallelismNode(name='p1', parallel_attr=[0.2, 0.8])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.2, 0.8])
    node_3 = TensorParallelismNode(name='p3', parallel_attr=[0.6, 0.4])
    # 对于 DP，一共有 request_type_num 个 list，每个 list 代表了一种 req 的划分比例
    node_4 = DataParallelismNode(name='p4', parallel_attr=[[0.7,0.3], [0.5,0.5]])
    node_5 = PipelineParallelismNode(name='p5', parallel_attr=[0.5, 0.5])
    node_6 = TensorParallelismNode(name='p6', parallel_attr=[0.8, 0.2])

    node_0.add_child(node_1)
    node_1.add_child(node_2)
    node_1.add_child(node_3)
    node_0.add_child(node_4)
    node_4.add_child(node_5)
    node_4.add_child(node_6)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')

    node_2.add_child(leaf_0)
    node_2.add_child(leaf_1)
    node_3.add_child(leaf_2)
    node_3.add_child(leaf_3)
    node_5.add_child(leaf_4)
    node_5.add_child(leaf_5)
    node_6.add_child(leaf_6)
    node_6.add_child(leaf_7)

    return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]


def build_case_3(request_type_num,
                 total_layer_num):
    # ------------------ CASE 3 --------------------------
    node_0 = ModuleParallelismNode(name='p0',
                                   dp_attr=[[0.0, 1.0]] * request_type_num,
                                   pp_attr=[0,total_layer_num-1],
                                   tp_attr=[0.0, 1.0],
                                   xp_attr=XpTag.BOTH,
                                   parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_1 = TensorParallelismNode(name='p1', parallel_attr=[0.25, 0.25, 0.25, 0.25])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.25, 0.25, 0.25, 0.25])

    node_0.add_child(node_1)
    node_0.add_child(node_2)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')

    node_1.add_child(leaf_0)
    node_1.add_child(leaf_1)
    node_1.add_child(leaf_2)
    node_1.add_child(leaf_3)
    node_2.add_child(leaf_4)
    node_2.add_child(leaf_5)
    node_2.add_child(leaf_6)
    node_2.add_child(leaf_7)

    # return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]
    return node_0, [leaf_0, leaf_4, leaf_1, leaf_5, leaf_2, leaf_6, leaf_3, leaf_7]


def build_case_4(request_type_num,
                 total_layer_num):
    # ------------------ CASE 4 --------------------------
    node_0 = DataParallelismNode(name='p0',
                                 dp_attr=[[0.0, 1.0]] * request_type_num,
                                 pp_attr=[0, total_layer_num - 1],
                                 tp_attr=[0.0, 1.0],
                                 xp_attr=XpTag.BOTH,
                                 parallel_attr=[[0.5,0.5], [0.5,0.5]])
    node_1 = ModuleParallelismNode(name='p1', parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    node_2 = TensorParallelismNode(name='p2', parallel_attr=[0.5, 0.5])
    node_3 = TensorParallelismNode(name='p3', parallel_attr=[0.5, 0.5])
    node_4 = ModuleParallelismNode(name='p4', parallel_attr=[XpTag.LINEAR, XpTag.ATTENTION])
    # node_4 = DataParallelismNode(name='p4', parallel_attr=[[0.5,0.5], [0.5,0.5]])
    node_5 = TensorParallelismNode(name='p5', parallel_attr=[0.5, 0.5])
    node_6 = TensorParallelismNode(name='p6', parallel_attr=[0.5, 0.5])

    node_0.add_child(node_1)
    node_0.add_child(node_4)
    node_1.add_child(node_2)
    node_1.add_child(node_3)
    node_4.add_child(node_5)
    node_4.add_child(node_6)

    leaf_0 = BasicHardwareNode(idx = 0, name='l0')
    leaf_1 = BasicHardwareNode(idx = 1, name='l1')
    leaf_2 = BasicHardwareNode(idx = 2, name='l2')
    leaf_3 = BasicHardwareNode(idx = 3, name='l3')
    leaf_4 = BasicHardwareNode(idx = 4, name='l4')
    leaf_5 = BasicHardwareNode(idx = 5, name='l5')
    leaf_6 = BasicHardwareNode(idx = 6, name='l6')
    leaf_7 = BasicHardwareNode(idx = 7, name='l7')

    node_2.add_child(leaf_0)
    node_2.add_child(leaf_1)
    node_3.add_child(leaf_2)
    node_3.add_child(leaf_3)
    node_5.add_child(leaf_4)
    node_5.add_child(leaf_5)
    node_6.add_child(leaf_6)
    node_6.add_child(leaf_7)

    # return node_0, [leaf_0, leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, leaf_7]
    return node_0, [leaf_0, leaf_2, leaf_1, leaf_3, leaf_4, leaf_6, leaf_5, leaf_7]
