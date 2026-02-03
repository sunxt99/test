# https://leopard-x.memofun.net/c/69437416-1b78-832c-8215-d2807cabdf0f

from hardware.htraversal import *

class HardwareTree:
    def __init__(self):
        self.root = None
        self.devices = None
        self.build()

    def build(self):
        # 构建一个硬件抽象架构示例
        self.root = HwGroup(idx=0, name="root", meta={"bw": 250, "lat": 5000}) # 250 GB/s, 3 us

        self.devices = []

        card_0 = HwGroup(idx=0, name="card_0", meta={"bw": 1500, "lat": 50}) # 1500 GB/s, 50 ns
        card_1 = HwGroup(idx=1, name="card_1", meta={"bw": 1500, "lat": 50})
        card_2 = HwGroup(idx=2, name="card_2", meta={"bw": 1500, "lat": 50})
        card_3 = HwGroup(idx=3, name="card_3", meta={"bw": 1500, "lat": 50})

        self.root.add(card_0)
        self.root.add(card_1)
        self.root.add(card_2)
        self.root.add(card_3)

        device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
        device_1 = HwUnit(idx=1, name="pim_0", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
        device_2 = HwUnit(idx=2, name="npu_1", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
        device_3 = HwUnit(idx=3, name="pim_1", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
        device_4 = HwUnit(idx=4, name="npu_2", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
        device_5 = HwUnit(idx=5, name="pim_2", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
        device_6 = HwUnit(idx=6, name="npu_3", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
        device_7 = HwUnit(idx=7, name="pim_3", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})

        self.devices.append(device_0)
        self.devices.append(device_1)
        self.devices.append(device_2)
        self.devices.append(device_3)
        self.devices.append(device_4)
        self.devices.append(device_5)
        self.devices.append(device_6)
        self.devices.append(device_7)

        card_0.add(device_0)
        card_0.add(device_1)
        card_1.add(device_2)
        card_1.add(device_3)
        card_2.add(device_4)
        card_2.add(device_5)
        card_3.add(device_6)
        card_3.add(device_7)

        for device in self.devices:
            fill_descendant_set(device)

        # 遍历 / 查找 / 聚合
        # 1) 打印所有叶子节点路径
        # for u in root.iter_leaves():
        #     print(u.path(), u.meta)

        # 2) 按路径找到节点
        # node = root.find_by_path("System/rack1/Chassis1/Board1/GPU1")
        # print(node.id, node.path())

        # 3) 统计总功耗
        # total_power = root.aggregate(lambda u: float(u.meta.get("power_w", 0.0)))
        # print("Total power:", total_power)

        # 打印所有叶子结点
        print("\n---------hardware tree---------")
        for n in self.root.iter_leaves():
            print(n.path())

    def build_2(self):
        self.root = HwGroup(idx=0, name="root", meta={"bw": 22, "lat": 20000}) # 22 GB/s, 20 us
        self.devices = []

        node_0 = HwGroup(idx=0, name="node_0", meta={"bw": 250, "lat": 3000}) # 600 GB/s, 3 us
        node_1 = HwGroup(idx=1, name="node_1", meta={"bw": 250, "lat": 3000})

        card_0 = HwGroup(idx=0, name="card_0", meta={"bw": 1500, "lat": 100}) # 1500 GB/s, 100 ns
        card_1 = HwGroup(idx=1, name="card_1", meta={"bw": 1500, "lat": 100})
        card_2 = HwGroup(idx=2, name="card_2", meta={"bw": 1500, "lat": 100})
        card_3 = HwGroup(idx=3, name="card_3", meta={"bw": 1500, "lat": 100})

        self.root.add(node_0)
        self.root.add(node_1)

        node_0.add(card_0)
        node_0.add(card_1)
        node_1.add(card_2)
        node_1.add(card_3)

        device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 3, "byte": 2})
        device_1 = HwUnit(idx=1, name="pim_0", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
        device_2 = HwUnit(idx=2, name="npu_1", meta={"type": "NPU", "flops": 300, "bw": 3, "byte": 2})
        device_3 = HwUnit(idx=3, name="pim_1", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
        device_4 = HwUnit(idx=4, name="npu_2", meta={"type": "NPU", "flops": 300, "bw": 3, "byte": 2})
        device_5 = HwUnit(idx=5, name="pim_2", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
        device_6 = HwUnit(idx=6, name="npu_3", meta={"type": "NPU", "flops": 300, "bw": 3, "byte": 2})
        device_7 = HwUnit(idx=7, name="pim_3", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})

        self.devices.append(device_0)
        self.devices.append(device_1)
        self.devices.append(device_2)
        self.devices.append(device_3)
        self.devices.append(device_4)
        self.devices.append(device_5)
        self.devices.append(device_6)
        self.devices.append(device_7)

        card_0.add(device_0)
        card_0.add(device_1)
        card_1.add(device_2)
        card_1.add(device_3)
        card_2.add(device_4)
        card_2.add(device_5)
        card_3.add(device_6)
        card_3.add(device_7)

        for device in self.devices:
            fill_descendant_set(device)

        print("\n---------hardware tree---------")
        for n in self.root.iter_leaves():
            print(n.path())


if __name__ == "__main__":
    htree = HardwareTree()
