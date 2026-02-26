from hardware.hnode import *

def build_case_0():
    # 构建一个硬件抽象架构示例
    root = HwGroup(idx=0, name="root", meta={"bw": 250, "lat": 5000})  # 250 GB/s, 3 us

    devices = []

    card_0 = HwGroup(idx=0, name="card_0", meta={"bw": 1500, "lat": 50})  # 1500 GB/s, 50 ns
    card_1 = HwGroup(idx=1, name="card_1", meta={"bw": 1500, "lat": 50})
    card_2 = HwGroup(idx=2, name="card_2", meta={"bw": 1500, "lat": 50})
    card_3 = HwGroup(idx=3, name="card_3", meta={"bw": 1500, "lat": 50})

    root.add(card_0)
    root.add(card_1)
    root.add(card_2)
    root.add(card_3)

    # Type-0
    # device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_1 = HwUnit(idx=1, name="pim_0", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
    # device_2 = HwUnit(idx=2, name="npu_1", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_3 = HwUnit(idx=3, name="pim_1", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
    # device_4 = HwUnit(idx=4, name="npu_2", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_5 = HwUnit(idx=5, name="pim_2", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
    # device_6 = HwUnit(idx=6, name="npu_3", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_7 = HwUnit(idx=7, name="pim_3", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})

    # Type-1 针对 pcase-3
    # pcase 中对于 idx 字段是不敏感的，所以 Type-0 和 Type-1 是等效的。
    # device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_1 = HwUnit(idx=4, name="pim_0", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
    # device_2 = HwUnit(idx=1, name="npu_1", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_3 = HwUnit(idx=5, name="pim_1", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
    # device_4 = HwUnit(idx=2, name="npu_2", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_5 = HwUnit(idx=6, name="pim_2", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
    # device_6 = HwUnit(idx=3, name="npu_3", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_7 = HwUnit(idx=7, name="pim_3", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})

    # Type-2 针对 pcase-4
    device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    device_1 = HwUnit(idx=2, name="pim_0", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
    device_2 = HwUnit(idx=1, name="npu_1", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    device_3 = HwUnit(idx=3, name="pim_1", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
    device_4 = HwUnit(idx=4, name="npu_2", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    device_5 = HwUnit(idx=6, name="pim_2", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
    device_6 = HwUnit(idx=5, name="npu_3", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    device_7 = HwUnit(idx=7, name="pim_3", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})

    devices.append(device_0)
    devices.append(device_1)
    devices.append(device_2)
    devices.append(device_3)
    devices.append(device_4)
    devices.append(device_5)
    devices.append(device_6)
    devices.append(device_7)

    card_0.add(device_0)
    card_0.add(device_1)
    card_1.add(device_2)
    card_1.add(device_3)
    card_2.add(device_4)
    card_2.add(device_5)
    card_3.add(device_6)
    card_3.add(device_7)

    return root, devices


def build_case_1():
    root = HwGroup(idx=0, name="root", meta={"bw": 22, "lat": 20000})  # 22 GB/s, 20 us
    devices = []

    node_0 = HwGroup(idx=0, name="node_0", meta={"bw": 250, "lat": 3000})  # 600 GB/s, 3 us
    node_1 = HwGroup(idx=1, name="node_1", meta={"bw": 250, "lat": 3000})

    card_0 = HwGroup(idx=0, name="card_0", meta={"bw": 1500, "lat": 100})  # 1500 GB/s, 100 ns
    card_1 = HwGroup(idx=1, name="card_1", meta={"bw": 1500, "lat": 100})
    card_2 = HwGroup(idx=2, name="card_2", meta={"bw": 1500, "lat": 100})
    card_3 = HwGroup(idx=3, name="card_3", meta={"bw": 1500, "lat": 100})

    root.add(node_0)
    root.add(node_1)

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

    devices.append(device_0)
    devices.append(device_1)
    devices.append(device_2)
    devices.append(device_3)
    devices.append(device_4)
    devices.append(device_5)
    devices.append(device_6)
    devices.append(device_7)

    card_0.add(device_0)
    card_0.add(device_1)
    card_1.add(device_2)
    card_1.add(device_3)
    card_2.add(device_4)
    card_2.add(device_5)
    card_3.add(device_6)
    card_3.add(device_7)

    return root, devices


def build_case_2():
    # 构建一个硬件抽象架构示例
    root = HwGroup(idx=0, name="root", meta={"bw": 250, "lat": 5000})  # 250 GB/s, 3 us

    devices = []

    card_0 = HwGroup(idx=0, name="card_0", meta={"bw": 1500, "lat": 50})  # 1500 GB/s, 50 ns
    card_1 = HwGroup(idx=1, name="card_1", meta={"bw": 1500, "lat": 50})

    root.add(card_0)
    root.add(card_1)

    device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_1 = HwUnit(idx=1, name="pim_0", meta={"type": "PIM", "flops": 150, "bw": 8, "byte": 2})
    device_1 = HwUnit(idx=1, name="pim_0", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})
    device_2 = HwUnit(idx=2, name="npu_1", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_3 = HwUnit(idx=3, name="pim_1", meta={"type": "PIM", "flops": 150, "bw": 8, "byte": 2})
    device_3 = HwUnit(idx=3, name="pim_1", meta={"type": "PIM", "flops": 30, "bw": 30, "byte": 2})

    devices.append(device_0)
    devices.append(device_1)
    devices.append(device_2)
    devices.append(device_3)

    card_0.add(device_0)
    card_0.add(device_1)
    card_1.add(device_2)
    card_1.add(device_3)

    return root, devices


def build_case_3():
    # 构建一个硬件抽象架构示例
    root = HwGroup(idx=0, name="root", meta={"bw": 250, "lat": 5000})  # 250 GB/s, 3 us

    devices = []

    device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    device_1 = HwUnit(idx=1, name="npu_1", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    device_2 = HwUnit(idx=2, name="npu_2", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    device_3 = HwUnit(idx=3, name="npu_3", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})

    devices.append(device_0)
    devices.append(device_1)
    devices.append(device_2)
    devices.append(device_3)

    root.add(device_0)
    root.add(device_1)
    root.add(device_2)
    root.add(device_3)

    return root, devices
