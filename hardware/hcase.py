from hardware.hnode import *

def build_case_0():
    # 4 npu + 4 pim, tightly coupled
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

    # Type-1 针对 pcase-3
    # run_sim.py 中对于 idx 字段是不敏感的，所以 Type-0 和 Type-1 是等效的。
    # 但是 run_evo.py 中需要通过 idx 字段进行 device mapping，所以会有所区别。
    # device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_1 = HwUnit(idx=4, name="pim_0", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})
    # device_2 = HwUnit(idx=1, name="npu_1", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_3 = HwUnit(idx=5, name="pim_1", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})
    # device_4 = HwUnit(idx=2, name="npu_2", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_5 = HwUnit(idx=6, name="pim_2", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})
    # device_6 = HwUnit(idx=3, name="npu_3", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_7 = HwUnit(idx=7, name="pim_3", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})

    # Type-2 针对 pcase-4、pcase-12
    # device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_1 = HwUnit(idx=2, name="pim_0", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})
    # device_2 = HwUnit(idx=1, name="npu_1", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_3 = HwUnit(idx=3, name="pim_1", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})
    # device_4 = HwUnit(idx=4, name="npu_2", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_5 = HwUnit(idx=6, name="pim_2", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})
    # device_6 = HwUnit(idx=5, name="npu_3", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    # device_7 = HwUnit(idx=7, name="pim_3", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})

    device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2, "mem_cap": 40})
    device_1 = HwUnit(idx=4, name="pim_0", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2, "mem_cap": 40})
    device_2 = HwUnit(idx=1, name="npu_1", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2, "mem_cap": 40})
    device_3 = HwUnit(idx=5, name="pim_1", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2, "mem_cap": 40})
    device_4 = HwUnit(idx=2, name="npu_2", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2, "mem_cap": 40})
    device_5 = HwUnit(idx=6, name="pim_2", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2, "mem_cap": 40})
    device_6 = HwUnit(idx=3, name="npu_3", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2, "mem_cap": 40})
    device_7 = HwUnit(idx=7, name="pim_3", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2, "mem_cap": 40})

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
    # 4 npu + 4 pim, two level
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
    # 2 npu + 2 pim
    root = HwGroup(idx=0, name="root", meta={"bw": 250, "lat": 5000})  # 250 GB/s, 3 us

    devices = []

    card_0 = HwGroup(idx=0, name="card_0", meta={"bw": 1500, "lat": 50})  # 1500 GB/s, 50 ns
    card_1 = HwGroup(idx=1, name="card_1", meta={"bw": 1500, "lat": 50})

    root.add(card_0)
    root.add(card_1)

    device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    device_1 = HwUnit(idx=1, name="pim_0", meta={"type": "PIM", "flops": 15, "bw": 15, "byte": 2})
    device_2 = HwUnit(idx=2, name="npu_1", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    device_3 = HwUnit(idx=3, name="pim_1", meta={"type": "PIM", "flops": 15, "bw": 15, "byte": 2})

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
    # 4 npu
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


def build_case_4():
    # 4 pim
    root = HwGroup(idx=0, name="root", meta={"bw": 250, "lat": 5000})  # 250 GB/s, 3 us

    devices = []

    device_0 = HwUnit(idx=0, name="pim_0", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})
    device_1 = HwUnit(idx=1, name="pim_1", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})
    device_2 = HwUnit(idx=2, name="pim_2", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})
    device_3 = HwUnit(idx=3, name="pim_3", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})

    devices.append(device_0)
    devices.append(device_1)
    devices.append(device_2)
    devices.append(device_3)

    root.add(device_0)
    root.add(device_1)
    root.add(device_2)
    root.add(device_3)

    return root, devices


def build_case_5():
    # 2 npu
    root = HwGroup(idx=0, name="root", meta={"bw": 250, "lat": 5000})  # 250 GB/s, 3 us

    devices = []

    device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    device_1 = HwUnit(idx=1, name="npu_1", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})

    devices.append(device_0)
    devices.append(device_1)

    root.add(device_0)
    root.add(device_1)

    return root, devices


def build_case_6():
    # 2 pim
    root = HwGroup(idx=0, name="root", meta={"bw": 250, "lat": 5000})  # 250 GB/s, 3 us

    devices = []

    device_0 = HwUnit(idx=0, name="pim_0", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})
    device_1 = HwUnit(idx=1, name="pim_1", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})
    # device_0 = HwUnit(idx=0, name="pim_0", meta={"type": "PIM", "flops": 128, "bw": 16, "byte": 2})
    # device_1 = HwUnit(idx=1, name="pim_1", meta={"type": "PIM", "flops": 128, "bw": 16, "byte": 2})

    devices.append(device_0)
    devices.append(device_1)

    root.add(device_0)
    root.add(device_1)

    return root, devices

def build_case_7():
    # 1 npu + 1 pim
    root = HwGroup(idx=0, name="root", meta={"bw": 250, "lat": 5000})  # 250 GB/s, 3 us

    devices = []

    device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2})
    device_1 = HwUnit(idx=1, name="pim_1", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2})

    devices.append(device_0)
    devices.append(device_1)

    root.add(device_0)
    root.add(device_1)

    return root, devices

def build_case_8():
    # 8 npu + 8 pim, tightly coupled

    root = HwGroup(idx=0, name="root", meta={"bw": 250, "lat": 5000})  # 250 GB/s, 3 us

    devices = []

    card_0 = HwGroup(idx=0, name="card_0", meta={"bw": 1500, "lat": 50})  # 1500 GB/s, 50 ns
    card_1 = HwGroup(idx=1, name="card_1", meta={"bw": 1500, "lat": 50})
    card_2 = HwGroup(idx=2, name="card_2", meta={"bw": 1500, "lat": 50})
    card_3 = HwGroup(idx=3, name="card_3", meta={"bw": 1500, "lat": 50})
    card_4 = HwGroup(idx=0, name="card_0", meta={"bw": 1500, "lat": 50})  # 1500 GB/s, 50 ns
    card_5 = HwGroup(idx=1, name="card_1", meta={"bw": 1500, "lat": 50})
    card_6 = HwGroup(idx=2, name="card_2", meta={"bw": 1500, "lat": 50})
    card_7 = HwGroup(idx=3, name="card_3", meta={"bw": 1500, "lat": 50})

    root.add(card_0)
    root.add(card_1)
    root.add(card_2)
    root.add(card_3)
    root.add(card_4)
    root.add(card_5)
    root.add(card_6)
    root.add(card_7)

    npu_flops = 300
    npu_bw = 1.5
    npu_mem_cap = 40
    pim_flops = 16
    pim_bw = 16
    pim_mem_cap = 40
    device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": npu_flops, "bw": npu_bw, "byte": 2, "mem_cap": npu_mem_cap})
    device_1 = HwUnit(idx=4, name="pim_0", meta={"type": "PIM", "flops": pim_flops, "bw": pim_bw, "byte": 2, "mem_cap": pim_mem_cap})
    device_2 = HwUnit(idx=1, name="npu_1", meta={"type": "NPU", "flops": npu_flops, "bw": npu_bw, "byte": 2, "mem_cap": npu_mem_cap})
    device_3 = HwUnit(idx=5, name="pim_1", meta={"type": "PIM", "flops": pim_flops, "bw": pim_bw, "byte": 2, "mem_cap": pim_mem_cap})
    device_4 = HwUnit(idx=2, name="npu_2", meta={"type": "NPU", "flops": npu_flops, "bw": npu_bw, "byte": 2, "mem_cap": npu_mem_cap})
    device_5 = HwUnit(idx=6, name="pim_2", meta={"type": "PIM", "flops": pim_flops, "bw": pim_bw, "byte": 2, "mem_cap": pim_mem_cap})
    device_6 = HwUnit(idx=3, name="npu_3", meta={"type": "NPU", "flops": npu_flops, "bw": npu_bw, "byte": 2, "mem_cap": npu_mem_cap})
    device_7 = HwUnit(idx=7, name="pim_3", meta={"type": "PIM", "flops": pim_flops, "bw": pim_bw, "byte": 2, "mem_cap": pim_mem_cap})
    device_8 = HwUnit(idx=8, name="npu_4", meta={"type": "NPU", "flops": npu_flops, "bw": npu_bw, "byte": 2, "mem_cap": npu_mem_cap})
    device_9 = HwUnit(idx=12, name="pim_4", meta={"type": "PIM", "flops": pim_flops, "bw": pim_bw, "byte": 2, "mem_cap": pim_mem_cap})
    device_10 = HwUnit(idx=9, name="npu_5", meta={"type": "NPU", "flops": npu_flops, "bw": npu_bw, "byte": 2, "mem_cap": npu_mem_cap})
    device_11 = HwUnit(idx=13, name="pim_5", meta={"type": "PIM", "flops": pim_flops, "bw": pim_bw, "byte": 2, "mem_cap": pim_mem_cap})
    device_12 = HwUnit(idx=10, name="npu_6", meta={"type": "NPU", "flops": npu_flops, "bw": npu_bw, "byte": 2, "mem_cap": npu_mem_cap})
    device_13 = HwUnit(idx=14, name="pim_6", meta={"type": "PIM", "flops": pim_flops, "bw": pim_bw, "byte": 2, "mem_cap": pim_mem_cap})
    device_14 = HwUnit(idx=11, name="npu_7", meta={"type": "NPU", "flops": npu_flops, "bw": npu_bw, "byte": 2, "mem_cap": npu_mem_cap})
    device_15 = HwUnit(idx=15, name="pim_7", meta={"type": "PIM", "flops": pim_flops, "bw": pim_bw, "byte": 2, "mem_cap": pim_mem_cap})

    devices.append(device_0)
    devices.append(device_1)
    devices.append(device_2)
    devices.append(device_3)
    devices.append(device_4)
    devices.append(device_5)
    devices.append(device_6)
    devices.append(device_7)
    devices.append(device_8)
    devices.append(device_9)
    devices.append(device_10)
    devices.append(device_11)
    devices.append(device_12)
    devices.append(device_13)
    devices.append(device_14)
    devices.append(device_15)

    card_0.add(device_0)
    card_0.add(device_1)
    card_1.add(device_2)
    card_1.add(device_3)
    card_2.add(device_4)
    card_2.add(device_5)
    card_3.add(device_6)
    card_3.add(device_7)
    card_4.add(device_8)
    card_4.add(device_9)
    card_5.add(device_10)
    card_5.add(device_11)
    card_6.add(device_12)
    card_6.add(device_13)
    card_7.add(device_14)
    card_7.add(device_15)

    return root, devices

def build_case_9():
    # 4 npu + 4 pim, loosely coupled
    root = HwGroup(idx=0, name="root", meta={"bw": 250, "lat": 5000})  # 250 GB/s, 3 us

    devices = []

    device_0 = HwUnit(idx=0, name="npu_0", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2, "mem_cap": 40})
    device_1 = HwUnit(idx=4, name="pim_0", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2, "mem_cap": 40})
    device_2 = HwUnit(idx=1, name="npu_1", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2, "mem_cap": 40})
    device_3 = HwUnit(idx=5, name="pim_1", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2, "mem_cap": 40})
    device_4 = HwUnit(idx=2, name="npu_2", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2, "mem_cap": 40})
    device_5 = HwUnit(idx=6, name="pim_2", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2, "mem_cap": 40})
    device_6 = HwUnit(idx=3, name="npu_3", meta={"type": "NPU", "flops": 300, "bw": 1.5, "byte": 2, "mem_cap": 40})
    device_7 = HwUnit(idx=7, name="pim_3", meta={"type": "PIM", "flops": 16, "bw": 16, "byte": 2, "mem_cap": 40})

    devices.append(device_0)
    devices.append(device_1)
    devices.append(device_2)
    devices.append(device_3)
    devices.append(device_4)
    devices.append(device_5)
    devices.append(device_6)
    devices.append(device_7)

    root.add(device_0)
    root.add(device_1)
    root.add(device_2)
    root.add(device_3)
    root.add(device_4)
    root.add(device_5)
    root.add(device_6)
    root.add(device_7)

    return root, devices