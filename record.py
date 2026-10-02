"""记录：把一次训练的过程存下来，训练结束后再交给 plot.py 画图。"""

import numpy as np
import torch


def snapshot_steps(steps, n_frames):
    """挑出要存快照的 step：前期变化快所以取得密，后期按几何级数变稀。"""
    spaced = np.geomspace(1, steps, n_frames - 1).round().astype(int)
    return [0] + sorted(set(spaced.tolist()))


class Recorder:
    """作为 train() 的 on_step 回调：记下每一步的 loss，并在指定 step 存下模型在网格上的输出。"""

    def __init__(self, label, model, grid, at_steps):
        self.label = label
        self.model = model
        self.grid = grid
        self.at_steps = set(at_steps)
        self.losses = []     # losses[step] = 第 step 次更新后的 loss
        self.snapshots = {}  # snapshots[step] = 模型在 grid 上的输出（side×side）

    def __call__(self, step, loss):
        self.losses.append(loss)
        if step in self.at_steps:
            side = int(len(self.grid) ** 0.5)
            with torch.no_grad():
                self.snapshots[step] = self.model(self.grid).reshape(side, side).numpy()
