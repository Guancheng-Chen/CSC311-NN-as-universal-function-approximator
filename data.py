"""生成数据：目标函数 f(x1, x2) = x1 + x2 - 3·x1·x2，以及它的零点曲线。"""

import numpy as np
import torch

DOMAIN_MIN, DOMAIN_MAX = -1.0, 2.0


def target(x):
    """f(x1, x2) = x1 + x2 - 3·x1·x2。要拟合的曲线就是 f = 0 的那条等高线。"""
    x1, x2 = x[:, 0:1], x[:, 1:2]
    return x1 + x2 - 3 * x1 * x2


def make_grid(n):
    """在 [DOMAIN_MIN, DOMAIN_MAX]² 上取 n×n 的均匀网格，返回 n²×2 的输入。"""
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    xx, yy = np.meshgrid(axis, axis)
    return torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)


def make_data(n=41):
    """返回 (X, Y)：X 是 n²×2 的网格点，Y 是 n²×1 的 f(X)。"""
    X = make_grid(n)
    return X, target(X)


def make_curve_points(n=400):
    """返回正好落在曲线 f = 0 上的点（n'×2），用来衡量模型把曲线学得多准。

    由 f = 0 解出 x2 = x1 / (3·x1 - 1)，这是一条以 x1 = 1/3 为渐近线的双曲线，
    只保留落在定义域内的那部分。
    """
    x1 = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    x1 = x1[np.abs(3 * x1 - 1) > 1e-6]
    x2 = x1 / (3 * x1 - 1)
    inside = (x2 >= DOMAIN_MIN) & (x2 <= DOMAIN_MAX)
    return torch.tensor(np.c_[x1[inside], x2[inside]], dtype=torch.float32)
