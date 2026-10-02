"""训练：只负责更新参数，不知道数据从哪来、也不知道怎么画图。"""

import torch
from torch import nn


def train(model, X, Y, lr=0.01, steps=3000, on_step=None):
    """用 Adam 做 full-batch 训练，固定更新 steps 次，最小化 MSE loss。

    on_step(step, loss) 会在初始化后（step 0）和每一次参数更新后被调用一次，
    调用方可以用它来记录快照、打日志等。返回最终 loss。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loss = loss_fn(model(X), Y)
    if on_step is not None:
        on_step(0, loss.item())

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss_fn(model(X), Y)
        if on_step is not None:
            on_step(step, loss.item())

    return loss.item()
