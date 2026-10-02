"""模型：隐藏层数、宽度、激活函数都可配置的 MLP。"""

from torch import nn

# "linear" 就是不加激活函数：无论叠多少层，整个网络仍然只是一个线性函数。
ACTIVATIONS = {
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """2 个输入 -> depth 个宽度为 width 的隐藏层 -> 1 个线性输出。"""

    def __init__(self, depth, width, activation):
        super().__init__()
        layers = []
        n_in = 2
        for _ in range(depth):
            # 每个隐藏层：h = activation(W h_prev + b)
            layers += [nn.Linear(n_in, width), ACTIVATIONS[activation]()]
            n_in = width
        # 输出层不加激活函数，直接输出一个实数
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
