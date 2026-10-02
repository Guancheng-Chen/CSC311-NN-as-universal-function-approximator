"""Fit the curve x1 + x2 - 3·x1·x2 = 0 with MLPs, comparing activations and depths."""

import argparse
from pathlib import Path

import torch

from data import make_curve_points, make_data, make_grid, target
from model import MLP
from plot import make_gif
from record import Recorder, snapshot_steps
from train import train

PLOT_GRID_N = 120

# 两组对比实验，每组只改变一个超参。每一项是 (图上的标签, MLP 的参数)。
EXPERIMENTS = {
    "activations": {
        "title": "Same depth (2 hidden layers × 16 units), different activations",
        "configs": [
            (name, dict(depth=2, width=16, activation=name))
            for name in ("linear", "relu", "sigmoid", "tanh")
        ],
    },
    "depths": {
        "title": "Same activation (ReLU, 6 units per layer), different depths",
        "configs": [
            (f"{depth} hidden layer{'s' if depth > 1 else ''}",
             dict(depth=depth, width=6, activation="relu"))
            for depth in (1, 2, 3, 4)
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--frames", type=int, default=100,
                        help="roughly how many snapshots go into each GIF")
    parser.add_argument("--demo-dir", type=Path, default=Path(__file__).parent / "demo")
    parser.add_argument("--images", action="store_true",
                        help="regenerate the comparison GIFs in --demo-dir")
    return parser.parse_args()


def report(name, records, models, curve_points):
    print(f"[{name}]")
    for record, model in zip(records, models):
        n_params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            curve_error = model(curve_points).abs().mean().item()
        print(f"  {record.label:16s} params {n_params:4d}   "
              f"loss {record.losses[-1]:.5f}   mean |y| on curve {curve_error:.4f}")


def main():
    args = parse_args()

    # 1. 数据：训练用的网格点、画图用的更密的网格、以及曲线上的点
    X, Y = make_data()
    plot_grid = make_grid(PLOT_GRID_N)
    truth = target(plot_grid).reshape(PLOT_GRID_N, PLOT_GRID_N).numpy()
    curve_points = make_curve_points()
    at_steps = snapshot_steps(args.steps, args.frames)

    for name, experiment in EXPERIMENTS.items():
        records, models = [], []
        for label, model_kwargs in experiment["configs"]:
            # 2. 模型：每个配置都从同一个 seed 出发做随机初始化
            torch.manual_seed(args.seed)
            model = MLP(**model_kwargs)

            # 3. 训练：Recorder 作为回调把过程记下来，训练本身不管画图
            record = Recorder(label, model, plot_grid, at_steps)
            train(model, X, Y, lr=args.lr, steps=args.steps, on_step=record)
            records.append(record)
            models.append(model)

        # 4. 收尾：打印结果；加了 --images 才重新生成 GIF
        report(name, records, models, curve_points)
        if args.images:
            args.demo_dir.mkdir(exist_ok=True)
            path = args.demo_dir / f"{name}.gif"
            make_gif(records, truth, at_steps, experiment["title"], path)
            print(f"  saved {path}")


if __name__ == "__main__":
    main()
