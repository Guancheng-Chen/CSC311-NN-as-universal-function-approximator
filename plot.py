"""画图：把几次训练的记录并排画成一帧，再把所有帧合成 GIF。"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data import DOMAIN_MAX, DOMAIN_MIN

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TRUE_CURVE_COLOR = "#008300"
LEVELS = np.linspace(-4, 4, 17)


def plot_frame(records, truth, step, title):
    """画出所有 records 在第 step 次更新后的样子，返回一张 PIL 图片。

    上排每个模型一格：颜色是模型输出（蓝负红正），黑线是模型输出 = 0 的曲线，
    绿线是真正的曲线 f = 0。下排是所有模型到目前为止的 loss 曲线。
    """
    n = len(records)
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, truth.shape[0])
    fig = plt.figure(figsize=(3.3 * n, 5.9), dpi=80)
    grid = fig.add_gridspec(2, n, height_ratios=[3, 1.5])

    for i, (record, color) in enumerate(zip(records, SERIES_COLORS)):
        zz = record.snapshots[step]
        ax = fig.add_subplot(grid[0, i])
        ax.contourf(axis, axis, zz, levels=LEVELS, cmap="RdBu_r", extend="both")
        ax.contour(axis, axis, truth, levels=[0], colors=TRUE_CURVE_COLOR,
                   linewidths=5, alpha=0.75)
        if zz.min() < 0 < zz.max():
            ax.contour(axis, axis, zz, levels=[0], colors="black", linewidths=1.8)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        if i == 0:
            ax.set_ylabel("$x_2$")
        ax.set_title(f"{record.label}\nloss {record.losses[step]:.4f}", fontsize=11)
        # 边框颜色和下面 loss 曲线的颜色一一对应
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)

    ax = fig.add_subplot(grid[1, :])
    for record, color in zip(records, SERIES_COLORS):
        steps = np.arange(1, step + 1)
        ax.plot(steps, record.losses[1:step + 1], color=color, linewidth=2,
                label=record.label)
    total = len(records[0].losses) - 1
    lowest = min(min(r.losses) for r in records)
    highest = max(max(r.losses) for r in records)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, total)
    ax.set_ylim(lowest / 2, highest * 2)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss")
    ax.grid(True, which="major", color="#e4e4e0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", ncol=n, frameon=False, fontsize=9)

    fig.suptitle(f"{title}  ·  step {step}\n"
                 "black: model output = 0     green: true curve "
                 "$x_1 + x_2 - 3x_1x_2 = 0$", fontsize=12)
    fig.tight_layout()
    fig.canvas.draw()
    image = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                             fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return image


def make_gif(records, truth, at_steps, title, path, duration_ms=80, hold_last_ms=3000):
    """每个快照 step 画一帧，合成 GIF，最后一帧多停一会儿。"""
    frames = [plot_frame(records, truth, step, title) for step in at_steps]
    frames = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    durations = [duration_ms] * (len(frames) - 1) + [hold_last_ms]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)
