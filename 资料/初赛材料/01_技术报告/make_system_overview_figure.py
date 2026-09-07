# -*- coding: utf-8 -*-
"""Generate paper-style system experiment overview (图1) — fully drawn, no photo embeds."""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Ellipse, Arc, Polygon, FancyArrow
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
OUT_DIRS = [ROOT / "figures", ROOT / "交稿" / "figures"]

C_TITLE = "#1A2332"
C_MUTED = "#5A6570"
C_BLUE = "#2C5F8A"
C_TEAL = "#2A9D6E"
C_ORANGE = "#C45C26"
C_PANEL = "#F7F8FA"
C_LINE = "#D0D5DC"
C_SOFT = "#E8EEF4"
C_SOFT_G = "#E6F2EC"
C_SOFT_O = "#F8EDE6"
C_WIRE = ["#E74C3C", "#E67E22", "#F1C40F", "#27AE60", "#3498DB", "#9B59B6", "#8E44AD", "#2C3E50"]


def _setup_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def _rounded(ax, xy, w, h, *, fc, ec="#C8CDD4", lw=1.0, r=0.012, z=1, alpha=1.0):
    p = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        alpha=alpha,
        zorder=z,
        mutation_aspect=0.55,
    )
    ax.add_patch(p)
    return p


def _arrow(ax, p0, p1, *, color="#6A7580", lw=1.6, style="-|>", ms=11, z=5):
    a = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=f"{style},head_length=3.2,head_width=2.0",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        zorder=z,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(a)


def _panel_frame(ax, x0, y0, w, h, title, accent, badge):
    _rounded(ax, (x0, y0), w, h, fc=C_PANEL, ec=C_LINE, lw=1.15, r=0.014, z=1)
    ax.add_patch(Rectangle((x0, y0 + h - 0.006), w, 0.006, facecolor=accent, edgecolor="none", zorder=3))
    ax.text(
        x0 + 0.018,
        y0 + h - 0.038,
        badge,
        fontsize=8.5,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=4,
        bbox=dict(boxstyle="round,pad=0.22", fc=accent, ec="none"),
    )
    ax.text(x0 + 0.055, y0 + h - 0.038, title, fontsize=10.5, fontweight="bold", color=C_TITLE, va="center", zorder=4)


def _draw_subject_with_cap(ax, cx, cy, scale=1.0):
    """Stylized seated subject + strap EEG cap (from wearing photo)."""
    s = scale
    # torso / desk
    ax.add_patch(
        FancyBboxPatch(
            (cx - 0.055 * s, cy - 0.095 * s),
            0.11 * s,
            0.08 * s,
            boxstyle="round,pad=0.004,rounding_size=0.01",
            fc="#4A5568",
            ec="none",
            zorder=2,
        )
    )
    # head
    ax.add_patch(Circle((cx, cy + 0.02 * s), 0.038 * s, fc="#F0D5BE", ec="#C9A88A", lw=0.8, zorder=3))
    # strap cap (black elastic web)
    ax.add_patch(Arc((cx, cy + 0.035 * s), 0.078 * s, 0.055 * s, theta1=10, theta2=170, color="#1A1A1A", lw=2.2, zorder=4))
    ax.add_patch(Arc((cx, cy + 0.018 * s), 0.072 * s, 0.048 * s, theta1=200, theta2=340, color="#1A1A1A", lw=1.6, zorder=4))
    # electrode holders (yellow/translucent like hardware photo)
    for ang, r in [(40, 0.032), (90, 0.034), (140, 0.032), (70, 0.028), (110, 0.028)]:
        rad = np.deg2rad(ang)
        ex = cx + r * s * np.cos(rad)
        ey = cy + 0.02 * s + r * s * np.sin(rad) * 0.7
        ax.add_patch(Circle((ex, ey), 0.0065 * s, fc="#F5E6A8", ec="#C4A84A", lw=0.6, zorder=5))
    # rainbow cable bundle from nape
    for i, col in enumerate(C_WIRE[:6]):
        ax.plot(
            [cx + 0.03 * s, cx + 0.07 * s + i * 0.004],
            [cy + 0.005 * s, cy - 0.02 * s - i * 0.003],
            color=col,
            lw=1.1,
            solid_capstyle="round",
            zorder=4,
        )
    # laptop
    ax.add_patch(Rectangle((cx - 0.02 * s, cy - 0.055 * s), 0.065 * s, 0.038 * s, fc="#2B3340", ec="none", zorder=3))
    ax.add_patch(Rectangle((cx - 0.017 * s, cy - 0.05 * s), 0.055 * s, 0.028 * s, fc="#A8D4E8", ec="none", zorder=4))


def _draw_cyton(ax, x, y, w=0.07, h=0.055):
    """Schematic OpenBCI Cyton board."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.006", fc="#F5F5F5", ec="#888", lw=0.9, zorder=3))
    ax.add_patch(Rectangle((x + 0.008, y + 0.03), 0.022, 0.016, fc="#2C3E50", ec="none", zorder=4))
    ax.add_patch(Circle((x + 0.05, y + 0.038), 0.008, fc="#27AE60", ec="#1E8449", lw=0.5, zorder=4))
    for i in range(8):
        ax.add_patch(Rectangle((x + 0.006 + i * 0.0075, y + 0.006), 0.005, 0.012, fc=C_WIRE[i], ec="none", zorder=4))
    ax.text(x + w / 2, y - 0.012, "OpenBCI Cyton", fontsize=7, ha="center", color=C_MUTED)


def _draw_eeg_wave(ax, x, y, w, h):
    t = np.linspace(0, 4 * np.pi, 200)
    for i, (amp, phase, col) in enumerate(
        [(0.9, 0, C_BLUE), (0.65, 0.7, "#5B8FB8"), (0.45, 1.4, "#8FB4D0")]
    ):
        yy = y + h * 0.55 + amp * h * 0.28 * np.sin(t + phase) * np.exp(-0.08 * t)
        xx = x + 0.01 + (w - 0.02) * t / t[-1]
        ax.plot(xx, yy - i * 0.012, color=col, lw=1.15, solid_capstyle="round", zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#C5CDD6", lw=0.8, zorder=2))


def _draw_game_ui(ax, x, y, w, h):
    """Cue UI mock: desk + cup + arms (from game screenshot)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.008", fc="#E8ECF0", ec="#B0B8C0", lw=0.9, zorder=2))
    # desk
    ax.add_patch(Polygon([(x + 0.01, y + 0.02), (x + w - 0.01, y + 0.02), (x + w - 0.02, y + 0.055), (x + 0.02, y + 0.055)], closed=True, fc="#C4A574", ec="none", zorder=3))
    # arms (blue sleeves)
    ax.add_patch(FancyBboxPatch((x + 0.018, y + 0.035), 0.045, 0.018, boxstyle="round,pad=0.001,rounding_size=0.006", fc="#4A90C8", ec="none", zorder=4))
    ax.add_patch(FancyBboxPatch((x + w - 0.063, y + 0.035), 0.045, 0.018, boxstyle="round,pad=0.001,rounding_size=0.006", fc="#4A90C8", ec="none", zorder=4))
    # cup
    ax.add_patch(FancyBboxPatch((x + w / 2 - 0.012, y + 0.048), 0.024, 0.028, boxstyle="round,pad=0.001,rounding_size=0.004", fc="#F4F4F4", ec="#BBB", lw=0.6, zorder=5))
    ax.add_patch(Arc((x + w / 2 + 0.014, y + 0.06), 0.014, 0.016, theta1=-60, theta2=60, color="#AAA", lw=1.0, zorder=5))
    # cue banner
    ax.add_patch(FancyBboxPatch((x + 0.015, y + h - 0.042), w - 0.03, 0.032, boxstyle="round,pad=0.002,rounding_size=0.006", fc="#2A3340", ec="none", alpha=0.88, zorder=6))
    ax.text(x + w / 2, y + h - 0.026, "想象：左手握紧杯子", fontsize=6.5, ha="center", va="center", color="white", zorder=7)


def _draw_operator(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.008", fc="#1E2430", ec="#4A5568", lw=0.9, zorder=2))
    rows = [
        ("phase / trial", "#7FDBDA"),
        ("[P] 暂停  [N] 代确认", "#E8E8E8"),
        ("[G] 准入  [R] Reject", "#E8E8E8"),
        ("状态：进行中", "#F0C674"),
    ]
    for i, (t, c) in enumerate(rows):
        ax.text(x + 0.01, y + h - 0.018 - i * 0.022, t, fontsize=6.2, color=c, va="center", zorder=3)


def build() -> None:
    _setup_font()
    fig = plt.figure(figsize=(14.2, 8.6), dpi=200, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # title
    ax.text(0.5, 0.965, "运动想象 BCI 完整系统实验总览", fontsize=16, fontweight="bold", ha="center", va="center", color=C_TITLE)
    ax.text(
        0.5,
        0.932,
        "现场佩戴 → 采集放大 → 预处理供窗 → CausalFuse-8 解码 → 人机界面闭环；底部为采后 Leave-Next 适配",
        fontsize=8.5,
        ha="center",
        va="center",
        color=C_MUTED,
    )

    # five panels
    y0, h = 0.30, 0.575
    gap = 0.012
    xs = [0.025]
    widths = [0.175, 0.175, 0.175, 0.22, 0.175]
    for i in range(1, 5):
        xs.append(xs[-1] + widths[i - 1] + gap)
    titles = ["现场", "采集", "预处理供窗", "CausalFuse-8 解码", "人机界面"]
    accents = [C_BLUE, C_BLUE, C_TEAL, C_ORANGE, C_BLUE]
    badges = ["A", "B", "C", "D", "E"]

    for x, w, title, acc, badge in zip(xs, widths, titles, accents, badges):
        _panel_frame(ax, x, y0, w, h, title, acc, badge)

    # --- A 现场 ---
    xa, wa = xs[0], widths[0]
    _draw_subject_with_cap(ax, xa + wa * 0.48, y0 + 0.34, scale=1.15)
    ax.text(xa + wa / 2, y0 + 0.18, "被试佩戴八通道弹性带帽", fontsize=8, ha="center", color=C_TITLE)
    ax.text(xa + wa / 2, y0 + 0.145, "彩色线束接至放大器", fontsize=7.2, ha="center", color=C_MUTED)
    ax.text(xa + wa / 2, y0 + 0.105, "实验室桌面 · 注视任务屏", fontsize=7.2, ha="center", color=C_MUTED)
    _rounded(ax, (xa + 0.02, y0 + 0.035), wa - 0.04, 0.05, fc=C_SOFT, ec="#B8C4D0", lw=0.8, r=0.01)
    ax.text(xa + wa / 2, y0 + 0.06, "现场闭环起点", fontsize=7.5, ha="center", color=C_BLUE, fontweight="bold")

    # --- B 采集 ---
    xb, wb = xs[1], widths[1]
    # strap cap icon small
    ax.add_patch(Ellipse((xb + 0.055, y0 + 0.42), 0.055, 0.04, fc="#2A2A2A", ec="none", zorder=3))
    for i, ang in enumerate([30, 70, 110, 150]):
        rad = np.deg2rad(ang)
        ax.add_patch(
            Circle(
                (xb + 0.055 + 0.022 * np.cos(rad), y0 + 0.42 + 0.014 * np.sin(rad)),
                0.005,
                fc="#F5E6A8",
                ec="#C4A84A",
                lw=0.4,
                zorder=4,
            )
        )
    ax.text(xb + 0.055, y0 + 0.365, "8 通道帽", fontsize=7, ha="center", color=C_MUTED)
    # wires to cyton
    for i, col in enumerate(C_WIRE):
        ax.plot(
            [xb + 0.085, xb + 0.10],
            [y0 + 0.42 - 0.01 + i * 0.003, y0 + 0.355 + i * 0.004],
            color=col,
            lw=0.9,
            zorder=3,
        )
    _draw_cyton(ax, xb + 0.095, y0 + 0.33, w=0.065, h=0.05)
    _rounded(ax, (xb + 0.02, y0 + 0.20), wb - 0.04, 0.09, fc="white", ec=C_LINE, lw=0.9, r=0.01)
    ax.text(xb + wb / 2, y0 + 0.265, "无线 / USB 链路", fontsize=7.5, ha="center", color=C_TITLE, fontweight="bold")
    ax.text(xb + wb / 2, y0 + 0.235, "OpenBCI → PC", fontsize=7, ha="center", color=C_MUTED)
    ax.text(xb + wb / 2, y0 + 0.155, "采样进入本机缓冲", fontsize=8, ha="center", color=C_TITLE)
    ax.text(xb + wb / 2, y0 + 0.12, "供下游 LSL 发布", fontsize=7.2, ha="center", color=C_MUTED)
    _rounded(ax, (xb + 0.02, y0 + 0.035), wb - 0.04, 0.05, fc=C_SOFT, ec="#B8C4D0", lw=0.8, r=0.01)
    ax.text(xb + wb / 2, y0 + 0.06, "硬件采集层", fontsize=7.5, ha="center", color=C_BLUE, fontweight="bold")

    # --- C 预处理 ---
    xc, wc = xs[2], widths[2]
    _draw_eeg_wave(ax, xc + 0.02, y0 + 0.38, wc - 0.04, 0.12)
    ax.text(xc + wc / 2, y0 + 0.355, "多通道连续脑电", fontsize=7, ha="center", color=C_MUTED)
    steps = [
        ("带通 / 陷波滤波", C_SOFT_G),
        ("伪迹抑制与质控", C_SOFT),
        ("滑动窗裁剪", C_SOFT_O),
    ]
    for i, (lab, fc) in enumerate(steps):
        yy = y0 + 0.28 - i * 0.055
        _rounded(ax, (xc + 0.025, yy), wc - 0.05, 0.042, fc=fc, ec="#C5CDD6", lw=0.8, r=0.01)
        ax.text(xc + wc / 2, yy + 0.021, lab, fontsize=8, ha="center", va="center", color=C_TITLE)
        if i < 2:
            _arrow(ax, (xc + wc / 2, yy - 0.002), (xc + wc / 2, yy - 0.01), color=C_TEAL, lw=1.2, ms=8)
    ax.text(xc + wc / 2, y0 + 0.105, "窗长 3 s · 步长 100 ms", fontsize=7.5, ha="center", color=C_TEAL, fontweight="bold")
    ax.text(xc + wc / 2, y0 + 0.075, "LSL 实时发布特征窗", fontsize=7, ha="center", color=C_MUTED)
    _rounded(ax, (xc + 0.02, y0 + 0.035), wc - 0.04, 0.028, fc=C_SOFT_G, ec="#A8C9B8", lw=0.7, r=0.008)
    ax.text(xc + wc / 2, y0 + 0.049, "供窗就绪", fontsize=7, ha="center", color=C_TEAL)

    # --- D CausalFuse-8 ---
    xd, wd = xs[3], widths[3]
    members = ["ShallowFBCSPNet", "Shallow-b", "EEGNet", "Conformer"]
    for i, m in enumerate(members):
        col = i % 2
        row = i // 2
        xx = xd + 0.02 + col * 0.095
        yy = y0 + 0.42 - row * 0.07
        _rounded(ax, (xx, yy), 0.085, 0.055, fc="white", ec="#D4A574", lw=1.0, r=0.01)
        ax.text(xx + 0.0425, yy + 0.028, m, fontsize=7.5, ha="center", va="center", color=C_TITLE)
    ax.text(xd + wd / 2, y0 + 0.355, "四成员异构基学习器", fontsize=7.2, ha="center", color=C_MUTED)
    _arrow(ax, (xd + wd / 2, y0 + 0.34), (xd + wd / 2, y0 + 0.30), color=C_ORANGE, lw=1.5)
    _rounded(ax, (xd + 0.035, y0 + 0.22), wd - 0.07, 0.07, fc=C_SOFT_O, ec="#E0B090", lw=1.1, r=0.012)
    ax.text(xd + wd / 2, y0 + 0.265, "因果融合 / 门控加权", fontsize=8.5, ha="center", va="center", color=C_ORANGE, fontweight="bold")
    ax.text(xd + wd / 2, y0 + 0.235, "CausalFuse-8", fontsize=7.5, ha="center", va="center", color=C_MUTED)
    _arrow(ax, (xd + wd / 2, y0 + 0.21), (xd + wd / 2, y0 + 0.17), color=C_ORANGE, lw=1.5)
    _rounded(ax, (xd + 0.04, y0 + 0.095), wd - 0.08, 0.06, fc="#2C5F8A", ec="none", r=0.012)
    ax.text(xd + wd / 2, y0 + 0.135, "三类 MI 决策", fontsize=8.5, ha="center", va="center", color="white", fontweight="bold")
    ax.text(xd + wd / 2, y0 + 0.11, "左 / 右 / 休息", fontsize=7, ha="center", va="center", color="#D6E4F0")
    ax.text(xd + wd / 2, y0 + 0.055, "实时解码输出 → 界面", fontsize=7.2, ha="center", color=C_MUTED)

    # --- E 人机界面 ---
    xe, we = xs[4], widths[4]
    ax.text(xe + we / 2, y0 + 0.48, "被试 Cue 游戏", fontsize=8, ha="center", color=C_TITLE, fontweight="bold")
    _draw_game_ui(ax, xe + 0.018, y0 + 0.30, we - 0.036, 0.15)
    ax.text(xe + we / 2, y0 + 0.265, "操作者监控台", fontsize=8, ha="center", color=C_TITLE, fontweight="bold")
    _draw_operator(ax, xe + 0.018, y0 + 0.12, we - 0.036, 0.12)
    ax.text(xe + we / 2, y0 + 0.08, "空格确认 · 热键干预", fontsize=7, ha="center", color=C_MUTED)
    _rounded(ax, (xe + 0.02, y0 + 0.035), we - 0.04, 0.028, fc=C_SOFT, ec="#B8C4D0", lw=0.7, r=0.008)
    ax.text(xe + we / 2, y0 + 0.049, "人在回路闭环", fontsize=7, ha="center", color=C_BLUE)

    # horizontal flow arrows between panels
    for i in range(4):
        x_end = xs[i] + widths[i]
        x_start = xs[i + 1]
        mid_y = y0 + h * 0.55
        _arrow(ax, (x_end + 0.002, mid_y), (x_start - 0.002, mid_y), color="#8A95A0", lw=1.8, ms=12)

    # --- F bottom: Leave-Next ---
    yf, hf = 0.035, 0.22
    _rounded(ax, (0.025, yf), 0.95, hf, fc="#FAFBFC", ec=C_LINE, lw=1.15, r=0.014, z=1)
    ax.add_patch(Rectangle((0.025, yf + hf - 0.006), 0.95, 0.006, facecolor=C_TEAL, edgecolor="none", zorder=3))
    ax.text(
        0.045,
        yf + hf - 0.032,
        "F",
        fontsize=8.5,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=4,
        bbox=dict(boxstyle="round,pad=0.22", fc=C_TEAL, ec="none"),
    )
    ax.text(0.08, yf + hf - 0.032, "采后适配 · Leave-Next", fontsize=10.5, fontweight="bold", color=C_TITLE, va="center")

    # timeline rounds
    n_rounds = 5
    tx0, tx1 = 0.08, 0.62
    ty = yf + 0.095
    ax.plot([tx0, tx1], [ty, ty], color="#C5CDD6", lw=2.5, zorder=2, solid_capstyle="round")
    for i in range(n_rounds):
        xx = tx0 + (tx1 - tx0) * i / (n_rounds - 1)
        ax.add_patch(Circle((xx, ty), 0.014, fc=C_TEAL if i < n_rounds - 1 else C_ORANGE, ec="white", lw=1.5, zorder=4))
        ax.text(xx, ty + 0.035, f"R{i+1}", fontsize=7.5, ha="center", color=C_TITLE, fontweight="bold")
        if i < n_rounds - 1:
            ax.annotate(
                "",
                xy=(xx + (tx1 - tx0) / (n_rounds - 1) - 0.018, ty),
                xytext=(xx + 0.018, ty),
                arrowprops=dict(arrowstyle="->", color="#8A95A0", lw=1.2),
                zorder=3,
            )
    ax.text(0.35, yf + 0.045, "逐轮留出下一位被试 · 增量更新", fontsize=7.5, ha="center", color=C_MUTED)

    # FT box
    _rounded(ax, (0.66, yf + 0.04), 0.28, 0.13, fc=C_SOFT_G, ec="#A8C9B8", lw=1.0, r=0.012)
    ax.text(0.80, yf + 0.135, "微调 / 校准（FT）", fontsize=9, ha="center", color=C_TEAL, fontweight="bold")
    ax.text(0.80, yf + 0.095, "更新 CausalFuse-8", fontsize=7.5, ha="center", color=C_TITLE)
    ax.text(0.80, yf + 0.065, "再进入下一轮在线会话", fontsize=7.2, ha="center", color=C_MUTED)
    _arrow(ax, (0.63, ty), (0.655, yf + 0.105), color=C_TEAL, lw=1.5, ms=10)

    # footer note
    ax.text(
        0.5,
        0.012,
        "示意图按实物硬件与界面绘制，不嵌入实拍缩略图；术语与正文一致（OpenBCI Cyton · LSL · CausalFuse-8 · Leave-Next）",
        fontsize=6.8,
        ha="center",
        color="#9AA3AB",
    )

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        for name in ("图0_系统实验总览.png", "fig00_system_experiment_overview.png"):
            p = d / name
            fig.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.08, facecolor="white")
            print("wrote", p)
        p_pdf = d / "fig00_system_experiment_overview.pdf"
        fig.savefig(p_pdf, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        print("wrote", p_pdf)
    plt.close(fig)


if __name__ == "__main__":
    build()
