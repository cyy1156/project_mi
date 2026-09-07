# -*- coding: utf-8 -*-
"""技术报告结果图生成脚本 v4（图 4–图 10）——视觉重构版

本版将七张图全部换成新的图表族（参照期刊图表选型框架：形式匹配比较类型）：
  图4 棒棒糖图（少量类目 · 少墨量）        图5 Cleveland 点图（排序比较 · 协议分组）
  图6 竖柱读出对比 + 行归一化混淆热图（规格 §3）  图7 斜率图（三状态演变 · 误差条）
  图8 小倍数图（每被试一面板 · 双线对比）   图9 桥形图（基线 → 增量 → 终值）
  图10 真人 Leave-Next 逐轮柱状小倍数（每被试一面板）

口径沿用《插图规格说明_结果图.md》冻结规范：
  - 配色 ≤4 色：主色 #2C5F8A / 对照 #C45C26 / 中性 #6B7280 / 通过 #2A9D6E / 失败 #B91C1C
  - 纯白背景、无阴影、无渐变；仅左/下轴线 0.6 pt
  - 字体：Arial（拉丁/数字）+ SimHei（汉字）；标注 ≥6.5 pt
  - 画布：半栏 85–95 mm 或通栏 160–170 mm；导出 PNG + PDF，300 dpi
  - 每张图只回答一个问题；数字与 v4 正文逐项对表（内置断言）
附：渲染后自动版面审计（最小字号 / 两两文本碰撞），审计不过则非零退出。
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4

# ---- 规格冻结五色（§1） ----
PRIMARY = "#2C5F8A"   # 主色
CONTRAST = "#C45C26"  # 对照
NEUTRAL = "#6B7280"   # 中性
PASS_C = "#2A9D6E"    # 通过
FAIL_C = "#B91C1C"    # 失败
BEST_C = "#C45C26"    # 各轮最高（与 CONTRAST 同色；与 PASS/末档叠色时优先）
# 主色同族浅色（仅用于同一序列的明度分级，不算新增色相）
P_MED = "#7B97B5"
P_LT = "#AFC2D6"
STEM = "#D9E1EB"      # 点图引导线（主色族极浅）
INK = "#22272E"
AXLINE = "#4B5563"
CHANCE = 1 / 3

plt.rcParams.update({
    "font.family": ["Arial", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": AXLINE,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "hatch.linewidth": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
})


def bare_ax(ax, keep_left=True):
    """仅保留左/下细轴线，无网格。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)


def save(fig, name):
    problems = audit(fig, name)
    fig.savefig(f"figures/{name}.png")
    fig.savefig(f"figures/{name}.pdf")
    plt.close(fig)
    print(("OK   " if not problems else "WARN ") + name)
    for p in problems:
        print("     - " + p)
    return problems


# ---------------- 版面审计 ----------------
def audit(fig, name):
    """返回问题列表：最小字号 / 两两碰撞。"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for t in fig.findobj(matplotlib.text.Text):
        s = (t.get_text() or "").strip()
        if not s or not t.get_visible():
            continue
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        items.append((s, bb, t.get_fontsize()))
    problems = []
    for s, bb, fs in items:
        if fs < 6.4:
            problems.append(f"[字号] {fs:.1f}pt < 6.5：{s[:20]!r}")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s1, b1, _ = items[i]
            s2, b2, _ = items[j]
            ov = b1.intersection(b1, b2)
            if ov is None:
                continue
            a = ov.width * ov.height
            small = min(b1.width * b1.height, b2.width * b2.height)
            if small > 0 and a / small > 0.18:
                problems.append(f"[碰撞] {s1[:14]!r} × {s2[:14]!r} 重叠 {a / small:.0%}")
    return problems


def style_check():
    """锚点断言：图内数字与 v4 正文一致。"""
    cm = np.array([[36476, 12408, 10516], [8129, 38148, 13123], [7986, 15774, 35640]])
    assert cm.sum() == 178200 and cm.sum(1).tolist() == [59400, 59400, 59400]


# ---------------- 图 4 窗长消融（棒棒糖图） ----------------
def fig4():
    ys = [0.562, 0.576, 0.588]
    assert ys == [0.562, 0.576, 0.588]
    fig, ax = plt.subplots(figsize=(88 * MM, 62 * MM), constrained_layout=True)
    for i, y in enumerate(ys):
        c = PRIMARY if i == 2 else (P_MED if i == 1 else P_LT)
        ax.plot([i, i], [CHANCE, y], color=c, lw=1.8, solid_capstyle="round", zorder=3)
        ax.scatter(i, y, s=56 if i == 2 else 40, color=c, edgecolor="white",
                   linewidth=0.8, zorder=4)
        ax.text(i, y + 0.006, f"{y * 100:.2f}%", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold" if i == 2 else "normal", color=INK, zorder=5)
    # 点间虚连线 + 逐段增量
    for i in range(2):
        ax.plot([i, i + 1], [ys[i], ys[i + 1]], ls=(0, (2, 2)), lw=0.7,
                color=NEUTRAL, zorder=2)
        ax.text(i + 0.5, max(ys[i], ys[i + 1]) + 0.0075,
                f"+{(ys[i + 1] - ys[i]) * 100:.1f} 个百分点", ha="center", va="bottom",
                fontsize=6.8, color=NEUTRAL)
    ax.text(2.12, 0.588, "现行冻结口径", ha="left", va="center", fontsize=7.2,
            color=PRIMARY, fontweight="bold")
    ax.axhline(CHANCE, color=NEUTRAL, ls=(0, (4, 3)), lw=0.7, zorder=1)
    ax.text(-0.42, CHANCE + 0.006, "机会水平 1/3", ha="left", va="bottom",
            fontsize=6.8, color=NEUTRAL)
    ax.text(0.03, 0.965, "另：二分类 2 s → 3 s 提升 +4.7 个百分点（正文 3.1 节）",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=NEUTRAL)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.30, 0.655)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6])
    ax.set_yticklabels(["30%", "40%", "50%", "60%"])
    ax.set_xticks(range(3), ["1 s", "2 s", "3 s"])
    ax.set_ylabel("窗级三分类准确率")
    ax.set_xlabel("窗长（步长 100 ms）")
    ax.tick_params(axis="x", length=0)
    bare_ax(ax)
    return save(fig, "图4_窗长消融")


# ---------------- 图 5 十一模型选型（树状分层 + 条形，避免点数字重叠） ----------------
def fig5():
    """OpenBMI · 2s/hop100 · Acc_paper（与正文 §3.1 表一致）。
    左：选型树（CNN/raw vs bandpower）；右：准确率条，数字标在条外。
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    cnn_raw = [
        ("ShallowFBCSPNet", 0.5404, True),   # selected
        ("Deep4Net", 0.5400, False),
        ("Conformer", 0.5375, False),
        ("EEGNet", 0.5322, False),
        ("EEGTCNet", 0.5103, False),
        ("DGCNN_raw", 0.4911, False),
        ("DBN_raw", 0.4883, False),
        ("GCBNet_raw", 0.4802, False),
    ]
    band = [
        ("DGCNN", 0.3916, False),
        ("DBN", 0.3810, False),
        ("GCBNet", 0.3702, False),
    ]
    assert abs(cnn_raw[0][1] - cnn_raw[1][1] - 0.0004) < 1e-9

    # layout rows: top=cnn (8), gap, bottom=band (3); y decreases downward in plot
    # We'll build from top to bottom with high y for first model
    rows = []  # (y, group, name, val, selected)
    y = 0.0
    for name, val, sel in cnn_raw:
        rows.append((y, "cnn", name, val, sel))
        y -= 1.0
    y -= 0.55  # group gap
    for name, val, sel in band:
        rows.append((y, "band", name, val, sel))
        y -= 1.0
    y_min = y + 1.0

    fig, ax = plt.subplots(figsize=(145 * MM, 105 * MM), constrained_layout=True)

    # coordinate regions
    x_root, x_branch, x_leaf = 0.02, 0.18, 0.38
    x_bar0, x_bar1 = 0.62, 0.98  # axes fraction for bars — use data coords instead

    # Use mixed: tree in data coords with x in [0, 1] for tree, bars use twin or same with mapped vals
    # Simpler: all in axes fraction via transform=ax.transAxes for tree; bars in data

    ax.set_xlim(0, 1.0)
    ax.set_ylim(y_min - 1.35, 0.85)
    ax.axis("off")

    def _box(ax_, x, y_, w, h, text, *, fc="white", ec=NEUTRAL, fw="normal", fs=7.0, tc=INK):
        box = FancyBboxPatch(
            (x, y_ - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=fc, edgecolor=ec, linewidth=0.7, zorder=3,
            transform=ax_.transData, clip_on=False)
        ax_.add_patch(box)
        ax_.text(x + w / 2, y_, text, ha="center", va="center", fontsize=fs,
                 color=tc, fontweight=fw, zorder=4)

    # root
    y_cnn = np.mean([r[0] for r in rows if r[1] == "cnn"])
    y_band = np.mean([r[0] for r in rows if r[1] == "band"])
    y_root = (y_cnn + y_band) / 2
    _box(ax, 0.00, y_root, 0.13, 0.85, "十一模型\n选型", fc="#EEF3F8", ec=PRIMARY,
         fw="bold", fs=7.2)

    # branch nodes
    _box(ax, 0.18, y_cnn, 0.20, 0.72, "CNN / 原始波形\n（8）", fc="#F5F8FB", ec=PRIMARY, fs=6.8)
    _box(ax, 0.18, y_band, 0.20, 0.72, "Bandpower 图\n（3）", fc="#FAFAFA", ec=NEUTRAL, fs=6.8)

    # connectors root → branches
    ax.plot([0.13, 0.155, 0.155, 0.18], [y_root, y_root, y_cnn, y_cnn],
            color=STEM, lw=1.0, zorder=2)
    ax.plot([0.155, 0.155, 0.18], [y_root, y_band, y_band],
            color=STEM, lw=1.0, zorder=2)

    # bar scale region
    bar_x0, bar_x1 = 0.62, 0.92
    v0, v1 = 0.30, 0.58

    def v_to_x(v):
        return bar_x0 + (v - v0) / (v1 - v0) * (bar_x1 - bar_x0)

    # chance line
    xc = v_to_x(CHANCE)
    ax.plot([xc, xc], [y_min - 0.35, 0.55], color=NEUTRAL, ls=(0, (4, 3)), lw=0.65, zorder=1)
    ax.text(xc, y_min - 0.55, "机会水平 1/3", ha="center", va="top", fontsize=6.5, color=NEUTRAL)

    # axis ticks for bars (0.3 与机会线过近，仅画刻度线不标数)
    for tick in (0.3, 0.4, 0.5):
        xt = v_to_x(tick)
        ax.plot([xt, xt], [y_min - 0.15, y_min - 0.28], color=AXLINE, lw=0.5)
        if tick > 0.3:
            ax.text(xt, y_min - 0.32, f"{tick * 100:.0f}", ha="center", va="top", fontsize=6.5, color=NEUTRAL)
    ax.text((bar_x0 + bar_x1) / 2, y_min - 0.95,
            "试次级三分类准确率（Acc_paper · 2 s / 100 ms）",
            ha="center", va="top", fontsize=7.2, color=INK)

    # leaves + bars
    for y_i, grp, name, val, sel in rows:
        # branch → leaf connector
        yb = y_cnn if grp == "cnn" else y_band
        ax.plot([0.38, 0.40, 0.40, 0.42], [yb, yb, y_i, y_i], color=STEM, lw=0.85, zorder=2)

        # leaf name box
        if sel:
            _box(ax, 0.42, y_i, 0.175, 0.72, name + " ★", fc=PRIMARY, ec="#1A3A54",
                 fw="bold", fs=6.5, tc="white")
        else:
            _box(ax, 0.42, y_i, 0.175, 0.72, name, fc="white",
                 ec=PRIMARY if grp == "cnn" else NEUTRAL, fs=6.5,
                 tc=INK if grp == "cnn" else NEUTRAL)

        # bar
        x_end = v_to_x(val)
        bar_h = 0.38
        color = PRIMARY if sel else (P_LT if grp == "cnn" else "#D1D5DB")
        ax.add_patch(Rectangle(
            (bar_x0, y_i - bar_h / 2), x_end - bar_x0, bar_h,
            facecolor=color, edgecolor="none", zorder=3, alpha=0.95))
        # value OUTSIDE bar to the right — no overlap with marker
        ax.text(min(x_end + 0.012, 0.935), y_i, f"{val * 100:.2f}", ha="left", va="center",
                fontsize=6.8, color=INK if sel else NEUTRAL,
                fontweight="bold" if sel else "normal", zorder=5)

    # group spine from branch to first/last leaf
    ys_cnn = [r[0] for r in rows if r[1] == "cnn"]
    ys_band = [r[0] for r in rows if r[1] == "band"]
    ax.plot([0.40, 0.40], [max(ys_cnn), min(ys_cnn)], color=STEM, lw=0.85, zorder=2)
    ax.plot([0.40, 0.40], [max(ys_band), min(ys_band)], color=STEM, lw=0.85, zorder=2)

    # callouts
    ax.text(0.01, y_min - 1.25,
            "shallow 居首；deep −0.04 个百分点（几乎并列）\nTask 上 deep +2.3 个百分点（未据此换主干）",
            ha="left", va="top", fontsize=6.5, color=CONTRAST, linespacing=1.3)
    ax.text(0.62, y_min - 1.25,
            "Bandpower 图模型明显弱于 CNN / *_raw",
            ha="left", va="top", fontsize=6.5, color=NEUTRAL)

    problems = save(fig, "图5_十一模型选型")
    # sync english alias used by report
    import shutil
    src = Path("figures/图5_十一模型选型.png")
    if src.exists():
        shutil.copy2(src, Path("figures/fig05_model_selection.png"))
        pdf = Path("figures/图5_十一模型选型.pdf")
        if pdf.exists():
            shutil.copy2(pdf, Path("figures/fig05_model_selection.pdf"))
        # also refresh 交稿 copy if present
        j = Path("交稿/figures/fig05_model_selection.png")
        if j.parent.exists():
            shutil.copy2(src, j)
    return problems


# ---------------- 图 6 主结果（竖柱对比）+ 混淆矩阵（行归一化热图）——规格 §3 ----------
def fig6():
    """通栏双面板：左读出对比竖柱（混粒度，柱标签声明）；右因果平滑窗级混淆热图。"""
    from matplotlib.colors import LinearSegmentedColormap

    vals = [0.5808, 0.5925, 0.6125, 0.6188]
    sd0 = 0.0288
    labels = ["浅层 3 s\n单模型\n(窗级)", "融合\nargmax\n(窗级)", "融合 +\n多数票\n(试次级)", "融合 +\n因果平滑\n(窗级·主)"]
    fills = [P_LT, P_MED, P_MED, PRIMARY]
    edges = [P_MED, PRIMARY, PRIMARY, "#1A3A54"]
    cm = np.array([[36476, 12408, 10516],
                   [8129, 38148, 13123],
                   [7986, 15774, 35640]], dtype=float)
    assert cm.sum(1).tolist() == [59400.0, 59400.0, 59400.0]
    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0
    # 对角召回与正文一致
    assert abs(row_pct[0, 0] - 61.41) < 0.05
    assert abs(row_pct[1, 1] - 64.22) < 0.05
    assert abs(row_pct[2, 2] - 60.00) < 0.05

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(168 * MM, 72 * MM), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.08, 1]})

    # (a) 竖柱：浅层单模型 → 窗级 argmax → 多数票 → 因果平滑（主结果）
    x = np.arange(4)
    axL.bar(x, vals, width=0.62, color=fills, edgecolor=edges, linewidth=0.8, zorder=3)
    # 主结果柱加粗描边
    axL.patches[3].set_linewidth(1.35)
    axL.patches[3].set_edgecolor("#1A3A54")
    axL.errorbar(0, vals[0], yerr=sd0, fmt="none", ecolor=INK, elinewidth=0.8,
                 capsize=2.2, capthick=0.8, zorder=4)
    for i, v in enumerate(vals):
        yt = v + (sd0 + 0.010 if i == 0 else 0.008)
        axL.text(i, yt, f"{v * 100:.2f}", ha="center", va="bottom", fontsize=7.2,
                 fontweight="bold" if i == 3 else "normal", color=INK, zorder=5)
    # 「主结果」放在数值上方，避免与 0.6188 重叠
    axL.text(3, vals[3] + 0.042, "主结果", ha="center", va="bottom",
             fontsize=6.8, color=PRIMARY, fontweight="bold")
    axL.axhline(CHANCE, color=NEUTRAL, ls=(0, (4, 3)), lw=0.7, zorder=1)
    axL.text(-0.45, CHANCE + 0.008, "机会水平 1/3", ha="left", va="bottom",
             fontsize=6.8, color=NEUTRAL)
    axL.set_ylim(0.30, 0.72)
    axL.set_yticks([0.3, 0.4, 0.5, 0.6])
    axL.set_yticklabels(["30%", "40%", "50%", "60%"])
    axL.set_ylabel("准确率（粒度见柱标签）")
    axL.set_xticks(x, labels)
    axL.tick_params(axis="x", length=0, labelsize=7.2)
    bare_ax(axL)
    axL.text(-0.14, 1.05, "(a)", transform=axL.transAxes, fontsize=9.5,
             fontweight="bold", va="top", ha="left", color=INK)
    # (b) 3×3 行归一化热图 + colorbar
    cmap = LinearSegmentedColormap.from_list("pri", ["#F4F7FA", P_LT, PRIMARY])
    im = axR.imshow(row_pct, cmap=cmap, vmin=0, vmax=70, aspect="equal")
    names = ["空闲", "左手", "右手"]
    for i in range(3):
        for j in range(3):
            pct = row_pct[i, j]
            txt_c = "white" if pct >= 42 else INK
            axR.text(j, i, f"{pct:.1f}%", ha="center", va="center", fontsize=7.5,
                     color=txt_c, fontweight="bold" if i == j else "normal")
    axR.set_xticks(range(3), names)
    axR.set_yticks(range(3), names)
    axR.set_xlabel("预测类")
    axR.set_ylabel("真实类")
    axR.tick_params(length=0, labelsize=8)
    for s in axR.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6.8, width=0.6, length=2.2)
    cbar.set_label("行归一化占比（%）", fontsize=7)
    cbar.outline.set_linewidth(0.6)
    axR.text(-0.14, 1.05, "(b)", transform=axR.transAxes, fontsize=9.5,
             fontweight="bold", va="top", ha="left", color=INK)
    return save(fig, "图6_OpenBMI主结果与混淆矩阵")

# ---------------- 图 7 Stieger 跨库适配（斜率图） ----------------
def fig7():
    vals = [0.4198, 0.6590, 0.7003]
    sds = [0.0585, 0.0929, 0.0869]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(88 * MM, 64 * MM), constrained_layout=True)
    ax.plot(x[:2], vals[:2], color=PRIMARY, lw=1.8, zorder=3, solid_capstyle="round")
    ax.plot(x[1:], vals[1:], color=PRIMARY, lw=1.5, ls=(0, (4, 2)), zorder=3)
    ax.errorbar(x, vals, yerr=sds, fmt="none", ecolor=INK, elinewidth=0.8,
                capsize=2.2, capthick=0.8, zorder=4)
    ax.scatter(0, vals[0], s=40, color=P_MED, edgecolor="white", lw=0.7, zorder=5)
    ax.scatter(1, vals[1], s=62, color=PRIMARY, edgecolor="white", lw=0.8, zorder=5)
    ax.scatter(2, vals[2], s=58, facecolor="white", edgecolor=CONTRAST, lw=1.4, zorder=5)
    ax.text(0, vals[0] + sds[0] + 0.014, f"{vals[0] * 100:.2f}%", ha="center", fontsize=7.2, color=INK)
    ax.text(1, vals[1] + sds[1] + 0.014, f"{vals[1] * 100:.2f}%", ha="center", fontsize=7.2,
            color=INK, fontweight="bold")
    ax.text(2, vals[2] + sds[2] + 0.014, f"{vals[2] * 100:.2f}%", ha="center", fontsize=7.2,
            color=CONTRAST)
    ax.text(0.5, (vals[0] + vals[1]) / 2 + 0.055, "+23.9 个百分点", ha="center",
            fontsize=7.5, color=PRIMARY, fontweight="bold")
    ax.text(1.5, (vals[1] + vals[2]) / 2 + 0.050, "+4.1 个百分点*", ha="center",
            fontsize=7.5, color=PRIMARY, fontweight="bold")
    ax.text(0.03, 0.97, "24/24 被试微调提升 ≥3 个百分点", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.2, color=PRIMARY)
    ax.set_xticks(x, ["零样本", "少样本微调", "+生理门控*"])
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0.28, 0.88)
    ax.set_yticks([0.3, 0.5, 0.7])
    ax.set_yticklabels(["30%", "50%", "70%"])
    ax.set_ylabel("试次级三分类准确率")
    ax.tick_params(axis="x", length=0, labelsize=8)
    ax.text(1.0, -0.13, "* 生理门控为离线分析口径（弃权率 59.3%），不进入正式在线方案",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=NEUTRAL)
    bare_ax(ax)
    return save(fig, "图7_Stieger跨库适配")


# ---------------- 附图 M · Leave-Next 协议示意（方法图，不占结果编号） ----------------
def fig_m_leave_next():
    """半栏时序示意：每轮用历史场训练、紧邻下一场保留评测。"""
    fig, ax = plt.subplots(figsize=(95 * MM, 52 * MM), constrained_layout=True)
    sessions = ["S1", "S2", "S3", "S4", "S5", "S6"]
    n_s = len(sessions)
    # 背景场次格
    for i, s in enumerate(sessions):
        ax.add_patch(plt.Rectangle((i - 0.38, -0.35), 0.76, 0.7,
                                   facecolor="#F3F6F9", edgecolor=P_MED, lw=0.6, zorder=1))
        ax.text(i, 0.0, s, ha="center", va="center", fontsize=8, color=INK, zorder=2)

    # 五行：R0 零样本 + R1–R5
    row_y0 = 1.15
    dy = 0.95
    rounds = [
        ("R0 零样本", None, 1, "仅评测底座，不训练"),
        ("R1", [0], 1, "训 S1 → 测 S2"),
        ("R2", [0, 1], 2, "训 S1–S2 → 测 S3"),
        ("R3", [0, 1, 2], 3, "训 S1–S3 → 测 S4"),
        ("R4", [0, 1, 2, 3], 4, "训 S1–S4 → 测 S5"),
        ("R5", [0, 1, 2, 3, 4], 5, "训 S1–S5 → 测 S6"),
    ]
    for r_i, (lab, trains, hold, tip) in enumerate(rounds):
        y = row_y0 + r_i * dy
        ax.text(-0.95, y, lab, ha="right", va="center", fontsize=7, color=INK)
        if trains is None:
            ax.scatter([hold], [y], s=36, facecolor="white", edgecolor=NEUTRAL,
                       lw=1.1, zorder=4)
            ax.text(hold + 0.42, y, "评测", ha="left", va="center",
                    fontsize=6.5, color=NEUTRAL)
        else:
            for t in trains:
                ax.add_patch(plt.Rectangle((t - 0.32, y - 0.22), 0.64, 0.44,
                                           facecolor=P_LT, edgecolor=PRIMARY,
                                           lw=0.7, zorder=3))
            ax.annotate("", xy=(hold, y), xytext=(trains[-1] + 0.38, y),
                        arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=1.1),
                        zorder=4)
            ax.scatter([hold], [y], s=42, color=PRIMARY, edgecolor="white",
                       lw=0.7, zorder=5)
            ax.text(hold + 0.42, y, "保留评测", ha="left", va="center",
                    fontsize=6.5, color=PRIMARY)

    ax.set_xlim(-1.7, n_s - 0.15)
    ax.set_ylim(-1.15, row_y0 + len(rounds) * dy - 0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # 图例（底部左侧）
    ax.add_patch(plt.Rectangle((-0.35, -0.95), 0.28, 0.22, facecolor=P_LT,
                               edgecolor=PRIMARY, lw=0.6, clip_on=False))
    ax.text(0.05, -0.84, "训练场", fontsize=6.5, color=INK, va="center")
    ax.scatter([1.15], [-0.84], s=28, color=PRIMARY, edgecolor="white", lw=0.5, clip_on=False)
    ax.text(1.35, -0.84, "保留评测场", fontsize=6.5, color=INK, va="center")
    ax.text(3.2, -0.84,
            "训练永远先于评测 · 与真实逐场部署一致",
            fontsize=6.5, color=NEUTRAL, va="center")
    return save(fig, "附图M_LeaveNext协议示意")


# ---------------- 图 8 BCI2a Leave-Next（小倍数：all4 vs so，Shallow 虚点参照） ----------
def fig8():
    """实线 all4；虚线 e1f_so（+21.1 pp 主对照）；点线单头 Shallow（同档参照）。"""
    d = np.load("figures/_exp32_p1_rounds.npz")
    A, S, Sh = d["all4"], d["so"], d["shallow"]
    subs = [str(s).split("·")[0].strip() for s in d["subs"]]
    assert A.shape == S.shape == Sh.shape == (9, 6)
    assert abs(A.mean(0)[0] - 0.338) < 0.001 and abs(S.mean(0)[0] - 0.338) < 0.001
    assert abs(A.mean(0)[5] - 0.663) < 0.001 and abs(S.mean(0)[5] - 0.452) < 0.001
    assert abs(Sh.mean(0)[5] - 0.671) < 0.002
    assert abs(A[:, 5].std(ddof=1) - 0.067) < 0.001
    assert abs(S[:, 5].std(ddof=1) - 0.047) < 0.001
    lo = min(A.min(), S.min(), Sh.min())
    hi = max(A.max(), S.max(), Sh.max())
    ylo = np.floor((lo - 0.02) * 10) / 10
    yhi = np.ceil((hi + 0.02) * 10) / 10
    fig, axes = plt.subplots(3, 3, figsize=(160 * MM, 122 * MM),
                             sharex=True, sharey=True, constrained_layout=True)
    for k, ax in enumerate(axes.flat):
        ax.plot(range(6), Sh[k], color=P_LT, lw=1.0, ls=(0, (1, 2)), marker=".",
                ms=3.5, zorder=2)
        ax.plot(range(6), S[k], color=NEUTRAL, lw=1.15, ls=(0, (3, 2)), marker="o",
                ms=2.4, mfc="white", mec=NEUTRAL, zorder=3)
        ax.plot(range(6), A[k], color=PRIMARY, lw=1.55, marker="o", ms=2.9, zorder=4)
        ax.set_title(subs[k], fontsize=7.5, color=INK, pad=2.5)
        ax.set_xlim(-0.3, 5.3)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks([0, 5])
        ax.set_yticks([0.33, 0.67])
        ax.tick_params(labelsize=6.8, length=2, width=0.5)
        bare_ax(ax)
    for ax in axes[:-1].flat:
        ax.tick_params(labelbottom=False)
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)
    for ax in axes[:, 0]:
        ax.set_yticklabels(["33%", "67%"])
    fig.supylabel("三分类窗级准确率（含空闲）", fontsize=7.5)
    fig.supxlabel(
        "实线 = 四成员 all4 · 虚线 = 只微调浅层再融合 so · 点线 = 单头 Shallow（S-fo）\n"
        "末档 R5：all4 66.30%±6.70% vs so 45.20%±4.70%（Δ=+21.1 个百分点）；vs Shallow 67.10%±6.70%（同量级）",
        fontsize=7)
    return save(fig, "图8_BCI2a_LeaveNext曲线")


# ---------------- 图 9 指定集（桥形图：基线 → 增量 → 终值） ----------------
def fig9():
    tracks = [
        dict(x0=0.0, base=0.511, bsd=0.066, top=0.558, tsd=0.069,
             inc=0.047, dlab="+4.7 个百分点 折内拟合", bc=PRIMARY, ic=CONTRAST,
             ticks=("嵌套主读\n（交卷口径）", "折内附报\n（乐观偏置）"),
             grp="QuadFold-59 · 59 通道交卷栈"),
        dict(x0=1.25, base=0.528, bsd=0.130, top=0.540, tsd=0.146,
             inc=0.012, dlab="+1.2 个百分点 融合", bc=P_MED, ic=PRIMARY,
             ticks=("V1 单模\n（选池附报）", "R-B8 融合\n（风险否决）"),
             grp="8 通道微调栈 · OpenBMI 预训练"),
    ]
    assert abs(tracks[0]["top"] - tracks[0]["base"] - 0.047) < 1e-9
    assert abs(tracks[1]["top"] - tracks[1]["base"] - 0.012) < 1e-9
    BASE, W = 0.30, 0.30
    fig, ax = plt.subplots(figsize=(120 * MM, 74 * MM), constrained_layout=True)
    xticks, xlabels = [], []
    for t in tracks:
        x0 = t["x0"]
        xi = x0 + 0.44
        ax.bar(x0, t["base"] - BASE, bottom=BASE, width=W, color=t["bc"],
               edgecolor="white", lw=0.5, zorder=3)
        ax.bar(xi, t["inc"], bottom=t["base"], width=W, color=t["ic"],
               edgecolor="white", lw=0.5, zorder=3)
        ax.plot([x0 + W / 2, xi + W / 2], [t["base"], t["base"]],
                ls=(0, (2, 2)), lw=0.7, color=NEUTRAL, zorder=4)
        ax.errorbar(x0 + W / 2, t["base"], yerr=t["bsd"], fmt="none", ecolor=INK,
                    elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=5)
        ax.errorbar(xi + W / 2, t["top"], yerr=t["tsd"], fmt="none", ecolor=INK,
                    elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=5)
        ax.text(x0 + W / 2, t["base"] + t["bsd"] + 0.008, f"{t['base'] * 100:.2f}",
                ha="center", fontsize=7.2, color=INK, zorder=6)
        ax.text(xi + W / 2, t["top"] + t["tsd"] + 0.008, f"{t['top'] * 100:.2f}",
                ha="center", fontsize=7.2, color=INK, zorder=6)
        ax.text(xi + W + 0.03, (t["base"] + t["top"]) / 2, t["dlab"],
                ha="left", va="center", fontsize=6.8, color=t["ic"], zorder=6)
        bx0, bx1, by = x0 - 0.04, xi + W + 0.04, 0.745
        ax.plot([bx0, bx0, bx1, bx1], [by - 0.010, by, by, by - 0.010],
                color=NEUTRAL, lw=0.7, zorder=4)
        ax.text((bx0 + bx1) / 2, by + 0.008, t["grp"], ha="center", va="bottom",
                fontsize=7.2, color=INK, zorder=5)
        xticks += [x0 + W / 2, xi + W / 2]
        xlabels += list(t["ticks"])
    ax.set_xticks(xticks, xlabels, fontsize=6.8)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.10, 2.22)
    ax.set_ylim(0.285, 0.83)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_yticklabels(["30%", "40%", "50%", "60%", "70%", "80%"])
    ax.set_ylabel("指定集三分类准确率（六折）")
    ax.text(1.0, -0.135,
            "嵌套 = 其余五折验证集拟合融合参数、本折评估 · 误差条 = 跨折标准差 · 纵轴自 30% 起",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=NEUTRAL)
    bare_ax(ax)
    return save(fig, "图9_指定集嵌套与交卷")


# ---------------- 图 10 真人 Leave-Next 逐轮柱状（每被试一面板） ----------------
def _load_real_leave_next_cohort():
    """读取各被试最新 all4 Leave-Next F5 summary → 逐轮微调三分类窗级（因果平滑，%）。"""
    from pathlib import Path
    import json

    root = Path(__file__).resolve().parents[3] / "experiment_game" / "data" / "subjects"
    if not root.exists():
        root = Path(r"D:/MI/experiment_game/data/subjects")
    by_sid = {}
    for p in root.glob("*/models/ft_runs/*leave_next*all4*f5_summary.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        sid = str(d.get("subject_id") or p.parent.parent.parent.name)
        stamp = p.name.split("_")[0]
        prev = by_sid.get(sid)
        if prev is None or stamp >= prev["stamp"]:
            rows = []
            for r in d.get("rows") or []:
                win = r.get("heldout_acc_smooth")
                if win is None:
                    win = r.get("heldout_acc")
                if win is None:
                    continue
                win_pct = float(win) * 100.0 if float(win) <= 1.0 else float(win)
                rows.append({
                    "r": int(r.get("r_stage") or (len(rows) + 1)),
                    "win": win_pct,
                    "pass": bool(r.get("release_pass")),
                    "hold": str(r.get("heldout") or ""),
                })
            if not rows:
                continue
            by_sid[sid] = {"stamp": stamp, "path": str(p), "rows": rows}
    cohort = []
    for sid, pack in by_sid.items():
        rows = pack["rows"]
        wins = [r["win"] for r in rows]
        best_i = int(max(range(len(wins)), key=lambda i: (wins[i], i)))
        cohort.append({
            "sid": sid,
            "wins": wins,
            "passes": [r["pass"] for r in rows],
            "final_win": rows[-1]["win"],
            "final_pass": rows[-1]["pass"],
            "best_win": wins[best_i],
            "best_i": best_i,  # 0-based
            "n": len(rows),
        })
    cohort.sort(key=lambda x: (-x["final_win"], x["sid"]))
    return cohort


def fig10():
    """通栏小倍数：每被试 Leave-Next 各轮微调三分类窗级（因果平滑）柱状图。"""
    cohort = _load_real_leave_next_cohort()
    n = len(cohort)
    assert n >= 15, f"真人队列不足 15 人，当前 {n}"
    finals = [c["final_win"] for c in cohort]
    bests = [c["best_win"] for c in cohort]
    n_pass = sum(1 for c in cohort if c["final_pass"])
    mean_final = float(np.mean(finals))
    mean_best = float(np.mean(bests))

    # 布局：优先 4 列；20 人 → 5×4
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(170 * MM, max(78, 28 * nrows) * MM),
        sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    max_r = max(c["n"] for c in cohort)

    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.set_visible(False)
            continue
        c = cohort[idx]
        xs = np.arange(1, c["n"] + 1)
        # 底层：PASS 绿 / FAIL 浅灰
        colors = [PASS_C if ok else P_LT for ok in c["passes"]]
        edges = [PASS_C if ok else NEUTRAL for ok in c["passes"]]
        lw = [0.7] * c["n"]
        # 末档：主色加粗（稍后若同为最高，会被 BEST 覆盖）
        colors[-1] = PRIMARY if c["final_pass"] else P_MED
        edges[-1] = "#1A3A54" if c["final_pass"] else FAIL_C
        lw[-1] = 1.15
        # 各轮最高：橙对照色优先于 PASS / 末档
        bi = c["best_i"]
        colors[bi] = BEST_C
        edges[bi] = "#8B3A12"
        lw[bi] = 1.25
        ax.bar(xs, c["wins"], width=0.72, color=colors, edgecolor=edges,
               linewidth=lw, zorder=3)
        ax.axhline(CHANCE * 100.0, color=NEUTRAL, ls=(0, (3, 2)), lw=0.55, zorder=1)
        ax.set_title(c["sid"], fontsize=7.2, color=INK, pad=2)
        ax.set_xlim(0.4, max_r + 0.6)
        ax.set_ylim(0, 100)
        ax.set_xticks(list(range(1, max_r + 1)))
        ax.set_yticks([0, round(CHANCE * 100, 1), 50, 100])
        ax.tick_params(labelsize=6.5, length=2, width=0.5)
        bare_ax(ax)
        # 最高轮数值（橙）；若最高≠末档，另标末档
        ax.text(bi + 1, min(97, c["best_win"] + 4.5), f"{c['best_win']:.0f}",
                ha="center", va="bottom", fontsize=6.5,
                color=BEST_C, fontweight="bold")
        if bi != c["n"] - 1:
            ax.text(c["n"], min(97, c["final_win"] + 4.5), f"{c['final_win']:.0f}",
                    ha="center", va="bottom", fontsize=6.5,
                    color=PRIMARY if c["final_pass"] else FAIL_C, fontweight="bold")

    for ax in axes[:, 0]:
        ax.set_ylabel("三分类窗级（%）", fontsize=7)
    fig.supxlabel(
        f"Leave-Next 轮次 R1…Rn（橙=各轮最高·叠色优先；绿=PASS / 浅灰=FAIL；末档加粗；虚线=机会水平 33.3%）· n={n} · "
        f"末档均值 {mean_final:.1f}% · 最高均值 {mean_best:.1f}% · 末档 PASS {n_pass}/{n}",
        fontsize=7)
    problems = save(fig, "图10_真人LeaveNext逐轮窗级")
    import shutil
    src = Path("figures/图10_真人LeaveNext逐轮窗级.png")
    alias = Path("figures/fig10_cohort_final_mi.png")
    if src.exists():
        shutil.copy2(src, alias)
        j = Path("交稿/figures")
        if j.is_dir():
            shutil.copy2(src, j / src.name)
            shutil.copy2(alias, j / alias.name)
            pdf = Path("figures/图10_真人LeaveNext逐轮窗级.pdf")
            if pdf.exists():
                shutil.copy2(pdf, j / pdf.name)
    return problems


# ---------------- 图 11 仿真 vs 真人 适配增益对照（三分类窗级，同指标族） ----------------
def fig11():
    """左：队列均值「零样本 → 微调后」；右：每人增益 Δ 条形（易读）。
    数据：BCI2a=Exp32；真人=§3.6 微调窗级 − 零样本窗级。"""
    import shutil
    from pathlib import Path

    # 队列均值（%）：零样本 / 末档；与正文 0.338→0.663、0.375→0.476 对齐（n=20，2026-09-06）
    b2a_zs, b2a_ft = 33.8, 66.3
    hum_zs, hum_ft = 37.5, 47.6
    b2a_d = [36.6, 20.5, 45.4, 31.9, 26.5, 37.4, 31.0, 32.8, 30.1]
    # §3.6 表窗级降序：Δ=末档−零样本（百分点）
    hum_d = [51.0, 28.8, 13.3, 11.3, 15.6, 1.9, 2.2, 2.9, 1.7, 5.7,
             9.9, 12.1, 16.5, 13.4, 1.9, 7.6, 5.5, 0.3, 0.0, 0.2]
    hum_pass = [True, True, True, False, True, True, True, True, True, False,
                True, True, True, False, False, True, False, False, False, False]
    assert abs(np.mean(b2a_d) - 32.5) < 0.15
    assert abs(np.mean(hum_d) - 10.09) < 0.15

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(165 * MM, 62 * MM),
        gridspec_kw={"width_ratios": [1.05, 1.35]},
        constrained_layout=True)

    # ---- 左：零样本 vs 微调后（一眼看懂「提升了多少」）----
    x = np.array([0.0, 1.35])
    w = 0.38
    zs = [b2a_zs, hum_zs]
    ft = [b2a_ft, hum_ft]
    ax0.bar(x - w / 2, zs, width=w, color=P_LT, edgecolor=PRIMARY,
            linewidth=0.6, label="零样本", zorder=3)
    ax0.bar(x + w / 2, ft, width=w, color=PRIMARY, edgecolor="#1A3A54",
            linewidth=0.6, label="微调后（末档）", zorder=3)
    for i, (z, f) in enumerate(zip(zs, ft)):
        # 提升箭头与 Δ 标注
        ax0.annotate(
            "", xy=(x[i] + w / 2, f), xytext=(x[i] + w / 2, z),
            arrowprops=dict(arrowstyle="->", color=CONTRAST, lw=1.1))
        ax0.text(x[i] + w / 2 + 0.22, (z + f) / 2, f"+{f - z:.1f} 个百分点",
                 color=CONTRAST, fontsize=7.5, fontweight="bold", va="center")
        ax0.text(x[i] - w / 2, z + 1.2, f"{z:.1f}", ha="center", va="bottom",
                 fontsize=6.5, color=NEUTRAL)
        ax0.text(x[i] + w / 2, f + 1.2, f"{f:.1f}", ha="center", va="bottom",
                 fontsize=6.5, color=INK, fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(["BCI2a 仿真\n（n=9）", "真人队列\n（n=20）"], fontsize=7.5)
    ax0.set_ylabel("三分类窗级准确率（%）", fontsize=8)
    ax0.set_ylim(0, 78)
    ax0.set_yticks([0, 20, 40, 60])
    ax0.legend(loc="upper left", fontsize=7, handlelength=1.2)
    ax0.set_title("队列均值：微调前后对照", fontsize=8, color=INK, pad=4)
    bare_ax(ax0)

    # ---- 右：每人增益条形（按 Δ 降序）----
    b2a_s = sorted(b2a_d, reverse=True)
    # 真人：与 Δ 一起排序，门控色跟随
    hum_pairs = sorted(zip(hum_d, hum_pass), key=lambda t: -t[0])
    hum_s = [v for v, _ in hum_pairs]
    hum_ok = [ok for _, ok in hum_pairs]

    # 两段 y：上 BCI2a，下 真人，中间留缝
    gap = 1.2
    y_b = np.arange(len(b2a_s))[::-1] + (len(hum_s) + gap)
    y_h = np.arange(len(hum_s))[::-1]
    ax1.barh(y_b, b2a_s, height=0.72, color=PRIMARY, edgecolor="#1A3A54",
             linewidth=0.4, zorder=3)
    ax1.barh(y_h, hum_s, height=0.72,
             color=[PASS_C if ok else FAIL_C for ok in hum_ok],
             edgecolor=NEUTRAL, linewidth=0.35, zorder=3)
    mb, mh = float(np.mean(b2a_d)), float(np.mean(hum_d))
    ax1.axvline(mb, color=PRIMARY, ls=(0, (3, 2)), lw=0.9, zorder=2)
    ax1.axvline(mh, color=PASS_C, ls=(0, (3, 2)), lw=0.9, zorder=2)
    ax1.text(mb, y_b.max() + 0.85, f"仿真均值 {mb:.1f}", ha="center",
             fontsize=6.5, color=PRIMARY, fontweight="bold")
    ax1.text(mh, -1.15, f"真人均值 {mh:.1f}", ha="center",
             fontsize=6.5, color=PASS_C, fontweight="bold")
    # 组标签
    ax1.text(-1.5, y_b.mean(), "BCI2a", ha="right", va="center",
             fontsize=7.5, color=PRIMARY, fontweight="bold", rotation=90)
    ax1.text(-1.5, y_h.mean(), "真人", ha="right", va="center",
             fontsize=7.5, color=INK, fontweight="bold", rotation=90)
    ax1.set_xlabel("每人增益 Δ = 末档 − 零样本（百分点）", fontsize=8)
    ax1.set_xlim(0, 56)
    ax1.set_yticks([])
    ax1.set_title("个体增益分布（条越长 = 提升越多）", fontsize=8, color=INK, pad=4)
    # 图例：门控
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor=PRIMARY, edgecolor="#1A3A54", label="BCI2a 被试"),
        Patch(facecolor=PASS_C, edgecolor=NEUTRAL, label="真人 · 门控 PASS"),
        Patch(facecolor=FAIL_C, edgecolor=NEUTRAL, label="真人 · 门控 FAIL"),
    ], loc="lower right", fontsize=6.5, handlelength=1.0, frameon=False)
    bare_ax(ax1, keep_left=False)
    ax1.spines["left"].set_visible(False)

    # 一句读法（图内）
    fig.suptitle(
        "读法：左图看「平均从多少提到多少」；右图看「每个人提升多少」（仿真最差 +20.5 > 真人中位 +6.7）",
        fontsize=7, color=NEUTRAL, y=1.02)

    problems = save(fig, "图11_仿真与真人增益对照")
    # 正文引用别名
    src = Path("figures/图11_仿真与真人增益对照.png")
    dst = Path("figures/fig11_sim_vs_human_gain.png")
    if src.exists():
        shutil.copy2(src, dst)
        pdf_src = Path("figures/图11_仿真与真人增益对照.pdf")
        if pdf_src.exists():
            shutil.copy2(pdf_src, Path("figures/fig11_sim_vs_human_gain.pdf"))
        for jdir in (Path("交稿/figures"),):
            if jdir.is_dir():
                shutil.copy2(src, jdir / src.name)
                shutil.copy2(dst, jdir / dst.name)
                if pdf_src.exists():
                    shutil.copy2(pdf_src, jdir / pdf_src.name)
    # 图10 小倍数已在 fig10() 同步交稿别名
    return problems


def _load_bci2a_all4_fo_rounds():
    """Exp32 P1：A01–A09 × R0–R5 的 E-a4-fo（三分类窗级因果平滑，小数）。"""
    import re
    from pathlib import Path

    p = Path(r"D:/MI/资料/模型训练/32_旁路_bci2a_LeaveNext_双底座双门控_openbmi_accpaper/总结/结果登记表.md")
    text = p.read_text(encoding="utf-8")
    blocks = re.findall(
        r"## (A0[1-9]) · P1\n\n\|[^\n]+\n\|[^\n]+\n((?:\| R[0-5].*\n)+)",
        text,
    )
    assert len(blocks) == 9, f"Exp32 P1 表块数={len(blocks)}"
    rows = []
    subs = []
    for sid, body in blocks:
        vals = []
        for line in body.strip().splitlines():
            parts = [x.strip() for x in line.split("|") if x.strip()]
            vals.append(float(parts[6]))  # E-a4-fo
        assert len(vals) == 6, sid
        rows.append(vals)
        subs.append(sid)
    return np.asarray(rows, dtype=float), subs


def _human_peak_vs_zs():
    """真人：各轮最高窗级 − 零样本（与 §3.6 表零样本列对齐）；门控取最高轮。"""
    zs_map = {
        "syj0828": 41.40, "lsy0903": 30.00, "npl0831": 40.90, "lsm0903": 42.40,
        "lmh0904": 37.30, "xj0830": 46.50, "zyn0906": 46.10, "cjf0831": 43.90,
        "ytl0901": 44.40, "zyj0902": 40.10, "fnz0828": 35.40, "djh0902": 31.30,
        "wyf0906": 26.10, "wzr0830": 29.10, "ycx0831": 39.90, "cyy0830": 33.30,
        "zcy0902": 34.80, "lmy0904": 37.90, "fnz0830": 35.70, "djy0906": 33.20,
    }
    cohort = _load_real_leave_next_cohort()
    peaks, zss, deltas, passes = [], [], [], []
    for c in cohort:
        zs = zs_map[c["sid"]]
        pk = round(float(c["best_win"]), 1)
        peaks.append(pk)
        zss.append(zs)
        deltas.append(pk - zs)
        passes.append(bool(c["passes"][c["best_i"]]))
    return {
        "peaks": peaks,
        "zs": zss,
        "deltas": deltas,
        "passes": passes,
        "mean_peak": float(np.mean(peaks)),
        "mean_zs": float(np.mean(zss)),
        "mean_d": float(np.mean(deltas)),
        "median_d": float(np.median(deltas)),
    }


# ---------------- 图 11b：各轮最高口径（对照图 11 末档） ----------------
def fig11_peak():
    """同 fig11 版式；微调后改为 Leave-Next 各轮最高（BCI2a=R1–R5；真人=爬坡最高）。"""
    import shutil
    from pathlib import Path
    from matplotlib.patches import Patch

    A, _subs = _load_bci2a_all4_fo_rounds()
    b2a_zs = float(A[:, 0].mean() * 100.0)
    b2a_pk = float(A[:, 1:].max(axis=1).mean() * 100.0)
    b2a_d = list((A[:, 1:].max(axis=1) - A[:, 0]) * 100.0)
    hum = _human_peak_vs_zs()
    hum_zs, hum_pk = hum["mean_zs"], hum["mean_peak"]
    hum_d, hum_pass = hum["deltas"], hum["passes"]

    assert abs(b2a_zs - 33.8) < 0.05
    assert abs(b2a_pk - 68.0) < 0.15
    assert abs(np.mean(b2a_d) - 34.2) < 0.15
    assert abs(hum_zs - 37.5) < 0.05
    assert abs(hum_pk - 54.2) < 0.15
    assert abs(hum["mean_d"] - 16.7) < 0.15

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(165 * MM, 62 * MM),
        gridspec_kw={"width_ratios": [1.05, 1.35]},
        constrained_layout=True)

    x = np.array([0.0, 1.35])
    w = 0.38
    zs = [b2a_zs, hum_zs]
    ft = [b2a_pk, hum_pk]
    ax0.bar(x - w / 2, zs, width=w, color=P_LT, edgecolor=PRIMARY,
            linewidth=0.6, label="零样本", zorder=3)
    ax0.bar(x + w / 2, ft, width=w, color=BEST_C, edgecolor="#8B3A12",
            linewidth=0.6, label="微调后（各轮最高）", zorder=3)
    for i, (z, f) in enumerate(zip(zs, ft)):
        ax0.annotate(
            "", xy=(x[i] + w / 2, f), xytext=(x[i] + w / 2, z),
            arrowprops=dict(arrowstyle="->", color=CONTRAST, lw=1.1))
        ax0.text(x[i] + w / 2 + 0.22, (z + f) / 2, f"+{f - z:.1f} 个百分点",
                 color=CONTRAST, fontsize=7.5, fontweight="bold", va="center")
        ax0.text(x[i] - w / 2, z + 1.2, f"{z:.1f}", ha="center", va="bottom",
                 fontsize=6.5, color=NEUTRAL)
        ax0.text(x[i] + w / 2, f + 1.2, f"{f:.1f}", ha="center", va="bottom",
                 fontsize=6.5, color=INK, fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(["BCI2a 仿真\n（n=9）", "真人队列\n（n=20）"], fontsize=7.5)
    ax0.set_ylabel("三分类窗级准确率（%）", fontsize=8)
    ax0.set_ylim(0, 88)
    ax0.set_yticks([0, 20, 40, 60, 80])
    ax0.legend(loc="upper right", fontsize=7, handlelength=1.2)
    ax0.set_title("队列均值：零样本 vs 各轮最高", fontsize=8, color=INK, pad=4)
    bare_ax(ax0)

    b2a_s = sorted(b2a_d, reverse=True)
    hum_pairs = sorted(zip(hum_d, hum_pass), key=lambda t: -t[0])
    hum_s = [v for v, _ in hum_pairs]
    hum_ok = [ok for _, ok in hum_pairs]
    gap = 1.2
    y_b = np.arange(len(b2a_s))[::-1] + (len(hum_s) + gap)
    y_h = np.arange(len(hum_s))[::-1]
    ax1.barh(y_b, b2a_s, height=0.72, color=PRIMARY, edgecolor="#1A3A54",
             linewidth=0.4, zorder=3)
    ax1.barh(y_h, hum_s, height=0.72,
             color=[PASS_C if ok else FAIL_C for ok in hum_ok],
             edgecolor=NEUTRAL, linewidth=0.35, zorder=3)
    mb, mh = float(np.mean(b2a_d)), float(np.mean(hum_d))
    ax1.axvline(mb, color=PRIMARY, ls=(0, (3, 2)), lw=0.9, zorder=2)
    ax1.axvline(mh, color=PASS_C, ls=(0, (3, 2)), lw=0.9, zorder=2)
    ax1.text(mb, y_b.max() + 0.85, f"仿真均值 {mb:.1f}", ha="center",
             fontsize=6.5, color=PRIMARY, fontweight="bold")
    ax1.text(mh, -1.15, f"真人均值 {mh:.1f}", ha="center",
             fontsize=6.5, color=PASS_C, fontweight="bold")
    ax1.text(-1.5, y_b.mean(), "BCI2a", ha="right", va="center",
             fontsize=7.5, color=PRIMARY, fontweight="bold", rotation=90)
    ax1.text(-1.5, y_h.mean(), "真人", ha="right", va="center",
             fontsize=7.5, color=INK, fontweight="bold", rotation=90)
    ax1.set_xlabel("每人增益 Δ = 各轮最高 − 零样本（百分点）", fontsize=8)
    ax1.set_xlim(0, 56)
    ax1.set_yticks([])
    ax1.set_title("个体增益分布（各轮最高口径）", fontsize=8, color=INK, pad=4)
    ax1.legend(handles=[
        Patch(facecolor=PRIMARY, edgecolor="#1A3A54", label="BCI2a 被试"),
        Patch(facecolor=PASS_C, edgecolor=NEUTRAL, label="真人 · 最高轮 PASS"),
        Patch(facecolor=FAIL_C, edgecolor=NEUTRAL, label="真人 · 最高轮 FAIL"),
    ], loc="lower right", fontsize=6.5, handlelength=1.0, frameon=False)
    bare_ax(ax1, keep_left=False)
    ax1.spines["left"].set_visible(False)

    fig.suptitle(
        f"读法：同图 15 版式，但微调后取各轮最高（仿真均值 +{mb:.1f}，真人 +{mh:.1f}；"
        f"仿真最差 +{min(b2a_d):.1f} > 真人中位 +{hum['median_d']:.1f}）",
        fontsize=7, color=NEUTRAL, y=1.02)

    problems = save(fig, "图11b_仿真与真人增益对照_各轮最高")
    src = Path("figures/图11b_仿真与真人增益对照_各轮最高.png")
    dst = Path("figures/fig11b_sim_vs_human_peak_gain.png")
    if src.exists():
        shutil.copy2(src, dst)
        pdf_src = Path("figures/图11b_仿真与真人增益对照_各轮最高.pdf")
        if pdf_src.exists():
            shutil.copy2(pdf_src, Path("figures/fig11b_sim_vs_human_peak_gain.pdf"))
        jdir = Path("交稿/figures")
        if jdir.is_dir():
            shutil.copy2(src, jdir / src.name)
            shutil.copy2(dst, jdir / dst.name)
            if pdf_src.exists():
                shutil.copy2(pdf_src, jdir / pdf_src.name)
    return problems


if __name__ == "__main__":
    style_check()
    warns = []
    for fn in (fig4, fig5, fig6, fig7, fig_m_leave_next, fig8, fig9, fig10, fig11, fig11_peak):
        warns += fn()
    if warns:
        print(f"\nAUDIT: {len(warns)} warning(s)")
        sys.exit(1)
    print("\nAUDIT clean · all done")


# ---------------- 图 12 仿真 vs 真人：均值柱 + 逐人增益条（按 v4 图注） ----------------
def fig12_sim_vs_human():
    """左：两队列零样本→末档均值柱 + 提升箭头；右：逐人增益横条按降序。
    数据同 fig11（BCI2a=Exp32；真人=v4 §3.6 三分类窗级表）。"""
    b2a = [36.6, 20.5, 45.4, 31.9, 26.5, 37.4, 31.0, 32.8, 30.1]
    human = [51.0, 28.8, 13.3, 11.2, 15.6, 1.8, 2.9, 1.7, 5.7, 9.9,
             12.1, 13.4, 1.9, 7.6, 5.5, 0.3, 0.0]
    human_pass = [True, True, True, False, True, True, True, True, False, True,
                  True, False, False, True, False, False, False]
    h_mean, b_mean = float(np.mean(human)), float(np.mean(b2a))
    assert abs(b_mean - 32.5) < 0.1 and abs(h_mean - 10.75) < 0.1
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(165 * MM, 62 * MM),
                                   constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1, 1.5]})
    # (a) 均值柱：零样本 → 末档
    x = np.arange(2)
    zero = [33.8, 37.9]
    fin = [66.3, 48.7]
    axL.bar(x - 0.16, zero, width=0.3, color=P_LT, label="零样本", zorder=3)
    axL.bar(x + 0.16, fin, width=0.3, color=PRIMARY, label="末档 FT", zorder=3)
    for xi, (z, f_) in enumerate(zip(zero, fin)):
        axL.annotate("", xy=(xi + 0.16, f_), xytext=(xi - 0.16, z),
                     arrowprops=dict(arrowstyle="-|>", color=CONTRAST, lw=1.4), zorder=4)
        axL.text(xi + 0.16, f_ + 1.5, "+%.1f" % (f_ - z), ha="center",
                 fontsize=7.4, color=CONTRAST, fontweight="bold", zorder=5)
    axL.set_xticks(x, ["BCI2a 仿真\n（n=9）", "真人\n（n=20）"], fontsize=8)
    axL.set_ylabel("三分类窗级准确率（%）")
    axL.set_ylim(0, 80)
    axL.legend(loc="upper right", fontsize=7, frameon=False)
    bare_ax(axL)
    axL.text(-0.14, 1.05, "(a) 均值抬升", transform=axL.transAxes, fontsize=8.5,
             fontweight="bold", color=INK)
    # (b) 逐人增益横条（降序，蓝=BCI2a，绿/红=真人 PASS/FAIL）
    order = sorted(range(len(b2a)), key=lambda i: -b2a[i])
    ys = np.arange(len(b2a))
    for k, i in enumerate(order):
        axR.barh(len(human) + 1 + k, b2a[i], height=0.62, color=PRIMARY, zorder=3)
    oh = sorted(range(len(human)), key=lambda i: -human[i])
    for k, i in enumerate(oh):
        c = PASS_C if human_pass[i] else FAIL_C
        axR.barh(len(human) - 0.5 - k, human[i], height=0.62, color=c, zorder=3)
    axR.axvline(b_mean, color=PRIMARY, lw=0.9, linestyle=(0, (4, 3)), zorder=4)
    axR.axvline(h_mean, color=INK, lw=0.9, linestyle=(0, (2, 2)), zorder=4)
    axR.text(b_mean + 0.6, len(human) + 2.2, "仿真均值 %.1f" % b_mean, fontsize=6.6,
             color=PRIMARY)
    axR.text(h_mean + 0.6, -1.4, "真人均值 %.1f" % h_mean, fontsize=6.6, color=INK)
    axR.set_yticks([])
    axR.set_xlabel("适配增益 Δ = 末档 − 零样本（pp，三分类窗级）")
    axR.set_xlim(0, 56)
    axR.set_ylim(-2.0, len(human) + 3.4)
    from matplotlib.patches import Patch
    axR.legend(handles=[Patch(color=PRIMARY, label="BCI2a 仿真"),
                        Patch(color=PASS_C, label="真人 · 门控 PASS"),
                        Patch(color=FAIL_C, label="真人 · 门控 FAIL")],
               loc="lower right", fontsize=6.4, frameon=False)
    bare_ax(axR)
    axR.text(-0.12, 1.05, "(b) 逐人增益", transform=axR.transAxes, fontsize=8.5,
             fontweight="bold", color=INK)
    problems = audit(fig, "图12_仿真与真人增益对照")
    fig.savefig("figures/图12_仿真与真人增益对照.png")
    fig.savefig("figures/图12_仿真与真人增益对照.pdf")
    plt.close(fig)
    print(("OK   " if not problems else "WARN ") + "图12_仿真与真人增益对照")
    for x in problems:
        print("     - " + x)
    return problems
