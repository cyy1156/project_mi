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
        ax.text(i, y + 0.006, f"{y:.3f}", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold" if i == 2 else "normal", color=INK, zorder=5)
    # 点间虚连线 + 逐段增量
    for i in range(2):
        ax.plot([i, i + 1], [ys[i], ys[i + 1]], ls=(0, (2, 2)), lw=0.7,
                color=NEUTRAL, zorder=2)
        ax.text(i + 0.5, max(ys[i], ys[i + 1]) + 0.0075,
                f"+{(ys[i + 1] - ys[i]) * 100:.1f} pp", ha="center", va="bottom",
                fontsize=6.8, color=NEUTRAL)
    ax.text(2, 0.622, "现行冻结口径", ha="center", va="bottom", fontsize=7.2,
            color=PRIMARY, fontweight="bold")
    ax.axhline(CHANCE, color=NEUTRAL, ls=(0, (4, 3)), lw=0.7, zorder=1)
    ax.text(-0.42, CHANCE + 0.006, "机会水平 1/3", ha="left", va="bottom",
            fontsize=6.8, color=NEUTRAL)
    ax.text(0.03, 0.965, "另：二分类 2 s → 3 s 提升 +4.7 pp（正文 3.1 节）",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=NEUTRAL)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.30, 0.655)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6])
    ax.set_xticks(range(3), ["1 s", "2 s", "3 s"])
    ax.set_ylabel("窗级三分类准确率")
    ax.set_xlabel("窗长（步长 100 ms）")
    ax.tick_params(axis="x", length=0)
    bare_ax(ax)
    return save(fig, "图4_窗长消融")


# ---------------- 图 5 十一模型选型（Cleveland 点图） ----------------
def fig5():
    """OpenBMI · 2s/hop100 · Acc_paper（与正文 §3.1 表一致）。
    协议：04_5060_旁路_2s滑窗100ms_openbmi_accpaper；正式数字锁自
    资料/实验结果/5090/openbmi滑窗_paper_acc。
    """
    models = [
        ("Deep4Net", 0.5431),
        ("ShallowFBCSPNet", 0.5398),
        ("Conformer", 0.5378),
        ("EEGNet", 0.5307),
        ("EEGTCNet", 0.5067),
        ("DGCNN_raw", 0.4906),
        ("DBN_raw", 0.4885),
        ("GCBNet_raw", 0.4764),
        ("DGCNN", 0.3891),
        ("DBN", 0.3809),
        ("GCBNet", 0.3746),
    ][::-1]
    names = [m[0] for m in models]
    vals = [m[1] for m in models]
    assert abs(vals[names.index("Deep4Net")] - vals[names.index("ShallowFBCSPNet")] - 0.0033) < 1e-9
    noz = {"DGCNN", "DBN", "GCBNet"}

    fig, ax = plt.subplots(figsize=(100 * MM, 94 * MM), constrained_layout=True)
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.plot([0.30, v - 0.0015], [i, i], color=STEM, lw=1.1, zorder=2)
        if n == "ShallowFBCSPNet":
            ax.scatter(v, i, s=68, color=PRIMARY, edgecolor="white", lw=0.85, zorder=5)
        elif n == "Deep4Net":
            ax.scatter(v, i, s=46, color=P_MED, edgecolor="white", lw=0.7, zorder=4)
        elif n in noz:
            ax.scatter(v, i, s=36, facecolor="white", edgecolor=NEUTRAL, lw=1.0, zorder=4)
        else:
            ax.scatter(v, i, s=38, color=P_LT, edgecolor=P_MED, lw=0.6, zorder=4)
        # Deep4 数值放点左侧，右侧留给旁注
        if n == "Deep4Net":
            ax.text(v - 0.004, i, f"{v:.4f}", va="center", ha="right", fontsize=6.8,
                    color=INK, zorder=5)
        else:
            ax.text(v + 0.004, i, f"{v:.4f}", va="center", ha="left", fontsize=6.8,
                    color=INK if n == "ShallowFBCSPNet" else NEUTRAL,
                    fontweight="bold" if n == "ShallowFBCSPNet" else "normal", zorder=5)

    ax.axvline(CHANCE, color=NEUTRAL, ls=(0, (4, 3)), lw=0.7, zorder=1)
    ax.text(CHANCE + 0.004, -0.75, "机会水平 1/3", fontsize=6.5, color=NEUTRAL,
            ha="left", va="center")
    ax.axhline(2.5, color=NEUTRAL, lw=0.6, ls=(0, (2, 2)), zorder=3)
    i_deep = names.index("Deep4Net")
    ax.annotate("+0.33 pp（折间噪声量级，未采纳）",
                xy=(0.5431, i_deep), xytext=(0.618, i_deep - 0.15),
                ha="left", va="center", fontsize=6.6, color=CONTRAST,
                arrowprops=dict(arrowstyle="-", color=CONTRAST, lw=0.6))
    ax.text(0.01, 0.02, "底部三项：无 z-score 预处理（与上方不可比）",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=NEUTRAL)

    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0.28, 0.72)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6])
    ax.set_ylim(-0.85, len(names) - 0.15)
    ax.set_xlabel("试次级三分类准确率（Acc_paper · 2 s / 100 ms）")
    ax.tick_params(axis="y", length=0, labelsize=7.2, pad=2)
    bare_ax(ax, keep_left=False)
    return save(fig, "图5_十一模型选型")


# ---------------- 图 6 主结果（竖柱对比）+ 混淆矩阵（行归一化热图）——规格 §3 ----------
def fig6():
    """通栏双面板：左读出对比竖柱（混粒度，柱标签声明）；右因果平滑窗级混淆热图。"""
    from matplotlib.colors import LinearSegmentedColormap

    vals = [0.5876, 0.5925, 0.6125, 0.6188]
    sd0 = 0.0296
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
        axL.text(i, yt, f"{v:.4f}", ha="center", va="bottom", fontsize=7.2,
                 fontweight="bold" if i == 3 else "normal", color=INK, zorder=5)
    # 「主结果」放在数值上方，避免与 0.6188 重叠
    axL.text(3, vals[3] + 0.042, "主结果", ha="center", va="bottom",
             fontsize=6.8, color=PRIMARY, fontweight="bold")
    axL.axhline(CHANCE, color=NEUTRAL, ls=(0, (4, 3)), lw=0.7, zorder=1)
    axL.text(-0.45, CHANCE + 0.008, "机会水平 1/3", ha="left", va="bottom",
             fontsize=6.8, color=NEUTRAL)
    axL.set_ylim(0.30, 0.72)
    axL.set_yticks([0.3, 0.4, 0.5, 0.6])
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
    ax.text(0, vals[0] + sds[0] + 0.014, "0.4198", ha="center", fontsize=7.2, color=INK)
    ax.text(1, vals[1] + sds[1] + 0.014, "0.6590", ha="center", fontsize=7.2,
            color=INK, fontweight="bold")
    ax.text(2, vals[2] + sds[2] + 0.014, "0.7003", ha="center", fontsize=7.2,
            color=CONTRAST)
    ax.text(0.5, (vals[0] + vals[1]) / 2 + 0.055, "+23.9 pp", ha="center",
            fontsize=7.5, color=PRIMARY, fontweight="bold")
    ax.text(1.5, (vals[1] + vals[2]) / 2 + 0.050, "+4.1 pp*", ha="center",
            fontsize=7.5, color=PRIMARY, fontweight="bold")
    ax.text(0.03, 0.97, "24/24 被试微调提升 ≥3 pp", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.2, color=PRIMARY)
    ax.set_xticks(x, ["零样本", "少样本微调", "+生理门控*"])
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0.28, 0.88)
    ax.set_yticks([0.3, 0.5, 0.7])
    ax.set_ylabel("试次级三分类准确率")
    ax.tick_params(axis="x", length=0, labelsize=8)
    ax.text(1.0, -0.13, "* 生理门控为离线分析口径（弃权率 59.3%），不进入正式在线协议",
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
    fig.supylabel("三分类窗级准确率（含空闲）", fontsize=7.5)
    fig.supxlabel(
        "实线 = 四成员 all4 · 虚线 = 只微调浅层再融合 so · 点线 = 单头 Shallow（S-fo）\n"
        "末档 R5：all4 0.663±0.067 vs so 0.452±0.047（Δ=+21.1 pp）；vs Shallow 0.671±0.067（同量级）",
        fontsize=7)
    return save(fig, "图8_BCI2a_LeaveNext曲线")


# ---------------- 图 9 指定集（桥形图：基线 → 增量 → 终值） ----------------
def fig9():
    tracks = [
        dict(x0=0.0, base=0.511, bsd=0.066, top=0.558, tsd=0.069,
             inc=0.047, dlab="+4.7 pp 折内拟合", bc=PRIMARY, ic=CONTRAST,
             ticks=("嵌套主读\n（交卷口径）", "折内附报\n（乐观偏置）"),
             grp="QuadFold-59 · 59 ch 交卷栈"),
        dict(x0=1.25, base=0.528, bsd=0.130, top=0.540, tsd=0.146,
             inc=0.012, dlab="+1.2 pp 融合", bc=P_MED, ic=PRIMARY,
             ticks=("V1 单模\n（选池附报）", "R-B8 融合\n（风险否决）"),
             grp="8 ch 微调栈 · OpenBMI 预训练"),
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
        ax.text(x0 + W / 2, t["base"] + t["bsd"] + 0.008, f"{t['base']:.3f}",
                ha="center", fontsize=7.2, color=INK, zorder=6)
        ax.text(xi + W / 2, t["top"] + t["tsd"] + 0.008, f"{t['top']:.3f}",
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
    ax.set_ylabel("指定集三分类准确率（六折）")
    ax.text(1.0, -0.135,
            "嵌套 = 其余五折 Val 拟合融合参数、本折评估 · 误差条 = 跨折 SD · 纵轴自 0.30 起",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=NEUTRAL)
    bare_ax(ax)
    return save(fig, "图9_指定集嵌套与交卷")


# ---------------- 图 10 真人 Leave-Next 逐轮柱状（每被试一面板） ----------------
def _load_real_leave_next_cohort():
    """读取各被试最新 all4 Leave-Next F5 summary → 逐轮试次级 MI（%）。"""
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
                f5 = r.get("f5_ft") or {}
                mi = f5.get("mi_acc")
                if mi is None:
                    continue
                mi_pct = float(mi) * 100.0 if float(mi) <= 1.0 else float(mi)
                rows.append({
                    "r": int(r.get("r_stage") or (len(rows) + 1)),
                    "mi": mi_pct,
                    "pass": bool(r.get("release_pass")),
                    "hold": str(r.get("heldout") or ""),
                })
            if not rows:
                continue
            by_sid[sid] = {"stamp": stamp, "path": str(p), "rows": rows}
    cohort = []
    for sid, pack in by_sid.items():
        rows = pack["rows"]
        cohort.append({
            "sid": sid,
            "mis": [r["mi"] for r in rows],
            "passes": [r["pass"] for r in rows],
            "final_mi": rows[-1]["mi"],
            "final_pass": rows[-1]["pass"],
            "n": len(rows),
        })
    cohort.sort(key=lambda x: (-x["final_mi"], x["sid"]))
    return cohort


def fig10():
    """通栏小倍数：每被试 Leave-Next 各轮 FT 试次级 MI 柱状图（含新增被试）。"""
    cohort = _load_real_leave_next_cohort()
    n = len(cohort)
    assert n >= 15, f"真人队列不足 15 人，当前 {n}"
    finals = [c["final_mi"] for c in cohort]
    n_pass = sum(1 for c in cohort if c["final_pass"])
    mean_final = float(np.mean(finals))

    # 布局：优先 4 列；17 人 → 5×4
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
        colors = [PASS_C if ok else P_LT for ok in c["passes"]]
        edges = [PASS_C if ok else NEUTRAL for ok in c["passes"]]
        # 末档加粗主色描边
        colors[-1] = PRIMARY if c["final_pass"] else P_MED
        edges[-1] = "#1A3A54" if c["final_pass"] else FAIL_C
        ax.bar(xs, c["mis"], width=0.72, color=colors, edgecolor=edges,
               linewidth=0.7, zorder=3)
        ax.axhline(50.0, color=NEUTRAL, ls=(0, (3, 2)), lw=0.55, zorder=1)
        ax.set_title(c["sid"], fontsize=7.2, color=INK, pad=2)
        ax.set_xlim(0.4, max_r + 0.6)
        ax.set_ylim(0, 100)
        ax.set_xticks(list(range(1, max_r + 1)))
        ax.set_yticks([0, 50, 100])
        ax.tick_params(labelsize=6.5, length=2, width=0.5)
        bare_ax(ax)
        # 末档数值
        ax.text(c["n"], min(97, c["final_mi"] + 4.5), f"{c['final_mi']:.0f}",
                ha="center", va="bottom", fontsize=6.5,
                color=PRIMARY if c["final_pass"] else FAIL_C, fontweight="bold")

    for ax in axes[:, 0]:
        ax.set_ylabel("试次级 MI（%）", fontsize=7)
    fig.supxlabel(
        f"Leave-Next 轮次 R1…Rn（柱色绿=该轮门控 PASS / 浅灰=FAIL；末档加粗）· n={n} · "
        f"末档均值 {mean_final:.1f}% · 末档 PASS {n_pass}/{n}",
        fontsize=7)
    return save(fig, "图10_真人LeaveNext逐轮MI")


if __name__ == "__main__":
    style_check()
    warns = []
    for fn in (fig4, fig5, fig6, fig7, fig_m_leave_next, fig8, fig9, fig10):
        warns += fn()
    if warns:
        print(f"\nAUDIT: {len(warns)} warning(s)")
        sys.exit(1)
    print("\nAUDIT clean · all done")
