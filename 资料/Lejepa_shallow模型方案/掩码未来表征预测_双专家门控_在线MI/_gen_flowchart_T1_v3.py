"""Generate T-series v3 train/infer flowchart with precise arrow geometry."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

OUT = Path(__file__).with_name("framework_flowchart_T1_v3_train_infer.png")

# pastel palette aligned with prior figures
C = {
    "enc": "#BBDEFB",
    "pred": "#C8E6C9",
    "cur": "#FFF9C4",
    "fut": "#E1BEE7",
    "gate": "#FFE0B2",
    "dec": "#F8BBD0",
    "past": "#B0BEC5",
    "now": "#90CAF9",
    "future": "#A5D6A7",
    "meta": "#E0F7FA",
    "dis": "#EEEEEE",
    "edge": "#37474F",
    "loss": "#C62828",
}


def box(ax, xy, wh, text, fc, *, fontsize=8, ec=None, ls="-", lw=1.2, alpha=1.0):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec or C["edge"],
        linewidth=lw,
        linestyle=ls,
        alpha=alpha,
        mutation_aspect=0.5,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#212121",
        wrap=True,
    )
    return (x, y, w, h)


def center(b):
    x, y, w, h = b
    return x + w / 2, y + h / 2


def top(b):
    x, y, w, h = b
    return x + w / 2, y + h


def bottom(b):
    x, y, w, h = b
    return x + w / 2, y


def left(b):
    x, y, w, h = b
    return x, y + h / 2


def right(b):
    x, y, w, h = b
    return x + w, y + h / 2


def arrow(ax, p0, p1, *, color=None, ls="-", lw=1.35, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            linestyle=ls,
            color=color or C["edge"],
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1,
            shrinkB=1,
        )
    )


def v_arrow(ax, b_from, b_to, **kw):
    arrow(ax, bottom(b_from), top(b_to), **kw)


def h_arrow(ax, b_from, b_to, **kw):
    arrow(ax, right(b_from), left(b_to), **kw)


def panel_frame(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.15, 0.2),
            9.7,
            13.5,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor="white",
            edgecolor="#CFD8DC",
            linewidth=1.5,
        )
    )
    ax.text(5, 13.35, title, ha="center", va="top", fontsize=11, fontweight="bold")


def draw_timeline(ax, y, *, future_disabled=False):
    segs = [
        (0.5, 2.6, C["past"], "Past 100\n(-0.4~0s)"),
        (3.2, 2.6, C["now"], "Current 500\n(0~2s)"),
        (5.9, 2.6, C["future"], "Future 400\n(2~3.6s)"),
    ]
    for x, w, fc, t in segs:
        b = box(ax, (x, y), (w, 0.7), t, fc, fontsize=7)
        if future_disabled and "Future" in t:
            ax.plot([x, x + w], [y, y + 0.7], color="#9E9E9E", lw=2)
            ax.plot([x, x + w], [y + 0.7, y], color="#9E9E9E", lw=2)
            ax.text(x + w / 2, y - 0.22, "不可见", ha="center", fontsize=7, color="#757575")
    return segs


def v_arrow_x(ax, x, y0, y1, **kw):
    arrow(ax, (x, y0), (x, y1), **kw)


def draw_train(ax):
    panel_frame(ax, "A. 训练阶段  Offline Training")
    draw_timeline(ax, 12.3)

    box(ax, (0.5, 11.35), (4.3, 0.55), r"$X_{mask}=[Past,Cur,0_{future}]$", "#FAFAFA", fontsize=7.5)
    box(ax, (5.2, 11.35), (4.3, 0.55), r"$X_{full}=[Past,Cur,Future]$", "#FAFAFA", fontsize=7.5)

    enc = box(ax, (2.4, 10.2), (5.2, 0.75), "Shared Encoder\n共享编码器", C["enc"], fontsize=9)
    arrow(ax, (2.65, 11.35), (4.0, 10.95))
    arrow(ax, (7.35, 11.35), (6.0, 10.95))

    # Left stack centered on x=2.7 : H_vis → Predictor → H_pre
    xL, wL = 0.7, 4.0
    xmid = xL + wL / 2

    hvis = box(
        ax,
        (xL, 8.95),
        (wL, 0.8),
        r"$H_{vis}$ 可见段 token" + "\n" + r"$(B\times L_{vis}\times D)$",
        "#E3F2FD",
        fontsize=8,
    )
    htgt = box(
        ax,
        (5.3, 8.95),
        (4.0, 0.8),
        r"$H_{target}$（sg）未来 token" + "\n" + r"$(B\times L_{fut}\times D)$",
        "#E8F5E9",
        fontsize=8,
    )
    arrow(ax, (4.0, 10.2), (xmid, 9.75))
    arrow(ax, (6.0, 10.2), top(htgt))

    pred = box(
        ax,
        (xL, 6.95),
        (wL, 1.5),
        "PosTokenFuturePredictor\n"
        + r"$ctx=MLP(mean(H_{vis}))$"
        + "\n"
        + r"$Z_{pre}[j]=LN(E_{pos}[j]+ctx)$"
        + "\n无 MHA · 无 $E_{phase}$",
        C["pred"],
        fontsize=7.5,
    )
    # ONE centered spine: H_vis → Predictor → H_pre
    v_arrow_x(ax, xmid, bottom(hvis)[1], top(pred)[1])

    hpre = box(
        ax,
        (xL, 5.65),
        (wL, 0.8),
        r"$H_{pre}/Z_{pre}$ 预测未来 token" + "\n" + r"$(B\times L_{fut}\times D)$",
        "#E8F5E9",
        fontsize=8,
    )
    v_arrow_x(ax, xmid, bottom(pred)[1], top(hpre)[1])

    # L_pred at H_pre height → up to H_target (does NOT enter Predictor)
    y_lp = center(hpre)[1]
    ax.plot([xL + wL, 7.3, 7.3], [y_lp, y_lp, 8.95], color=C["loss"], lw=1.35, ls="--")
    ax.annotate(
        "",
        xy=(7.3, 8.95),
        xytext=(7.3, y_lp + 0.05),
        arrowprops=dict(arrowstyle="-|>", color=C["loss"], lw=1.35, linestyle="--"),
    )
    ax.text(5.9, y_lp + 0.22, r"$L_{pred}$（token MSE）", color=C["loss"], fontsize=7.5, ha="center")

    ecur = box(
        ax,
        (0.45, 4.2),
        (3.0, 0.95),
        "Expert_cur\n" + r"$AttnPool(H_{vis})\to p_{cur}$",
        C["cur"],
        fontsize=7.5,
    )
    efut = box(
        ax,
        (3.7, 4.2),
        (3.0, 0.95),
        "Expert_future\n" + r"$AttnPool(H_{pre})\to p_{future}$",
        C["fut"],
        fontsize=7.5,
    )
    # H_vis → Expert_cur: RIGHT-side bypass (do not hug Predictor left edge)
    xr = xL + wL + 0.18
    ax.plot(
        [right(hvis)[0], xr, xr, center(ecur)[0]],
        [center(hvis)[1], center(hvis)[1], top(ecur)[1] + 0.2, top(ecur)[1] + 0.2],
        color=C["edge"],
        lw=1.2,
    )
    arrow(ax, (center(ecur)[0], top(ecur)[1] + 0.2), top(ecur))
    ax.text(xr + 0.08, 7.4, "→Expert_cur", fontsize=6.5, color="#546E7A", rotation=90, va="center")
    v_arrow_x(ax, center(efut)[0], bottom(hpre)[1], top(efut)[1])

    gate = box(
        ax,
        (1.6, 2.85),
        (4.0, 0.95),
        "Gate\n"
        + r"$\alpha\!\leftarrow\![z_{vis},z_{pre}]$; "
        + r"$p_{final}=\alpha p_{cur}+(1-\alpha)p_{future}$",
        C["gate"],
        fontsize=7,
    )
    arrow(ax, bottom(ecur), (2.6, 3.8))
    arrow(ax, bottom(efut), (4.6, 3.8))
    arrow(ax, (3.6, 3.8), top(gate))

    dec = box(
        ax,
        (7.0, 4.2),
        (2.5, 1.2),
        "Decoder\n(train only)\n"
        + r"$H_{pre}\!\to\!\hat X_{future}$"
        + "\n"
        + r"$L_{dec}$",
        C["dec"],
        fontsize=7,
    )
    arrow(ax, (xL + wL, center(hpre)[1] + 0.15), left(dec))

    ax.annotate(
        r"SIGReg$(z_{mask}^{vis})\!\to\!L_{SIGReg}$",
        xy=(xmid + 0.15, center(hvis)[1] - 0.05),
        xytext=(6.55, 7.85),
        fontsize=7,
        color=C["loss"],
        ha="left",
        arrowprops=dict(arrowstyle="-|>", color=C["loss"], lw=1.1, linestyle="--"),
    )

    ax.text(5.9, 2.55, r"$y$", fontsize=9, ha="center")
    ax.annotate(
        r"$L_{cls}=CE(p_{final},y)$" + "\n" + r"$y$ 仅 CE，不进 Predictor",
        xy=(5.6, 3.1),
        xytext=(7.15, 2.3),
        fontsize=7,
        color=C["loss"],
        arrowprops=dict(arrowstyle="-|>", color=C["loss"], lw=1.1, linestyle="--"),
    )

    ax.text(
        5.0,
        0.55,
        r"$L_{total}=\lambda_{pred}L_{pred}+\lambda_{dec}L_{dec}+\lambda_{sig}L_{SIGReg}+\lambda_{cls}L_{cls}$",
        ha="center",
        fontsize=8,
        color="#212121",
    )


def draw_infer(ax):
    panel_frame(ax, "B. 在线推理  Online Inference")
    draw_timeline(ax, 12.3, future_disabled=True)

    box(ax, (1.5, 11.35), (7.0, 0.55), r"$X_{mask}=[Past,Cur,0_{future}]$  （唯一输入）", "#FAFAFA", fontsize=8)

    enc = box(ax, (2.4, 10.2), (5.2, 0.75), "Shared Encoder\n共享编码器", C["enc"], fontsize=9)
    arrow(ax, (5.0, 11.35), top(enc))

    xL, wL = 2.5, 5.0
    xmid = xL + wL / 2

    hvis = box(
        ax,
        (xL, 8.95),
        (wL, 0.8),
        r"$H_{vis}$ 可见段 token" + "\n" + r"$(B\times L_{vis}\times D)$",
        "#E3F2FD",
        fontsize=8,
    )
    v_arrow_x(ax, xmid, 10.2, 8.95 + 0.8)

    ht = box(
        ax,
        (7.55, 8.95),
        (2.0, 0.8),
        r"$H_{target}$" + "\n无",
        C["dis"],
        fontsize=7,
        ls="--",
        ec="#9E9E9E",
    )
    ax.text(center(ht)[0], center(ht)[1] + 0.28, "X", ha="center", fontsize=12, color="#9E9E9E", fontweight="bold")

    pred = box(
        ax,
        (xL, 6.95),
        (wL, 1.5),
        "PosTokenFuturePredictor\n"
        + r"$ctx=MLP(mean(H_{vis}))$"
        + "\n"
        + r"$Z_{pre}[j]=LN(E_{pos}[j]+ctx)$"
        + "\n无 MHA · 无 Phase / $t0$",
        C["pred"],
        fontsize=7.5,
    )
    v_arrow_x(ax, xmid, bottom(hvis)[1], top(pred)[1])

    hpre = box(
        ax,
        (xL, 5.65),
        (wL, 0.8),
        r"$H_{pre}/Z_{pre}$" + "\n" + r"$(B\times L_{fut}\times D)$",
        "#E8F5E9",
        fontsize=8,
    )
    v_arrow_x(ax, xmid, bottom(pred)[1], top(hpre)[1])

    ecur = box(
        ax,
        (1.2, 4.2),
        (3.4, 0.95),
        "Expert_cur\n" + r"$AttnPool(H_{vis})\to p_{cur}$",
        C["cur"],
        fontsize=7.5,
    )
    efut = box(
        ax,
        (5.2, 4.2),
        (3.4, 0.95),
        "Expert_future\n" + r"$AttnPool(H_{pre})\to p_{future}$",
        C["fut"],
        fontsize=7.5,
    )
    xr = xL - 0.22
    ax.plot(
        [left(hvis)[0], xr, xr, center(ecur)[0]],
        [center(hvis)[1], center(hvis)[1], top(ecur)[1] + 0.2, top(ecur)[1] + 0.2],
        color=C["edge"],
        lw=1.2,
    )
    arrow(ax, (center(ecur)[0], top(ecur)[1] + 0.2), top(ecur))
    v_arrow_x(ax, center(efut)[0], bottom(hpre)[1], top(efut)[1])

    gate = box(
        ax,
        (2.5, 2.85),
        (5.0, 0.95),
        "Gate → " + r"$p_{final}$",
        C["gate"],
        fontsize=9,
    )
    arrow(ax, bottom(ecur), (3.5, 3.8))
    arrow(ax, bottom(efut), (6.5, 3.8))
    arrow(ax, (5.0, 3.8), top(gate))

    box(
        ax,
        (7.3, 4.05),
        (2.2, 1.1),
        "Decoder\n不跑 / X",
        C["dis"],
        fontsize=8,
        ls="--",
        ec="#9E9E9E",
    )

    yhat = box(
        ax,
        (2.5, 1.55),
        (5.0, 0.85),
        r"$\hat y=\mathrm{argmax}(p_{final})$",
        "#E8EAF6",
        fontsize=10,
    )
    v_arrow(ax, gate, yhat)

    box(
        ax,
        (1.5, 0.45),
        (7.0, 0.7),
        "合法：仅 $E_{pos}$ + 可见段上下文；禁止 $y$ / Future EEG",
        C["meta"],
        fontsize=7.5,
    )


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 10.5), dpi=160, facecolor="white")
    fig.suptitle(
        "T系列 v3 · E_pos Token Predictor（无Cross-Attn / 无Phase）· 双专家门控在线MI",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.945,
        r"相对P2：PosTokenFuturePredictor + token级$L_{pred}$ + AttnPool；Predictor 不读 $y$ / $t0_{sec}$",
        ha="center",
        fontsize=9,
        color="#546E7A",
    )

    ax1 = fig.add_axes([0.02, 0.08, 0.47, 0.85])
    ax2 = fig.add_axes([0.51, 0.08, 0.47, 0.85])
    draw_train(ax1)
    draw_infer(ax2)

    legend = [
        Line2D([0], [0], color=C["edge"], lw=1.5, label="实线 = 数据流"),
        Line2D([0], [0], color=C["loss"], lw=1.5, ls="--", label="红色虚线 = 损失"),
        Line2D([0], [0], color="#9E9E9E", lw=1.5, ls="--", label="灰虚线框+X = 在线禁用"),
        Line2D([0], [0], color="#2E7D32", lw=4, label="绿色框 = v3 Predictor（无Phase）"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=9)

    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
