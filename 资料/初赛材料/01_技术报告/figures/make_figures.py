# -*- coding: utf-8 -*-
"""报告插图生成脚本：图2-1 分层架构 / 图2-2 实时交互时序 / 图2-3 延迟构成"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
DARK = "#33475B"; GREY = "#8A94A0"; BODY = "#4A5A6A"; DUTY = "#8A94A0"

def band(ax, x, y, w, h, fc, ec, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + 18, y + 24, label, fontsize=10.5, fontweight="bold", color="#24425F")

def module(ax, x, y, w, h, ec, title, lines, duty=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                fc="#FFFFFF", ec=ec, lw=1.1))
    ax.text(x + 14, y + 22, title, fontsize=9.5, fontweight="bold", color="#24425F")
    yy = y + 41
    for ln in lines:
        ax.text(x + 14, yy, ln, fontsize=7.6, color=BODY)
        yy += 17
    if duty:
        ax.text(x + 14, y + h - 12, duty, fontsize=7.2, color=DUTY)

def arrow(ax, pts, label=None, lx=None, ly=None, ls="-", color=DARK, rot=0, fs=7.8):
    for i in range(len(pts) - 1):
        last = i == len(pts) - 2
        ax.add_patch(FancyArrowPatch(pts[i], pts[i + 1],
                     arrowstyle="-|>" if last else "-",
                     mutation_scale=11, color=color, lw=1.3, linestyle=ls,
                     shrinkA=0, shrinkB=0))
    if label:
        ax.text(lx, ly, label, fontsize=fs, color=DARK, rotation=rot)

# ============================== 图 2-1 分层架构 ==============================
fig, ax = plt.subplots(figsize=(12.6, 12.0), dpi=220)
ax.set_xlim(0, 1200); ax.set_ylim(1140, 0); ax.axis("off")
ax.text(34, 34, "在线系统分层架构与实时数据流（编号 ①–⑦ 对应实时链路）",
        fontsize=15, fontweight="bold", color="#24425F")

band(ax, 30, 55, 1140, 185, "#EEF3FB", "#B9CCE8", "客户端层（Client Layer）")
module(ax, 70, 92, 320, 132, "#7FA3D4", "① 游戏前端（被试视角）",
       ["Three.js 第一人称虚拟手与目标物", "Cue HUD 动觉引导语（左手/右手抓握）",
        "接收判定 → arm_reach 反馈动画", "试次计分显示（MI +1 / Rest +0.5）"],
       "职责：刺激呈现 · 反馈渲染 · 事件上报")
module(ax, 435, 92, 320, 132, "#7FA3D4", "② 操作台（实验员控制面）",
       ["被试登录 / 会话号管理", "实时波形、信号质量、三分类概率监视",
        "门控确认、暂停 / Reject / 中止", "按键化控制（P/N/G/R/Esc）"],
       "职责：会话控制 · 实时监视 · 审计留痕")
module(ax, 800, 92, 355, 132, "#7FA3D4", "③ 采集硬件（OpenBCI Cyton）",
       ["8 导运动皮层：FC3 C3 CP3 CZ CPZ FC4 C4 CP4", "采样率 250 Hz",
        "采集层滤波：0.5–45 Hz 带通 + 50 Hz 陷波", "支持合成板调试模式"],
       "职责：脑电信号获取（设备层）")

band(ax, 30, 262, 1140, 130, "#FFF7E8", "#E4CE9E", "通信接入层（Communication Layer）")
module(ax, 70, 300, 685, 78, "#D3B26A", "④ WebSocket 网关（双向消息通道）",
       ["控制面 token 鉴权 · 消息路由 · XSS 转义 · 心跳保活",
        "JSON 消息契约：范式状态 / 判定结果 / 概率流 / 控制指令"])
module(ax, 800, 300, 355, 78, "#D3B26A", "⑤ LSL 因果推流通道",
       ["Lab Streaming Layer 因果时间戳推流", "250 Hz 高频脑电流，仅供服务端消费"])

band(ax, 30, 414, 1140, 300, "#EDF7EE", "#B5D6B8", "服务层（Server Layer）")
module(ax, 70, 452, 340, 120, "#7BBF83", "⑥ 会话管理与范式状态机",
       ["被试登录、会话号分配与启停控制", "v3 时序：Rest 4s→prep 2s→Cue 1s→MI 4s→ITI 3s",
        "事件打标（Cue/MI 起止锚点）与试次计数"], "职责：交互协议驱动 · 会话生命周期")
module(ax, 435, 452, 340, 120, "#7BBF83", "⑦ 采集服务与信号质检",
       ["Cyton / 合成板接入，环形缓冲", "健康监测：缓冲龄 / 信号质量 / 通道完整性",
        "断流看门狗：停滞 2 s 告警 · 5 s 中止会话"], "职责：信号接入 · 质量防线")
module(ax, 800, 452, 355, 120, "#7BBF83", "⑧ 实时推理引擎（CausalFuse-8）",
       ["CAR → 8–30 Hz 带通 → 逐窗 z-score", "3 s / 100 ms 因果滑窗 · 四成员并行前向",
        "温度校准 + 固定权重融合（单窗 1.11 ms）"], "职责：脑电解码 · 概率输出")
module(ax, 70, 588, 340, 110, "#7BBF83", "⑨ 模型管理与晋升",
       ["按被试加载 subjects/*/models/current", "无个人模型时回退通用底座",
        "采后晋升 / 回滚 / 历史快照归档"], "职责：个体适配模型生命周期")
module(ax, 435, 588, 340, 110, "#7BBF83", "⑩ 状态同步与判定分发",
       ["100 ms 节拍：范式状态 × 滑窗判定同步", "因果平滑（lookback=2）+ 试次多数票",
        "判定 / 概率广播至游戏前端与操作台"], "职责：实时性枢纽 · 反馈触发源")
module(ax, 800, 588, 355, 110, "#7BBF83", "11· 安全与审计",
       ["控制指令鉴权与非法输入过滤", "操作审计日志 · 会话收尾崩溃安全落盘",
        "核心流水线 pytest 回归覆盖"], "职责：安全防线 · 可追溯性")

band(ax, 30, 736, 1140, 160, "#F5EFFA", "#D4C3E8",
     "数据存储层（Data Layer · 全链路落盘，任何评测可从原始数据重放）")
module(ax, 70, 775, 340, 105, "#B49BD6", "12· 会话原始数据",
       ["eeg.csv（原始脑电）· events.jsonl（事件标签）",
        "manifest.json / session.meta.json（元数据）", "continuous/ 与 by_phase/ 切分产物"])
module(ax, 435, 775, 340, 105, "#B49BD6", "13· 模型仓库",
       ["subjects/*/models/current（个人适配模型）", "members/ + e1f_overlay.json（融合参数）",
        "历史快照归档，可审计、可回滚"])
module(ax, 800, 775, 355, 105, "#B49BD6", "14· 配置权威与质检产物",
       ["v3_session.yaml / ft_policy.json / protocol.yaml",
        "alignment/verify_report.json（对齐校验）", "force_promote_warning.json（晋升告警）"])

band(ax, 30, 918, 1140, 98, "#FDF0EE", "#E8C0B8",
     "离线适配链路（会话结束自动触发，不在实时判定路径上）")
module(ax, 70, 954, 1060, 50, "#DCA79C", "15· Leave-Next 采后微调作业",
       ["全成员增量微调 + 10% 源域回放 → 门控（heldout 窗级 raw 准确率）→ 晋升或回滚；连续 FAIL 冻结在线更新"])

arrow(ax, [(595, 224), (595, 298)], "① 会话控制指令（上行）", 604, 266)
arrow(ax, [(230, 298), (230, 228)], "② 范式状态 / 计分（下行）", 96, 266)
arrow(ax, [(977, 224), (977, 298)], "③ 原始脑电 250 Hz", 986, 266)
arrow(ax, [(977, 378), (778, 450)], "③ LSL 因果推流", 806, 424)
arrow(ax, [(410, 512), (433, 512)], "启停/节拍", 322, 500, fs=7.2)
arrow(ax, [(775, 512), (798, 512)], "④ 最新 3 s 窗", 700, 498, fs=7.2)
arrow(ax, [(975, 572), (975, 580), (605, 580), (605, 586)], "⑤ 融合概率（100 ms 节拍）", 700, 574)
arrow(ax, [(433, 643), (412, 643), (412, 380)], "⑤ 判定 / 概率下行（推送双前端）", 420, 408)
arrow(ax, [(68, 512), (46, 512), (46, 828), (68, 828)], "⑥ 会话落盘（eeg / events / manifest）", 14, 690, rot=90)
arrow(ax, [(605, 698), (605, 773)], None)
arrow(ax, [(590, 773), (590, 698)], "⑦ 加载 / 晋升", 614, 742)
arrow(ax, [(977, 773), (977, 700)], "配置加载（冻结口径）", 985, 742, ls=(0, (5, 4)), color=GREY)
arrow(ax, [(240, 880), (240, 952)], "历史场次数据", 248, 922)
arrow(ax, [(700, 954), (700, 882)], "⑦ 晋升个人模型（下场生效）", 708, 922)

ax.text(34, 1052, "图例：", fontsize=8.5, color="#66788A")
leg = [("#EEF3FB", "#B9CCE8", "客户端层"), ("#FFF7E8", "#E4CE9E", "通信接入层"),
       ("#EDF7EE", "#B5D6B8", "服务层"), ("#F5EFFA", "#D4C3E8", "数据存储层"),
       ("#FDF0EE", "#E8C0B8", "离线适配链路（非实时路径）")]
xx = 78
for fc, ec, name in leg:
    ax.add_patch(FancyBboxPatch((xx, 1042), 16, 13, boxstyle="round,pad=0,rounding_size=3", fc=fc, ec=ec, lw=1))
    ax.text(xx + 22, 1052, name, fontsize=8.5, color="#66788A")
    xx += 22 + len(name) * 12 + 24
ax.text(34, 1085, "实时性要点：判定延迟由 3 s 窗口时程主导（因果平滑读出约 3.5 s），算法单窗前向仅 1.11 ms，远小于 100 ms 节拍；",
        fontsize=9, color=BODY)
ax.text(34, 1105, "门控、微调、落盘均在会话末触发或旁路执行，不占用实时判定路径；看门狗与质量门控保证异常显式化而非静默吸收。",
        fontsize=9, color=BODY)
fig.savefig(os.path.join(OUT, "图1_在线系统分层架构.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================== 图 2-2 实时交互时序 ==============================
fig, ax = plt.subplots(figsize=(12.6, 7.2), dpi=220)
ax.set_xlim(0, 1200); ax.set_ylim(760, 0); ax.axis("off")
ax.text(34, 34, "实时交互全链路（从被试进入会话到接收反馈，四泳道 × 八步）",
        fontsize=15, fontweight="bold", color="#24425F")
lanes = [
    ("被试 / 游戏前端", [
        ("1", "进入会话", ["操作台完成登录与会话号分配，", "游戏前端加载个人模型", "（无则回退通用底座）。"]),
        ("2", "接收范式状态", ["按 100 ms 节拍收到状态机广播", "（Rest→prep→Cue→MI），", "Cue 阶段 HUD 显示动觉引导语。"]),
        ("5", "实时反馈", ["判定窗命中后收到判定结果，", "虚拟手执行 arm_reach 动画", "（端到端约 3.5 s，由窗口时程主导）。"]),
        ("6", "试次结算", ["试次结束收到多数票判定与计分", "（MI +1 / Rest +0.5）。"]),
    ]),
    ("服务端 · 会话与采集", [
        ("1", "会话建立", ["会话管理分配会话号，启动 v3 范式", "状态机与事件打标", "（Cue/MI 起止锚点）。"]),
        ("3", "信号接入", ["Cyton 经 LSL 因果推流 250 Hz；", "采集服务缓冲并健康监测，", "看门狗停滞 2 s 告警、5 s 中止。"]),
        ("4", "供窗", ["按 100 ms 步长向推理引擎输出", "最新 3 s 窗（仅含历史数据，", "满足在线因果性）。"]),
        ("7", "落盘与校验", ["eeg.csv / events.jsonl / manifest", "落盘；标记—脑电对齐校验，", "不合格会话不入库。"]),
    ]),
    ("服务端 · 推理与分发", [
        ("4", "实时解码", ["预处理（CAR→8–30 Hz→z-score）后", "四成员并行前向，温度校准 +", "固定权重融合，单窗 1.11 ms。"]),
        ("5", "判定与分发", ["因果平滑（lookback=2）+ argmax", "得窗级判定；概率流推送操作台、", "判定推送游戏前端。"]),
        ("6", "试次多数票", ["11 档判定点多数票产生试次结果，", "驱动计分与反馈广播。"]),
        ("8", "采后适配", ["会话结束触发 Leave-Next 全成员", "微调（10% 源域回放），门控通过", "后晋升个人模型。"]),
    ]),
    ("操作台 / 数据存储", [
        ("1", "控制下发", ["实验员下发启停/暂停/中止指令", "（token 鉴权，全程审计日志）。"]),
        ("4–6", "实时监视", ["持续接收波形、信号质量与三分类", "概率流，必要时门控确认或 Reject。"]),
        ("7", "数据归档", ["原始脑电、事件、元数据、配置快照", "全链路落盘；校验报告与晋升告警", "写入质检产物。"]),
        ("8", "模型归档", ["晋升的个人模型写入 subjects/*/", "models/current，历史快照归档，", "下一会话自动生效。"]),
    ]),
]
for ci, (head, steps) in enumerate(lanes):
    x0 = 30 + ci * 292
    ax.add_patch(FancyBboxPatch((x0, 60), 272, 670, boxstyle="round,pad=0,rounding_size=10",
                                fc="#F8FAFC", ec="#DCE4EC", lw=1))
    ax.add_patch(FancyBboxPatch((x0, 60), 272, 34, boxstyle="round,pad=0,rounding_size=10",
                                fc="#3D6FB4", ec="#3D6FB4", lw=1))
    ax.text(x0 + 136, 82, head, fontsize=10.5, fontweight="bold", color="white", ha="center")
    yy = 112
    for num, title, body_lines in steps:
        h = 132
        ax.add_patch(FancyBboxPatch((x0 + 10, yy), 252, h, boxstyle="round,pad=0,rounding_size=7",
                                    fc="#FFFFFF", ec="#D8E0E8", lw=1))
        ax.add_patch(FancyBboxPatch((x0 + 18, yy + 8), 22, 18, boxstyle="round,pad=0,rounding_size=4",
                                    fc="#3D6FB4", ec="#3D6FB4", lw=0))
        ax.text(x0 + 29, yy + 17, num, fontsize=8, color="white", ha="center", va="center")
        ax.text(x0 + 46, yy + 17, title, fontsize=9.3, fontweight="bold", color="#24425F")
        lyy = yy + 42
        for ln in body_lines:
            ax.text(x0 + 18, lyy, ln, fontsize=7.6, color=BODY)
            lyy += 15
        yy += h + 12
fig.savefig(os.path.join(OUT, "图2_实时交互链路时序.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================== 图 2-3 判定延迟构成 ==============================
fig, ax = plt.subplots(figsize=(11.0, 2.8), dpi=220)
ax.set_xlim(-0.05, 3.9); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.02, 0.05, "在线判定延迟构成（W 臂全窗多数票读出为 4.00 s）", fontsize=11.5,
        fontweight="bold", color="#24425F")
ax.barh(0.45, 3.0, left=0.0, height=0.30, color="#3D6FB4", edgecolor="white")
ax.barh(0.45, 0.5, left=3.0, height=0.30, color="#7FA3D4", edgecolor="white")
ax.text(1.5, 0.45, "窗口时程 3.0 s（因果滑窗积累）", ha="center", va="center",
        color="white", fontsize=10.5, fontweight="bold")
ax.text(3.25, 0.45, "0.5 s", ha="center", va="center", color="white",
        fontsize=10, fontweight="bold")
ax.text(3.25, 0.22, "判定点覆盖 0.5 s\n（11 档判定点，3.0–4.0 s）", ha="center",
        va="top", fontsize=8.8, color="#33517E")
ax.annotate("算法前向 1.11 ms（此尺度下不可见，远小于 100 ms 节拍）",
            xy=(3.49, 0.52), xytext=(0.9, 0.90), fontsize=9.5, color="#B03A2E",
            arrowprops=dict(arrowstyle="->", color="#B03A2E", lw=1.2))
ax.plot([3.5, 3.5], [0.30, 0.60], color="#24425F", lw=1.4, ls="--")
ax.annotate("判定延迟上界 ≈ 3.5 s\n（自 Cue 起，因果平滑读出）",
            xy=(3.52, 0.38), xytext=(3.30, 0.92), fontsize=9.5, color="#24425F",
            arrowprops=dict(arrowstyle="->", color="#24425F", lw=1.2))
for s in [0, 1, 2, 3]:
    ax.plot([s, s], [0.60, 0.65], color="#33475B", lw=1)
    ax.text(s, 0.70, f"{s} s", ha="center", fontsize=9, color="#33475B")
fig.savefig(os.path.join(OUT, "图3_判定延迟构成.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
fig.savefig(os.path.join(OUT, "图3_判定延迟构成.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("done:", os.listdir(OUT))
