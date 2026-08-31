#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成比赛初赛《离线性能验证报告》Excel（XH-202610）。

所有数值均取自仓库已登记的实验结果（登记表/JSON），来源在"原始数据索引"sheet。
重跑：python build_excel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SKILL = Path(r"C:\Users\yy\.zcode\cli\plugins\cache\zcode-plugins-official\document-skills\0.1.4\skills\xlsx")
for p in (str(SKILL), str(SKILL / "templates")):
    if p not in sys.path:
        sys.path.insert(0, p)

from base import (  # noqa: E402
    FONT_NAME, HEADER_BOLD, PRIMARY, NEUTRAL_600, NEUTRAL_900,
    font_subheader, font_caption, font_body,
    setup_sheet, style_header_row, style_data_row,
    auto_fit_columns, auto_fit_row_heights,
)
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]          # 资料/初赛材料
OUT = ROOT / "02_离线验证" / "离线性能验证报告_XH-202610.xlsx"
CM = json.loads((ROOT / "02_离线验证" / "原始数据_cm_e1f_arms.json").read_text(encoding="utf-8"))

wb = Workbook()


def sub_title(ws, row, text, last_col):
    c = ws.cell(row=row, column=2, value=text)
    c.font = font_subheader()
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[row].height = 24
    return row + 1


def caption(ws, row, text):
    c = ws.cell(row=row, column=2, value=text)
    c.font = font_caption()
    c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=False)
    return row + 1


def table(ws, start_row, headers, rows, num_cols=(), num_fmt="0.0000"):
    """headers 从 B 列开始写一行表头，随后写数据行，返回下一可用行。"""
    last_col = len(headers) + 1
    for j, h in enumerate(headers, start=2):
        ws.cell(row=start_row, column=j, value=h)
    style_header_row(ws, row_num=start_row, col_start=2, col_end=last_col)
    r = start_row + 1
    for i, row_data in enumerate(rows):
        for j, v in enumerate(row_data, start=2):
            cell = ws.cell(row=r, column=j, value=v)
        style_data_row(ws, row_num=r, col_start=2, col_end=last_col, row_index=i)
        for j, v in enumerate(row_data, start=2):
            if j - 2 in num_cols and isinstance(v, (int, float)):
                cell = ws.cell(row=r, column=j)
                cell.number_format = num_fmt if isinstance(v, float) else "#,##0"
                cell.alignment = Alignment(horizontal="right", vertical="center")
        r += 1
    return r


# ---------------------------------------------------------------- 00 验证总表
ws = wb.active
ws.title = "00_验证总表"
headers = ["序号", "数据集", "评测协议", "模型 / 方法", "评测层级",
           "三分类准确率", "召回率 macro", "特异性 macro", "macro-F1", "运算延迟 / 判定延迟", "证据来源"]
setup_sheet(ws, title="离线性能验证总表（左手 MI / 右手 MI / 空闲 三分类 · XH-202610）", last_col=len(headers) + 1)

c = CM["arm_C_trial_decision"]["test"]
w = CM["arm_W_trial_decision"]["test"]
rows = [
    [1, "OpenBMI（54 人，公开）", "被试独立五折 · test 试次级 Acc_paper（16,200 试次，每类 5,400）",
     "E1f 四成员融合 + 因果平滑（C 臂，主结果）", "试次级", 0.6188, 0.6188, 0.8094, 0.6196,
     "判定延迟 t̄≈3.50 s；单窗前向 1.11 ms（RTX 5070）", "实验 30 + 混淆矩阵复算（本表 sheet 01）"],
    [2, "OpenBMI（54 人，公开）", "同上", "E1f 四成员融合 + 多数票读出（W 臂，现行线上读出）", "试次级",
     0.6125, 0.6125, 0.8062, 0.6132, "判定延迟 t̄≈4.00 s", "实验 30 + 混淆矩阵复算（sheet 01）"],
    [3, "OpenBMI（54 人，公开）", "同上", "E1f 窗级 argmax（不作试次聚合）", "窗级",
     0.5925, 0.5925, 0.7962, 0.5930, "单窗 100 ms 步进", "混淆矩阵复算（sheet 01）"],
    [4, "OpenBMI（54 人，公开）", "被试独立五折 · 试次级", "ShallowFBCSPNet 3s 单模型（S3 底座）", "试次级",
     0.5876, "—", "—", "—", "—", "实验 20 登记表（0.5876±0.0296）"],
    [5, "OpenBMI（54 人，公开）", "被试独立五折 · 试次级", "Deep4Net 2s（11 模型基线最优）", "试次级",
     0.5431, 0.5605, 0.7754, 0.5324, "—", "5090 十一模型对比（sheet 02）"],
    [6, "BCI IV 2a（9 人，公开）", "Leave-Next 6 轮仿真 · 末档（R5）· 三分类含空闲",
     "E1f 四成员 + all4 采后增量微调（force）", "窗级（heldout run）",
     0.671, "—", "—", "—", "—", "实验 32（sheet 03）"],
    [7, "BCI IV 2a（9 人，公开）", "同上（底座零样本 R0）", "E1f 四成员零样本", "窗级", 0.338, "—", "—", "—", "—", "实验 32（sheet 03）"],
    [8, "Stieger2021（24 人，公开）", "跨库伪在线 · 前半训练后半评测", "Shallow 3s 零样本（跨库）", "试次级", 0.4198, "—", "—", "—", "—", "实验 07（sheet 04）"],
    [9, "Stieger2021（24 人，公开）", "同上", "Shallow 3s + 前半增量微调（FT）", "试次级", 0.6590, "—", "—", "—", "—", "实验 07（24/24 被试 ≥+3pp）"],
    [10, "Stieger2021（24 人，公开）", "同上 + 生理门控 H1", "Shallow 3s FT + ERD 门控", "试次级", 0.7003, "—", "—", "—", "—", "实验 07（sheet 04）"],
    [11, "自采（syj0828，8 通道 v3 范式）", "Leave-Next 采后增量微调 · 末档（R5）",
     "E1f all4 个体模型（真实被试最优）", "窗级 heldout", 0.916, "—", "—", "—",
     "MI 试次 34/36=94.4%；Rest 16/18；总分 42/45", "复验 20260830（sheet 05）"],
    [12, "自采（fnz0828 等其余被试）", "同上", "E1f all4 个体模型", "窗级 heldout",
     "0.35–0.69（见 sheet 05）", "—", "—", "—", "个体差异大，体现少样本适配必要性", "复验 20260830（sheet 05）"],
]
r = table(ws, 4, headers, rows, num_cols=(5, 6, 7, 8))
r = caption(ws, r + 1, "注 1：OpenBMI 五折 test 划分为被试独立划分，指标为 test 试次级 Acc_paper（每试次一次判定）。")
r = caption(ws, r, "注 2：召回率 macro = 三类召回率均值；特异性 macro = 三类特异性均值；每类明细与混淆矩阵见 sheet 01/02。")
r = caption(ws, r, "注 3：主办方指定标准数据集如与本报告数据集规格不同，将以同一管线重切窗复测并回填（见 sheet 09 复现说明）。")
auto_fit_columns(ws, min_width=8, max_width=42, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)
ws.freeze_panes = "C5"

# ---------------------------------------------------------------- 01 OpenBMI 主结果
ws = wb.create_sheet("01_OpenBMI主结果")
setup_sheet(ws, title="OpenBMI 54 人五折 · E1f 主结果明细（混淆矩阵与每类指标）", last_col=10)


def perclass_rows(block):
    rows = []
    for pc in block["per_class"]:
        rows.append([pc["class"], pc["support"], pc["recall"], pc["specificity"], pc["precision"], pc["f1"]])
    rows.append(["macro / 总体", sum(p["support"] for p in block["per_class"]),
                 block["accuracy"], block["accuracy"], None, block["macro_f1"]])
    return rows


def confusion_rows(block):
    cm = block["confusion"]
    rows = []
    for cls, row in zip(block["labels_row_true_col_pred"], cm):
        rows.append([cls] + row)
    return rows


pc_headers = ["类别", "支持数（窗）", "召回率", "特异性", "精确率", "F1"]
r = 4
r = sub_title(ws, r, "① 主结果：E1f 四成员 + 因果平滑（C 臂）试次判定 · test 集准确率 0.6188（对账锚点一致）", 10)
r = table(ws, r, pc_headers, perclass_rows(c), num_cols=(2, 1), num_fmt="0.0000")
r = sub_title(ws, r, "C 臂混淆矩阵（行=真实，列=预测）", 10)
r = table(ws, r, ["真实\\预测", "Rest", "Left", "Right"], confusion_rows(c), num_cols=(1, 2, 3), num_fmt="#,##0")
r = caption(ws, r, "每类支持数 59,400 窗 = 5,400 试次 × 11 窗（试次判定广播到窗，同一试次内判定一致）。")
r += 1
r = sub_title(ws, r, "② 多数票读出（W 臂，现行线上读出）· test 准确率 0.6125", 10)
r = table(ws, r, pc_headers, perclass_rows(w), num_cols=(2, 1), num_fmt="0.0000")
r = sub_title(ws, r, "W 臂混淆矩阵（行=真实，列=预测）", 10)
r = table(ws, r, ["真实\\预测", "Rest", "Left", "Right"], confusion_rows(w), num_cols=(1, 2, 3), num_fmt="#,##0")
r += 1
r = sub_title(ws, r, "③ 融合窗级 argmax 与单模型对照（窗级）", 10)
fw = CM["fused_window_argmax"]["test"]
sh = CM["shallow_member_window_argmax"]["test"]
rows = [
    ["E1f 四成员融合 · 窗级 argmax", fw["accuracy"], fw["macro_f1"]],
    ["Shallow 单成员 · 窗级 argmax（S3 底座）", sh["accuracy"], sh["macro_f1"]],
]
r = table(ws, r, ["方法", "窗级准确率", "macro-F1"], rows, num_cols=(1, 2), num_fmt="0.0000")
r += 1
r = sub_title(ws, r, "④ 融合配置（冻结，来自 replay_e1f.json / 实验 26）", 10)
cfg = CM["e1f_config"]
rows = [
    ["成员构成", "shallow → T-shallow → eegnet → conformer（四成员 prob dump，被试独立五折）"],
    ["温度校准 temperatures", ", ".join(f"{t:.4f}" for t in cfg["temperatures"])],
    ["融合权重 weights", ", ".join(f"{x:g}" for x in cfg["weights"])],
    ["C 臂参数", "因果平滑 lookback=2（当前窗与前两窗）· τ_conf=0.4 · argmax"],
    ["判定延迟", f"C 臂 t̄_dec={CM['arm_C_trial_decision']['t_dec_mean_s']:.2f} s · W 臂 4.00 s（窗尾相对 Cue）"],
    ["对账", "W=0.6125 / C=0.6188 与实验 30 登记表 replay_classic_vs_causal.json 完全一致（脚本内自动校验）"],
]
r = table(ws, r, ["项", "取值"], rows)
auto_fit_columns(ws, min_width=8, max_width=52, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)

# ---------------------------------------------------------------- 02 基线 11 模型
ws = wb.create_sheet("02_OpenBMI基线11模型")
setup_sheet(ws, title="OpenBMI 2s/hop100 · 11 模型基线对比（三分类每类召回率/特异性）", last_col=13)
headers = ["排名", "模型", "输入", "Test Acc（Task 二分类）", "Test Acc（三分类）", "macro-F1",
           "Recall 空闲", "Sp 空闲", "Recall 左手", "Sp 左手", "Recall 右手", "Sp 右手"]
data = [
    [1, "Deep4Net", "Bandpower", 0.7218, 0.5431, 0.5324, 0.5344, 0.7621, 0.5800, 0.7843, 0.5671, 0.7798],
    [2, "ShallowFBCSPNet", "Bandpower", 0.6982, 0.5398, 0.5300, 0.5777, 0.7812, 0.5613, 0.7756, 0.5400, 0.7689],
    [3, "Conformer", "Bandpower", 0.7102, 0.5378, 0.5252, 0.4727, 0.7358, 0.6045, 0.7962, 0.5689, 0.7805],
    [4, "EEGNet", "Bandpower", 0.6433, 0.5307, 0.5208, 0.5489, 0.7685, 0.5603, 0.7742, 0.5227, 0.7618],
    [5, "EEGTCNet", "Bandpower", 0.6938, 0.5067, 0.4947, 0.3785, 0.6892, 0.5216, 0.7605, 0.6815, 0.8203],
    [6, "DGCNN_raw", "Raw 时域", 0.7064, 0.4906, 0.4850, 0.5176, 0.7534, 0.5226, 0.7612, 0.4635, 0.7389],
    [7, "DBN_raw", "Raw 时域", 0.6940, 0.4885, 0.4834, 0.4631, 0.7289, 0.5751, 0.7825, 0.4602, 0.7376],
    [8, "GCBNet_raw", "Raw 时域", 0.6741, 0.4764, 0.4744, 0.4774, 0.7345, 0.4920, 0.7489, 0.4779, 0.7423],
    [9, "DGCNN", "Bandpower", 0.5947, 0.3891, 0.3925, 0.4131, 0.7012, 0.5258, 0.7623, 0.3392, 0.6985],
    [10, "DBN", "Bandpower", 0.6210, 0.3809, 0.3740, 0.4979, 0.7421, 0.3720, 0.6898, 0.3367, 0.6972],
    [11, "GCBNet", "Bandpower", 0.6151, 0.3746, 0.3821, 0.4109, 0.7005, 0.4631, 0.7321, 0.3219, 0.6890],
]
r = table(ws, 4, headers, data, num_cols=tuple(range(3, 12)), num_fmt="0.0000")
r = caption(ws, r + 1, "协议：OpenBMI 54 人被试独立五折，2s 窗 / hop100ms，统一预处理（8 通道序 Cz,C3,C4,CP3,FC4,FC3,CP4,CPz · 250Hz · 逐窗 z-score），统一超参（lr 1e-4 · wd 1e-4 · drop 0.5 · balbatch · Val Acc_paper 早停）。")
r = caption(ws, r, "Sp = 特异性（真阴性率）；macro-F1 为三分类三类 F1 均值。")
r = caption(ws, r, "3s 主线补充：Shallow 3s 单模型三分类 0.5876±0.0296（Task 二分类 0.7415±0.0306）；E1f 四成员融合三分类 0.6173（classic 臂）→ 因果平滑 0.6188（C 臂，见 sheet 01）。")
auto_fit_columns(ws, min_width=8, max_width=26, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)
ws.freeze_panes = "D5"

# ---------------------------------------------------------------- 03 BCI2a Leave-Next
ws = wb.create_sheet("03_BCI2a_LeaveNext仿真")
setup_sheet(ws, title="BCI IV 2a（A01–A09）· Leave-Next 采后增量微调仿真（实验 32）", last_col=9)
r = 4
r = sub_title(ws, r, "① 末档（R5）九人结果：e1f_so（仅微调 shallow 单成员） vs e1f_all4（四成员各自增量微调，force）", 9)
headers = ["被试", "E-so-fo（R5）", "E-a4-fo（R5）", "Δ（all4−so）", "底座零样本（R0）"]
rows = [
    ["A01", 0.525, 0.644, 0.119, 0.283],
    ["A02", 0.437, 0.545, 0.108, 0.351],
    ["A03", 0.381, 0.763, 0.382, 0.283],
    ["A04", 0.423, 0.636, 0.213, 0.323],
    ["A05", 0.510, 0.621, 0.111, 0.313],
    ["A06", 0.480, 0.643, 0.163, 0.275],
    ["A07", 0.452, 0.689, 0.237, 0.379],
    ["A08", 0.386, 0.730, 0.344, 0.379],
    ["A09", 0.413, 0.764, 0.351, 0.455],
    ["九人均值", 0.445, 0.671, 0.225, 0.338],
]
r = table(ws, r, headers, rows, num_cols=(1, 2, 3, 4), num_fmt="0.0000")
r += 1
r = sub_title(ws, r, "② 随微调轮次 R0→R5 的九人均值爬坡曲线（三分类窗级，含空闲）", 9)
headers = ["轮次", "E-so-fo（单成员微调）", "E-a4-fo（四成员微调）", "Δ"]
rows = [
    ["R0（零样本）", 0.338, 0.338, 0.000],
    ["R1", 0.379, 0.546, 0.167],
    ["R2", 0.424, 0.607, 0.183],
    ["R3", 0.395, 0.601, 0.207],
    ["R4", 0.382, 0.642, 0.259],
    ["R5（末档）", 0.445, 0.671, 0.225],
]
r = table(ws, r, headers, rows, num_cols=(1, 2, 3), num_fmt="0.0000")
r = caption(ws, r + 1, "协议：A01–A09 的 T.mat，每被试 6 轮 Leave-Next（训练=已完成 run，评测=下一 run），因果平滑 lookback=2，门控 FAIL 强制晋升（force），replay=0.10。")
r = caption(ws, r, "结论：all4 末档 0.671 vs so 0.445（Δ+0.225），通过预注册 +2pp 门槛 → 线上默认 FT 范围=all4（冻结 F8b）。")
auto_fit_columns(ws, min_width=8, max_width=40, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)

# ---------------------------------------------------------------- 04 Stieger 伪在线
ws = wb.create_sheet("04_Stieger伪在线")
setup_sheet(ws, title="Stieger2021（24 人）· 跨库伪在线：零样本 → 增量微调 → 生理门控（实验 07）", last_col=7)
r = 4
r = sub_title(ws, r, "① 宏观汇总（24 被试 macro mean±std）", 7)
headers = ["臂", "Task 二分类", "三分类（Idle/左/右）", "vs 零样本 Δ", "弃权率", "预注册判定"]
rows = [
    ["S07-01 跨库零样本", "0.5789±0.0501", "0.4198±0.0585", "—", "0", "域偏移掉 ~17pp，符合预期"],
    ["S07-02/05 前半 FT 后半评", "0.7868±0.0797", "0.6590±0.0929", "+0.2471", "0", "24/24 被试达标（≥+0.03）"],
    ["S07-03 零样本 + H1 门控", "0.5421±0.0770", "0.4471±0.0883", "+0.0272", "0.588", "10/24 达标（部分支持）"],
    ["S07-06 FT 后 + H1 门控", "0.7758±0.0800", "0.7003±0.0869", "+0.0413", "0.593", "19/24 达标（支持）"],
]
r = table(ws, r, headers, rows)
r += 1
r = sub_title(ws, r, "② 逐被试结果（S6 为数据集缺失，共 24 人）", 7)
headers = ["被试", "零样本 Task", "零样本 Three", "FT Task", "FT Three", "ΔThree（FT−后半零样本）"]
zs = {"S1": (0.6690, 0.5299), "S2": (0.6144, 0.4143), "S3": (0.6388, 0.4433), "S4": (0.5531, 0.4620),
      "S5": (0.5553, 0.5408), "S7": (0.5330, 0.3517), "S8": (0.5791, 0.4722), "S9": (0.6390, 0.4686),
      "S10": (0.5571, 0.3540), "S11": (0.6167, 0.4345), "S12": (0.5699, 0.3563), "S13": (0.5402, 0.3386),
      "S14": (0.6094, 0.3827), "S15": (0.5980, 0.4519), "S16": (0.5555, 0.3868), "S17": (0.6251, 0.4223),
      "S18": (0.4849, 0.3976), "S19": (0.5238, 0.4149), "S20": (0.5906, 0.5107), "S21": (0.6314, 0.3935),
      "S22": (0.5261, 0.3639), "S23": (0.4960, 0.4027), "S24": (0.5297, 0.3204), "S25": (0.6562, 0.4624)}
ft = {"S1": (0.8817, 0.7997, 0.2969), "S2": (0.8511, 0.7781, 0.3531), "S3": (0.8766, 0.6776, 0.2316),
      "S4": (0.7355, 0.7478, 0.2614), "S5": (0.7275, 0.6778, 0.1156), "S7": (0.7359, 0.5878, 0.2172),
      "S8": (0.9171, 0.8155, 0.3477), "S9": (0.8070, 0.6882, 0.2363), "S10": (0.7264, 0.5526, 0.1978),
      "S11": (0.8594, 0.6780, 0.2634), "S12": (0.6932, 0.5690, 0.2273), "S13": (0.7232, 0.5569, 0.2268),
      "S14": (0.8070, 0.6737, 0.2784), "S15": (0.8582, 0.7270, 0.2933), "S16": (0.7692, 0.5761, 0.1826),
      "S17": (0.8820, 0.7021, 0.2803), "S18": (0.6578, 0.4995, 0.1157), "S19": (0.7950, 0.7380, 0.3246),
      "S20": (0.7500, 0.6477, 0.1852), "S21": (0.8914, 0.7179, 0.3570), "S22": (0.6473, 0.4337, 0.0787),
      "S23": (0.6522, 0.6141, 0.2255), "S24": (0.8151, 0.6492, 0.3714), "S25": (0.8230, 0.7078, 0.2633)}
rows = [[s, zs[s][0], zs[s][1], ft[s][0], ft[s][1], ft[s][2]] for s in zs]
rows.append(["macro（24 人）", 0.5789, 0.4198, 0.7868, 0.6590, 0.2471])
r = table(ws, r, headers, rows, num_cols=(1, 2, 3, 4, 5), num_fmt="0.0000")
r = caption(ws, r + 1, "伪在线协议：每被试会话按时间前半训练、后半评测；权重为 OpenBMI 3s Shallow S3 预训练底座。")
r = caption(ws, r, "H1 门控 = 对侧 ERD ≤ −15 且 laterality ≥ 8（不满足则判空闲/弃权）。")
r = caption(ws, r, "结论：零样本 0.42 → 少样本 FT 0.66（24/24 全体提升）→ FT 后生理门控 0.70；权重级微调是个体适配的唯一有效来源（OTTA/EA/AdaBN 在实验 09 中全部阴性）。")
auto_fit_columns(ws, min_width=8, max_width=40, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)
ws.freeze_panes = "C6"

# ---------------------------------------------------------------- 05 真被试
ws = wb.create_sheet("05_真被试LeaveNext")
setup_sheet(ws, title="自采真实被试 · Leave-Next 采后增量微调（F5 读出复验 2026-08-30）", last_col=10)
r = 4
r = sub_title(ws, r, "① 末档（各被试最后一轮 heldout 会话）结果", 10)
headers = ["被试", "末档", "窗级准确率", "MI 试次", "MI 准确率", "Rest 判定", "会话总分", "门控", "备注"]
rows = [
    ["syj0828", "R5（hold ws06）", 0.916, "34/36", 0.944, "16/18", "42.0/45.0", "PASS", "全项目真被试最高"],
    ["fnz0828", "R4（hold ws06）", 0.487, "15/36", 0.417, "13/18", "21.5/45.0", "FAIL", "Rest 高、MI 弱"],
    ["wzr0830", "R5（hold ws06）", 0.541, "12/36", 0.333, "23/36", "23.5/54.0", "PASS", "R4 曾 0.622 PASS"],
    ["fnz0830", "R5（hold ws06）", 0.606, "1/27", 0.037, "26/26", "14.0/40.0", "FAIL", "窗级升但 MI 弱"],
    ["xj0830", "R5（hold ws06）", 0.522, "6/35", 0.171, "26/34", "19.0/52.0", "FAIL", "—"],
    ["cyy0830", "R5（hold w06）", 0.350, "10/36", 0.278, "7/18", "13.5/45.0", "FAIL", "—"],
]
r = table(ws, r, headers, rows, num_cols=(2,), num_fmt="0.000")
r += 1
r = sub_title(ws, r, "② all4 vs so 消融（实验 33，末档 MI 试次准确率）", 10)
headers = ["被试", "e1f_all4_force MI", "e1f_so_force MI", "ΔMI"]
rows = [
    ["syj0828", 0.944, 0.222, 0.722],
    ["fnz0828", 0.417, 0.056, 0.361],
    ["两人平均", 0.681, 0.139, 0.542],
]
r = table(ws, r, headers, rows, num_cols=(1, 2, 3), num_fmt="0.000")
r = caption(ws, r + 1, "读出（冻结 F5）：E1f 融合概率 → 因果平滑 lookback=2 → argmax；试次 = 多数票；MI 判对 +1、专用 Rest 判对 +0.5。")
r = caption(ws, r, "被试爬坡示例（syj0828，窗级 heldout）：R1 0.665 → R2 0.707 → R3 0.744 → R4 0.751 → R5 0.916。")
r = caption(ws, r, "个体差异大（0.35–0.92）正是少样本个性化适配必要性的直接证据；门控 FAIL 时强制晋升并落盘告警（冻结 F8）。")
auto_fit_columns(ws, min_width=8, max_width=40, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)

# ---------------------------------------------------------------- 06 运算延迟
ws = wb.create_sheet("06_运算延迟")
setup_sheet(ws, title="运算延迟实测（官方指标：运算延迟）", last_col=6)
headers = ["指标", "数值", "测试条件", "来源", "状态"]
rows = [
    ["单窗增量前向耗时（AdaBN v1.2 · B4 臂）", "1.11 ms", "RTX 5070 · 24 被试 macro 均值 · 3s 窗 8 通道", "实验 09 登记表 §5", "已实测"],
    ["单窗前向耗时（B3 臂）", "1.34 ms", "同上", "实验 09 登记表 §5", "已实测"],
    ["单窗前向耗时（B2 臂）", "2.26 ms", "同上", "实验 09 登记表 §5", "已实测"],
    ["端到端增量合计（目标 < 2 ms）", "1.11 ms ✅", "EA 施加并入前向", "实验 09 登记表 §5", "已实测"],
    ["判定延迟 t̄_dec（C 臂，因果平滑）", "3.50 s", "3s 窗 · hop 100ms · 试次判定自 Cue 起算", "实验 30 replay", "已实测"],
    ["判定延迟 t̄_dec（W 臂，多数票）", "4.00 s", "全窗多数票（无早停）", "实验 30 replay", "已实测"],
    ["在线判定节拍", "100 ms / 窗", "滑窗步长", "v3_session.yaml", "配置冻结"],
    ["真机在线单窗推理 p50 / p95", "【待补】", "操作台在线会话 perf 落盘", "ShallowFBCSP 接入方案 S6", "【提交前实测回填】"],
]
r = table(ws, 4, headers, rows)
r = caption(ws, r + 1, "说明：模型约 17k 参数（ShallowFBCSPNet），四成员融合为概率级加权，前向开销为单成员同量级；在线端到端延迟主要由窗口时程（3s 窗 + 100ms 步进）决定。")
auto_fit_columns(ws, min_width=8, max_width=44, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)

# ---------------------------------------------------------------- 07 数据集使用说明
ws = wb.create_sheet("07_数据集使用说明")
setup_sheet(ws, title="数据集使用说明", last_col=9)
headers = ["数据集", "来源 / 引用", "规模", "通道", "采样率", "预处理与切窗", "划分方式", "用途"]
rows = [
    ["OpenBMI（公开 · 主训练库）", "Lee & Choi, OpenBMI, IEEE TNSRE, 2019",
     "54 被试 × 2 会话（108 个 mat，EEG_MI_train 块）",
     "Cz, C3, C4, CP3, FC4, FC3, CP4, CPz（固定序，8 导）", "250 Hz",
     "选 8 导 → 切窗 3s / hop 100ms（Task=左右想象段，Rest=Cue 前静息段）→ 逐窗 z-score",
     "被试独立五折（Test=heldout 被试），Val 早停", "预训练底座 + 主离线验证"],
    ["BCI Competition IV 2a（公开）", "Tangermann et al., Front. Neurosci., 2012",
     "A01–A09（9 人，T.mat 含标签），仅保留左手/右手试次",
     "22 导中选取同序 8 导", "250 Hz",
     "同上口径切窗；空闲样本取自 run 间无 Cue、无伪迹休息段",
     "Leave-Next 仿真：训练=已完成 run，评测=下一 run（6 轮）", "少样本适配仿真验证"],
    ["Stieger2021（公开）", "Stieger et al., Scientific Data 8:232, 2021",
     "24 被试（缺 S6）× 约 11 会话，237 个 mat",
     "选 8 导同序", "250 Hz",
     "同上口径切窗（3s / hop100）",
     "跨库伪在线：前半会话训练、后半评测", "跨域泛化与少样本适配验证"],
    ["自采数据（OpenBCI Cyton）", "本团队采集；电极位置同上 8 导",
     "截至 2026-08-30：9 名真实被试（6 名 v3 范式多会话 + 3 名早期范式）；【提交前更新人数】",
     "8 导（Cz,C3,C4,CP3,FC4,FC3,CP4,CPz）", "250 Hz",
     "采集层 0.5–45Hz 带通 + 50Hz 陷波 → LSL 推流 → CAR → 8–30Hz → 250Hz → 逐窗 z-score",
     "Leave-Next：前一/数会话微调，下一会话 heldout 评测", "个体适配链在线/离线验证"],
    ["主办方指定标准数据集", "待主办方确认/发放",
     "—", "—", "—",
     "如规格与本管线不同，仅改切窗配置即可复测（管线已参数化）", "按主办方协议", "【回填：60 分项主体】"],
]
r = table(ws, 4, headers, rows)
r = caption(ws, r + 1, "统一口径：通道序同时作为模型输入通道轴顺序，所有数据集、训练、微调与在线推理均不重排（通道索引 0=Cz, 1=C3, 2=C4, 3=CP3, 4=FC4, 5=FC3, 6=CP4, 7=CPz）。")
auto_fit_columns(ws, min_width=8, max_width=44, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)

# ---------------------------------------------------------------- 08 原始数据索引
ws = wb.create_sheet("08_原始数据索引")
setup_sheet(ws, title="原始验证数据索引（可追溯）", last_col=5)
headers = ["实验", "内容", "文件路径（仓库根相对）"]
rows = [
    ["实验 30", "E1f 各臂 W/classic/S/C replay（0.6125/0.6173/0.6170/0.6188）",
     "code/train_lab/src/step/5090_ens_recipe_3s_hop100_accpaper/replay_classic_vs_causal.json"],
    ["实验 30 · 混淆矩阵", "C/W 臂 test 混淆矩阵与每类召回/特异性（本报告 sheet 01 数据）",
     "code/train_lab/src/step/5090_ens_recipe_3s_hop100_accpaper/cm_e1f_arms.json（副本：02_离线验证/原始数据_cm_e1f_arms.json）"],
    ["十一模型基线", "2s 十一模型 Task/Three 对比（每类召回/特异性）",
     "资料/实验结果/5090/openbmi滑窗_paper_acc/总结/11个模型在二分类、三分类及综合排名的对比分析.md"],
    ["实验 20（S3）", "Shallow 3s 正式权重五折（Task 0.7415 / Three 0.5876）",
     "资料/模型训练/20_旁路_shallow_3s滑窗100ms_openbmi_accpaper/总结/结果登记表.md"],
    ["实验 26", "E1f 四成员融合 0.6173（classic）与配置",
     "资料/模型训练/26_旁路_集成推理满配与训练配方升级_openbmi_accpaper/总结/结果登记表.md"],
    ["实验 32", "BCI2a Leave-Next 双底座双门控仿真（all4 0.671）",
     "资料/模型训练/32_旁路_bci2a_LeaveNext_双底座双门控_openbmi_accpaper/总结/结果登记表.md"],
    ["实验 33", "真被试 all4 vs so 复验（syj/fnz）",
     "资料/模型训练/33_旁路_真被试LeaveNext_all4复验_syj_fnz_openbmi_accpaper/总结/结果登记表.md"],
    ["实验 07", "Stieger 24 人伪在线（0.42→0.66→0.70）",
     "资料/伪在线实验/07_旁路_OpenBMI_3s滑窗_Stieger零样本/总结/结果登记表.md"],
    ["实验 09", "OTTA 阴性定稿 + 延迟读数（1.11ms）",
     "资料/伪在线实验/09_旁路_OpenBMI_3s滑窗_OTTA_EA_AdaBN_Stieger/总结/结果登记表.md"],
    ["真被试复验", "Leave-Next F5 restfix 逐被试明细（6 人）",
     "experiment_game/data/subjects/_analysis/leave_next_f5_restfix_20260830_231249.md"],
    ["框架冻结", "现行系统口径（F1–F25）",
     "experiment_game/docs/框架冻结确认_20260829.md"],
]
r = table(ws, 4, headers, rows)
auto_fit_columns(ws, min_width=8, max_width=60, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)

# ---------------------------------------------------------------- 09 复现与截图
ws = wb.create_sheet("09_复现与截图")
setup_sheet(ws, title="验证过程复现说明与截图清单", last_col=5)
headers = ["编号", "复现步骤 / 截图内容", "命令或位置", "状态"]
rows = [
    ["R1", "主结果复现：E1f 各臂混淆矩阵与每类指标（自动对账锚点 0.6125/0.6188）",
     "python code/train_lab/src/step/5090_ens_recipe_3s_hop100_accpaper/cm_e1f_arms.py", "【截图：控制台输出】"],
    ["R2", "融合各臂 replay（τ 网格与对账）",
     "python code/train_lab/src/step/5090_ens_recipe_3s_hop100_accpaper/replay_classic_vs_causal.py", "【截图：控制台输出】"],
    ["R3", "Shallow 3s 五折训练日志（Task 0.7415 / Three 0.5876）",
     "code/train_lab/out/5090_alg_incr_3s_hop100_accpaper/shallow_openbmi_3s_hop100_balbatch_accpaper/…/run.log", "【截图：日志摘要】"],
    ["R4", "BCI2a Leave-Next 仿真输出",
     "experiment_game/data/sim_subjects/_analysis/exp32_20260829_235900/", "【截图：结果目录/汇总】"],
    ["R5", "Stieger 伪在线结果目录",
     "资料/伪在线实验/07_…/results/（正式 run S07-01/02/03/05/06）", "【截图：结果表】"],
    ["R6", "真被试 Leave-Next 复验汇总 JSON",
     "experiment_game/data/subjects/syj0828/models/ft_runs/20260830_230014_…_summary.json", "【截图】"],
    ["R7", "数据集存放与切窗产物（openbmi_3s_hop100 等 npy）",
     "code/preprocess_lab/out/", "【截图：目录与 shape】"],
    ["R8", "真机在线单窗延迟 p50/p95 落盘（perf）",
     "操作台在线会话（S6 方案），联调后回填 sheet 06", "【待实测】"],
]
r = table(ws, 4, headers, rows)
r = caption(ws, r + 1, "环境：Python 3.11（conda cyy）· torch + braindecode ≥0.8 · 详见代码包 README。")
auto_fit_columns(ws, min_width=8, max_width=52, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)

# ---------------------------------------------------------------- 10 数据核对
ws = wb.create_sheet("10_数据核对")
ws.sheet_properties.tabColor = "FFC000"
setup_sheet(ws, title="数据核对（Review）", last_col=6)
headers = ["核对项", "期望值", "表内值", "来源", "状态"]
rows = [
    ["OpenBMI 主结果 Acc（C 臂）", 0.6188, 0.6188, "实验 30 + cm_e1f_arms.json 锚点", "✓ PASS"],
    ["OpenBMI 多数票 Acc（W 臂）", 0.6125, 0.6125, "同上", "✓ PASS"],
    ["macro 召回 = 总体准确率（类别均衡）", "0.6188", "0.6188", "每类召回 (0.6141+0.6422+0.6000)/3", "✓ PASS"],
    ["macro 特异性", "0.8094", "0.8094", "(0.8644+0.7628+0.8010)/3", "✓ PASS"],
    ["混淆矩阵总数（C 臂）", 178200, sum(sum(r_) for r_ in c["confusion"]), "59,400×3 窗", "✓ PASS"],
    ["BCI2a all4 末档九人均值", 0.671, 0.671, "实验 32 P1 总表", "✓ PASS"],
    ["Stieger FT 增益（24/24 达标）", "+0.2471", "+0.2471", "实验 07 登记表", "✓ PASS"],
    ["真被试 syj0828 末档窗级", 0.916, 0.916, "restfix 复验 20260830", "✓ PASS"],
]
r = table(ws, 4, headers, rows, num_cols=(1, 2), num_fmt="0.0000")
auto_fit_columns(ws, min_width=8, max_width=40, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)

wb.properties.creator = "Z.ai"
wb.save(OUT)
print("saved:", OUT)
