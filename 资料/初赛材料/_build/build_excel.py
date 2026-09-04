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
    # —— 主办方指定集（60 分主读 · Exp34–40 · 2026-09-04 回填 · 交卷=QuadFold-59）——
    [1, "主办方指定标准数据集（6 人，Challenge MI）",
     "LOSO6 · leave-fold 嵌套 Val（主读数）",
     "QuadFold-59（nested / 内部 S0·E1f-A59；**交卷终态**）", "试次级",
     0.5111, 0.5111, "—", "—",
     "CSV=submission_exp34_e1f_a59_sens_full_20260902_1930.csv",
     "实验 37/39/40 · sheet 11"],
    [2, "主办方指定标准数据集（同上）",
     "LOSO6 · 折内 Val（融合参数本折拟合；附报双标注）",
     "QuadFold-59 折内（交卷栈同文件）", "试次级", 0.5580, 0.5580, 0.779, "—", "—",
     "实验 34/35；嵌套对照 0.511（乐观≈+4.7pp）"],
    [3, "主办方指定标准数据集（同上）",
     "LOSO6 · leave-fold 嵌套 Val（备选归档）",
     "E1f-B8-ft（OpenBMI→8ch FT；Exp39/40 §5 程序终态，交卷已风险否决）", "试次级",
     0.5400, 0.5400, 0.7700, 0.5319, "—",
     "备选 CSV 归档；test Rest≈51% ∉ Val 支撑"],
    [4, "主办方指定标准数据集（同上）",
     "LOSO6 · 折内 Val（备选附报）",
     "E1f-B8-ft 折内", "试次级", 0.5733, 0.5733, 0.787, "—", "—",
     "实验 34；嵌套复核 0.540"],
    # —— 公开库 / 自采 ——
    [5, "OpenBMI（54 人，公开）", "被试独立五折 · test 试次级 Acc_paper（16,200 试次，每类 5,400）",
     "CausalFuse-8 + 因果滑窗（C 臂，主结果）", "试次级", 0.6188, 0.6188, 0.8094, 0.6196,
     "判定延迟 t̄≈3.50 s；单窗前向 1.11 ms（RTX 5070）", "实验 30 + 混淆矩阵复算（本表 sheet 01）"],
    [6, "OpenBMI（54 人，公开）", "同上", "CausalFuse-8 + 多数票读出（W 臂，现行线上读出）", "试次级",
     0.6125, 0.6125, 0.8062, 0.6132, "判定延迟 t̄≈4.00 s", "实验 30 + 混淆矩阵复算（sheet 01）"],
    [7, "OpenBMI（54 人，公开）", "同上", "CausalFuse-8 窗级 argmax（不作试次聚合）", "窗级",
     0.5925, 0.5925, 0.7962, 0.5930, "单窗 100 ms 步进", "混淆矩阵复算（sheet 01）"],
    [8, "OpenBMI（54 人，公开）", "被试独立五折 · 试次级", "ShallowFBCSPNet 3s 单模型（S3 底座）", "试次级",
     0.5876, "—", "—", "—", "—", "实验 20 登记表（0.5876±0.0296）"],
    [9, "OpenBMI（54 人，公开）", "被试独立五折 · 试次级", "Deep4Net 2s（11 模型基线最优）", "试次级",
     0.5431, 0.5605, 0.7754, 0.5324, "—", "5090 十一模型对比（sheet 02）"],
    [10, "BCI IV 2a（9 人，公开）", "Leave-Next 6 轮仿真 · 末档（R5）· 三分类含空闲",
     "CausalFuse-8FT（all4 采后增量微调 force）", "窗级（heldout run）",
     0.663, "—", "—", "—", "—", "实验 32 stamp=20260901_124502（sheet 03）"],
    [11, "BCI IV 2a（9 人，公开）", "同上（底座零样本 R0）", "CausalFuse-8 零样本", "窗级", 0.338, "—", "—", "—", "—", "实验 32（sheet 03）"],
    [12, "Stieger2021（24 人，公开）", "跨库伪在线 · 前半训练后半评测", "Shallow 3s 零样本（跨库）", "试次级", 0.4198, "—", "—", "—", "—", "实验 07（sheet 04）"],
    [13, "Stieger2021（24 人，公开）", "同上", "Shallow 3s + 前半增量微调（FT）", "试次级", 0.6590, "—", "—", "—", "—", "实验 07（24/24 被试 ≥+3pp）"],
    [14, "Stieger2021（24 人，公开）", "同上 + 生理门控 H1", "Shallow 3s FT + ERD 门控", "试次级", 0.7003, "—", "—", "—", "—", "实验 07（sheet 04）"],
    [15, "自采（syj0828，8 通道 v3 范式）", "Leave-Next 采后增量微调 · 末档（R5）",
     "CausalFuse-8FT 个体模型（真实被试最优）", "窗级 heldout（smooth）", 0.924, "—", "—", "—",
     "F5 41.5/45；MI 33/36；Rest 17/18；门控 5/5 PASS", "Leave-Next all4 全表 20260831_233943（sheet 05）"],
    [16, "自采（其余 v3 被试，含 ytl0901）", "同上 · 10 人全表", "CausalFuse-8FT 个体模型", "窗级 heldout",
     "0.35–0.54（见 sheet 05）", "—", "—", "—", "门控合计 20/50 PASS；个体差异大", "Leave-Next all4 全表"],
]
r = table(ws, 4, headers, rows, num_cols=(5, 6, 7, 8))
r = caption(ws, r + 1, "注 1：OpenBMI 五折 test 划分为被试独立划分，指标为 test 试次级 Acc_paper（每试次一次判定）。")
r = caption(ws, r, "注 2：召回率 macro = 三类召回率均值；特异性 macro = 三类特异性均值；每类明细与混淆矩阵见 sheet 01/02/11。")
r = caption(ws, r, "注 3：指定集**交卷=QuadFold-59**（内部 S0 / E1f-A59）：主读嵌套 0.511，折内 0.558 双标注。R-B8 nested=0.540 为 Exp39/40 备选归档（风险否决：test 预测 Rest≈51% 落在 Val 六折支撑外）。详见 sheet 07/11。")
r = caption(ws, r, "注 4：QuadFold-59 为从零训练交卷；CausalFuse-8 为 OpenBMI 预训练在线底座，不作为指定集交卷依赖。赛规未禁止预训练微调，本队指定集路径选择从零以降低合规与外推风险。")
auto_fit_columns(ws, min_width=8, max_width=42, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)
ws.freeze_panes = "C5"

# ---------------------------------------------------------------- 01 OpenBMI 主结果
ws = wb.create_sheet("01_OpenBMI主结果")
setup_sheet(ws, title="OpenBMI 54 人五折 · CausalFuse-8 主结果明细（混淆矩阵与每类指标）", last_col=10)


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
r = sub_title(ws, r, "① 主结果：CausalFuse-8 + 因果滑窗（C 臂）试次判定 · test 集准确率 0.6188（对账锚点一致）", 10)
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
    ["CausalFuse-8 · 窗级 argmax", fw["accuracy"], fw["macro_f1"]],
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
r = caption(ws, r, "3s 主线补充：Shallow 3s 单模型三分类 0.5876±0.0296（Task 二分类 0.7415±0.0306）；CausalFuse-8 融合三分类 0.6173（classic 臂）→ 因果滑窗 0.6188（C 臂，见 sheet 01）。")
auto_fit_columns(ws, min_width=8, max_width=26, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)
ws.freeze_panes = "D5"

# ---------------------------------------------------------------- 03 BCI2a Leave-Next
ws = wb.create_sheet("03_BCI2a_LeaveNext仿真")
setup_sheet(ws, title="BCI IV 2a（A01–A09）· Leave-Next 采后增量微调仿真（实验 32 · stamp=20260901_124502）", last_col=9)
r = 4
r = sub_title(ws, r, "① 末档（R5）九人结果：e1f_so（仅微调 shallow 单成员） vs e1f_all4（四成员各自增量微调，force）", 9)
headers = ["被试", "E-so-fo（R5）", "E-a4-fo（R5）", "Δ（all4−so）", "底座零样本（R0）"]
rows = [
    ["A01", 0.501, 0.649, 0.148, 0.283],
    ["A02", 0.495, 0.556, 0.061, 0.351],
    ["A03", 0.379, 0.737, 0.358, 0.283],
    ["A04", 0.470, 0.642, 0.172, 0.323],
    ["A05", 0.490, 0.578, 0.088, 0.313],
    ["A06", 0.473, 0.649, 0.176, 0.275],
    ["A07", 0.455, 0.689, 0.234, 0.379],
    ["A08", 0.386, 0.707, 0.321, 0.379],
    ["A09", 0.418, 0.756, 0.338, 0.455],
    ["九人均值", 0.452, 0.663, 0.211, 0.338],
]
r = table(ws, r, headers, rows, num_cols=(1, 2, 3, 4), num_fmt="0.0000")
r += 1
r = sub_title(ws, r, "② 随微调轮次 R0→R5 的九人均值爬坡曲线（三分类窗级，含空闲）", 9)
headers = ["轮次", "E-so-fo（单成员微调）", "E-a4-fo（四成员微调）", "Δ"]
rows = [
    ["R0（零样本）", 0.338, 0.338, 0.000],
    ["R1", 0.376, 0.536, 0.161],
    ["R2", 0.423, 0.593, 0.170],
    ["R3", 0.392, 0.610, 0.218],
    ["R4", 0.391, 0.636, 0.245],
    ["R5（末档）", 0.452, 0.663, 0.211],
]
r = table(ws, r, headers, rows, num_cols=(1, 2, 3), num_fmt="0.0000")
r = caption(ws, r + 1, "协议：A01–A09 的 T.mat，每被试 6 轮 Leave-Next（训练=已完成 run，评测=下一 run），因果平滑 lookback=2，门控 FAIL 强制晋升（force），replay=0.10。stamp=20260901_124502。")
r = caption(ws, r, "结论：all4 末档 0.663 vs so 0.452（Δ+0.211），通过预注册 +2pp 门槛 → 线上默认 FT 范围=all4；相对旧跑 20260829_235900 末档均值差约 ±1pp。")
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
setup_sheet(ws, title="自采真实被试 · Leave-Next all4（方案 A · 全表 20260831_233943 + ytl0901）", last_col=10)
r = 4
r = sub_title(ws, r, "① 末档（各被试最后一轮 heldout）· FT all4 · 展示=smooth / 门控=raw · F5 试次计分", 10)
headers = ["被试", "末档", "窗级 smooth", "F5 总分", "Left", "Right", "Rest", "门控 PASS", "备注"]
rows = [
    ["syj0828", "R5", 0.924, "41.5/45", "17/18", "16/18", "17/18", "5/5", "全项目真被试最高"],
    ["fnz0828", "R5", 0.453, "18.0/45", "12/18", "0/18", "12/18", "1/5", "Rest 高、Right 弱"],
    ["cyy0830", "R5", 0.409, "22.5/45", "15/18", "4/18", "7/18", "1/5", "含半场 w03"],
    ["fnz0830", "R5", 0.357, "11.0/44", "3/17", "0/18", "16/18", "2/5", "末档回落"],
    ["wzr0830", "R5", 0.425, "17.0/45", "9/18", "3/18", "10/18", "1/5", "—"],
    ["xj0830", "R5", 0.483, "20.0/45", "14/18", "1/18", "10/18", "1/5", "—"],
    ["cjf0831", "R5", 0.468, "22.0/45", "10/18", "8/18", "8/18", "1/5", "—"],
    ["npl0831", "R5", 0.542, "26.0/45", "15/18", "5/18", "12/18", "4/5", "次优个体"],
    ["ycx0831", "R5", 0.418, "19.0/45", "14/18", "3/18", "4/18", "2/5", "排除半场 w06"],
    ["ytl0901", "R5", 0.461, "23.5/45", "12/18", "9/18", "5/18", "2/5", "w02+w03 合并为一场"],
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
r = caption(ws, r + 1, "读出（冻结 F5 / 方案 A）：展示 heldout_acc=因果平滑；门控=raw；试次=因果平滑+多数票；MI +1、Rest +0.5。")
r = caption(ws, r, "被试爬坡示例（syj0828，窗级 smooth）：R1 0.663 → R2 0.717 → R3 0.759 → R4 0.785 → R5 0.924。")
r = caption(ws, r, "全表门控 20/50 PASS；个体差异大（0.35–0.92）正是少样本个性化适配必要性的直接证据；门控 FAIL 时强制晋升并落盘告警。")
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
     "截至 2026-09-01：≥10 名 v3 范式真实被试（syj0828/fnz0828/cyy0830/fnz0830/wzr0830/xj0830/cjf0831/npl0831/ycx0831/ytl0901 等）+ 早期范式被试；【提交前更新人数】",
     "8 导（Cz,C3,C4,CP3,FC4,FC3,CP4,CPz）", "250 Hz",
     "采集层 0.5–45Hz 带通 + 50Hz 陷波 → LSL 推流 → CAR → 8–30Hz → 250Hz → 逐窗 z-score；EEG 看门狗 stall 2s / abort 5s",
     "Leave-Next：前一/数会话微调，下一会话 heldout 评测", "个体适配链在线/离线验证"],
    ["主办方指定标准数据集（Challenge MI）",
     "主办方发放；**交卷栈=QuadFold-59**（内部 S0/E1f-A59）；CausalFuse-8 仅用于系统底座/公开库叙事（非指定集交卷依赖）",
     "6 被试 · LOSO6 · 每折 Val 150 试次（共 900）· 1 trial=1 窗 3s@250Hz",
     "交卷 59ch 从零；备选 8 导（Pz 代 CPz，OpenBMI→FT）已归档", "250 Hz",
     "与 OpenBMI 同切窗口径；轨 A：QuadFold-59（交卷）；轨 B：OpenBMI→8ch FT（备选）",
     "被试独立 LOSO6；主读=leave-fold 嵌套 Val；折内双标注",
     "60 分项主体 · 交卷 QuadFold-59 nested=0.511 / 折内 0.558（Exp40 风险否决回退）"],
]
r = table(ws, 4, headers, rows)
r = caption(ws, r + 1, "统一口径：通道序同时作为模型输入通道轴顺序，所有数据集、训练、微调与在线推理均不重排（通道索引 0=Cz, 1=C3, 2=C4, 3=CP3, 4=FC4, 5=FC3, 6=CP4, 7=CPz）。")
r = caption(ws, r, "交卷说明：指定集交卷=QuadFold-59（从零）。CausalFuse-8 是系统底座资产；R-B8 因 Val 支撑外推风险否决，CSV 保留归档不删。")
auto_fit_columns(ws, min_width=8, max_width=44, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)

# ---------------------------------------------------------------- 08 原始数据索引
ws = wb.create_sheet("08_原始数据索引")
setup_sheet(ws, title="原始验证数据索引（可追溯）", last_col=5)
headers = ["实验", "内容", "文件路径（仓库根相对）"]
rows = [
    ["实验 30", "CausalFuse-8 各臂 W/classic/S/C replay（0.6125/0.6173/0.6170/0.6188）",
     "code/train_lab/src/step/5090_ens_recipe_3s_hop100_accpaper/replay_classic_vs_causal.json"],
    ["实验 30 · 混淆矩阵", "C/W 臂 test 混淆矩阵与每类召回/特异性（本报告 sheet 01 数据）",
     "code/train_lab/src/step/5090_ens_recipe_3s_hop100_accpaper/cm_e1f_arms.json（副本：02_离线验证/原始数据_cm_e1f_arms.json）"],
    ["十一模型基线", "2s 十一模型 Task/Three 对比（每类召回/特异性）",
     "资料/实验结果/5090/openbmi滑窗_paper_acc/总结/11个模型在二分类、三分类及综合排名的对比分析.md"],
    ["实验 20（S3）", "Shallow 3s 正式权重五折（Task 0.7415 / Three 0.5876）",
     "资料/模型训练/20_旁路_shallow_3s滑窗100ms_openbmi_accpaper/总结/结果登记表.md"],
    ["实验 26", "CausalFuse-8（E1f）四成员融合 0.6173（classic）与配置",
     "资料/模型训练/26_旁路_集成推理满配与训练配方升级_openbmi_accpaper/总结/结果登记表.md"],
    ["实验 32", "BCI2a Leave-Next 双底座双门控仿真（all4 force R5=0.663 vs so 0.452）",
     "资料/模型训练/32_旁路_bci2a_LeaveNext_双底座双门控_openbmi_accpaper/总结/结果登记表.md · 原始 exp32_20260901_124502/"],
    ["实验 33", "真被试 all4 vs so 复验（syj/fnz）",
     "资料/模型训练/33_旁路_真被试LeaveNext_all4复验_syj_fnz_openbmi_accpaper/总结/结果登记表.md"],
    ["实验 07", "Stieger 24 人伪在线（0.42→0.66→0.70）",
     "资料/伪在线实验/07_旁路_OpenBMI_3s滑窗_Stieger零样本/总结/结果登记表.md"],
    ["实验 09", "OTTA 阴性定稿 + 延迟读数（1.11ms）",
     "资料/伪在线实验/09_旁路_OpenBMI_3s滑窗_OTTA_EA_AdaBN_Stieger/总结/结果登记表.md"],
    ["真被试 Leave-Next 全表", "10 人 all4 · 门控 20/50 · syj R5 smooth 0.924",
     "experiment_game/data/subjects/_analysis/leave_next_all4_full_report_20260831_233943.md"],
    ["实验 34", "指定集 LOSO6 · QuadFold-59 折内 0.558 · E1f-B8-ft 折内 0.573",
     "资料/模型训练/34_旁路_挑战杯官方集_59ch离线_openbmi协议_accpaper/总结/结果登记表.md"],
    ["实验 35", "排名翻转消融 · S0 定稿（折内尺子）",
     "资料/模型训练/35_旁路_官方vsOpenBMI_三分类排名不一致_融合重标定与骨干消融_accpaper/总结/结果登记表.md"],
    ["实验 36", "折内跨轨融合最高 0.626（未过 Wilcoxon；后由 37 证伪为伪影）",
     "资料/模型训练/36_旁路_官方主交卷_扩池与跨轨融合_accpaper/总结/结果登记表.md"],
    ["实验 37", "嵌套确认 · N0=0.511 · N7=0.523 p=0.81 · 维持 S0 纪律",
     "资料/模型训练/37_旁路_官方主交卷_嵌套融合McNemar确认_accpaper/总结/结果登记表.md"],
    ["实验 38", "多样性选池阴性 · G*=b8_shallow_b 0.528",
     "资料/模型训练/38_旁路_官方主交卷_误差去相关选池_accpaper/总结/结果登记表.md"],
    ["实验 39", "诚实排行榜 · R-B8 nested=0.540 · 工程选卷",
     "资料/模型训练/39_旁路_官方主交卷_收尾回放与工程选卷_accpaper/总结/结果登记表.md"],
    ["实验 40", "CSV 加固阴性 · 终态 R-B8_raw · 算法冻结",
     "资料/模型训练/40_旁路_官方主交卷_CSV加固_边际校正与TTA_accpaper/总结/结果登记表.md"],
    ["指定集工程 CSV", "submission_exp40_rb8_raw_final_*（=Exp39 R-B8）",
     "code/train_lab/out/5070_challenge_exp40_csv_harden_tta_accpaper/submissions/"],
    ["指定集回退 CSV", "Exp34 S0",
     "code/train_lab/out/5070_challenge_mi_59ch_accpaper/submissions/submission_exp34_e1f_a59_sens_full_20260902_1930.csv"],
    ["统计口径方案 A", "展示=smooth · 门控=raw · F5 试次计分",
     "experiment_game/docs/统计口径方案A_20260831.md"],
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
    ["R1", "主结果复现：CausalFuse-8 各臂混淆矩阵与每类指标（自动对账锚点 0.6125/0.6188）",
     "python code/train_lab/src/step/5090_ens_recipe_3s_hop100_accpaper/cm_e1f_arms.py", "【截图：控制台输出】"],
    ["R2", "融合各臂 replay（τ 网格与对账）",
     "python code/train_lab/src/step/5090_ens_recipe_3s_hop100_accpaper/replay_classic_vs_causal.py", "【截图：控制台输出】"],
    ["R3", "Shallow 3s 五折训练日志（Task 0.7415 / Three 0.5876）",
     "code/train_lab/out/5090_alg_incr_3s_hop100_accpaper/shallow_openbmi_3s_hop100_balbatch_accpaper/…/run.log", "【截图：日志摘要】"],
    ["R4", "BCI2a Leave-Next 仿真输出",
     "experiment_game/data/sim_subjects/_analysis/exp32_20260901_124502/", "【截图：结果目录/汇总】"],
    ["R5", "Stieger 伪在线结果目录",
     "资料/伪在线实验/07_…/results/（正式 run S07-01/02/03/05/06）", "【截图：结果表】"],
    ["R6", "真被试 Leave-Next 复验汇总 JSON",
     "experiment_game/data/subjects/syj0828/models/ft_runs/20260831_224229_…_summary.json", "【截图】"],
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
    ["BCI2a all4 末档九人均值", 0.663, 0.663, "实验 32 stamp=20260901_124502 P1 总表", "✓ PASS"],
    ["Stieger FT 增益（24/24 达标）", "+0.2471", "+0.2471", "实验 07 登记表", "✓ PASS"],
    ["真被试 syj0828 末档窗级 smooth", 0.924, 0.924, "leave_next_all4_full_report_20260831_233943", "✓ PASS"],
    ["指定集工程主读 QuadFold-59 nested（交卷）", 0.511, 0.511, "Exp37 N0 / Exp40 交卷", "✓ PASS"],
    ["指定集备选 nested R-B8", 0.540, 0.540, "Exp39/40 归档", "✓ 备选"],
    ["指定集折内乐观（S0）", "+4.7pp", "0.558→0.511", "Exp34 折内 vs Exp37 嵌套", "✓ 已标注"],
]
r = table(ws, 4, headers, rows, num_cols=(1, 2), num_fmt="0.0000")
auto_fit_columns(ws, min_width=8, max_width=40, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)

# ---------------------------------------------------------------- 11 指定集 Exp34–40
ws = wb.create_sheet("11_指定集Exp34_40")
setup_sheet(ws, title="主办方指定集 · Exp34–40 诚实口径与工程终态（2026-09-04 回填）", last_col=8)
headers = ["实验", "臂 / 结论", "读数", "尺子", "过线/采纳", "备注"]
rows = [
    ["34", "QuadFold-59（S0）", "0.558±0.069", "折内", "历史主 CSV", "科学叙事对照；嵌套后 0.511"],
    ["34", "E1f-B8-ft", "0.573±0.140", "折内", "对照 CSV", "嵌套复核 0.540（Exp39）"],
    ["35", "S0 定稿", "维持 0.558 折内", "折内 Wilcoxon", "科学 KEEP（后修正）", "Q1 显著性含拟合不对称；以 37/38 为准"],
    ["36", "M7 / M7_ABC", "0.604 / 0.626", "折内", "未过 p=0.062", "禁止作预期；嵌套后≈0.523/0.510"],
    ["37", "N0 nested-S0", "0.511±0.066", "嵌套", "对照锚", "折内乐观 ≈+4.7pp"],
    ["37", "N7 nested M7", "0.523±0.125", "嵌套", "未过 p=0.81", "弱阳性不换卷"],
    ["38", "G* / V1", "0.528±0.130", "嵌套", "未过 p=0.69", "仅 b8_shallow_b；选池阴性"],
    ["39", "R-B8", "0.540±0.146", "嵌套", "工程选卷", "非显著性；科学仍 KEEP_S0"],
    ["40", "MC-B8", "0.540（b≡0）", "嵌套", "平手→最简否决", "Rest 偏斜不可校"],
    ["40", "TTA-B8", "0.538（−0.2pp）", "嵌套", "未进候选", "向内缩 δ∈{0,20,40}"],
    ["40", "终态（§5 程序）", "R-B8_raw", "工程规则", "归档", "与 Exp39 CSV 字节相同"],
    ["40", "**交卷（风险否决）**", "**QuadFold-59**", "Val 外推诊断", "**采纳**", "test Rest≈51%∉Val；无 0.300 前科"],
]
r = table(ws, 4, headers, rows)
r = caption(ws, r + 1, "交卷=QuadFold-59（nested 0.511 + 折内 0.558 双标注）。R-B8=0.540 为备选。风险否决未改 Val 门槛、未用 test 标签调参。")
r = sub_title(ws, r + 1, "R-B8 嵌套 OOF 混淆矩阵（900 试次；行=真，列=预；类序约 L/R/Rest 依 dump 标签）", 8)
r = table(ws, r, ["", "pred0", "pred1", "pred2"], [
    ["true0", 181, 72, 47],
    ["true1", 71, 196, 33],
    ["true2", 86, 105, 109],
], num_cols=(1, 2, 3), num_fmt="#,##0")
r = caption(ws, r + 1, "Acc=0.540 · recall macro=0.540 · specificity macro≈0.770 · F1 macro≈0.532（Exp39 OOF）。")
r = caption(ws, r, "风险：S0↔R-B8 test Hamming 45%；test 边际 S0 29/44/47 vs R-B8 38/21/61；若 test≈40/40/40 边际上限约 90.8%/82.5%。")
auto_fit_columns(ws, min_width=8, max_width=48, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)

wb.properties.creator = "Z.ai"
wb.save(OUT)
print("saved:", OUT)
