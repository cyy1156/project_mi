"""结果登记表：参数量统计 + left<->right 混淆率 + S0 基线对比 + 成功/失败判定 + 自动写入。

严格按方案 section3（成功/失败/停机规则）和 section6（必报指标模板）。

主入口：update_registry() —— 在 run_baseline_main 末尾调用。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ================================================================
# 1. 参数量统计
# ================================================================

def count_params(model) -> int:
    """统计模型可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ================================================================
# 2. left<->right 混淆率提取
# ================================================================

def extract_lr_confusion_rate(cm) -> float:
    """从 3x3 混淆矩阵提取 left<->right 混淆率。

    cm 格式（行=真实, 列=预测）:
      [[idle->idle, idle->left, idle->right],
       [left->idle, left->left, left->right],
       [right->idle, right->left, right->right]]

    left<->right 混淆率 = (left->right + right->left) / (total_left + total_right)
    """
    cm = np.asarray(cm, dtype=float)
    if cm.shape != (3, 3):
        return float("nan")
    left_total = cm[1].sum()
    right_total = cm[2].sum()
    lr = cm[1, 2]  # left->right
    rl = cm[2, 1]  # right->left
    denom = left_total + right_total
    if denom == 0:
        return float("nan")
    return float((lr + rl) / denom)


# ================================================================
# 3. S0 基线加载
# ================================================================

def find_s0_meta(out_root_tag_dir: Path) -> dict | None:
    """在 out 目录中找最新的 S0 正式五折 meta.json。

    只接受 max_folds 缺失或 ==5（正式五折）。
    """
    s0_base = out_root_tag_dir / "shallow_s0_net_enhance_three"
    if not s0_base.exists():
        return None

    best_stamp = ""
    best_meta = None

    for meta_path in s0_base.rglob("meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # 只取正式五折
        task_sum = meta.get("task") or {}
        max_folds = task_sum.get("max_folds", 5)
        if max_folds not in (0, 5):
            continue
        # 只取 fast 模式（正式出数）
        if meta.get("train_mode") != "fast":
            continue
        stamp = meta.get("stamp", "")
        if stamp > best_stamp:
            best_stamp = stamp
            best_meta = meta

    return best_meta


# ================================================================
# 4. Delta 计算
# ================================================================

def compute_deltas(
    cur_task: dict | None,
    cur_three: dict | None,
    s0_task: dict | None,
    s0_three: dict | None,
) -> dict:
    """计算当前臂 vs S0 的 delta。"""
    d: dict = {}

    # Three deltas
    if cur_three and s0_three:
        d["d_three_acc_paper"] = (
            float(cur_three["test_acc_paper_mean"])
            - float(s0_three["test_acc_paper_mean"])
        )
        d["d_three_f1_macro"] = (
            float(cur_three.get("test_f1_macro_maj_mean", 0))
            - float(s0_three.get("test_f1_macro_maj_mean", 0))
        )
        d["d_three_balacc_maj"] = (
            float(cur_three["test_balacc_maj_mean"])
            - float(s0_three["test_balacc_maj_mean"])
        )
        # recall deltas (fold-level mean)
        for cls in ("idle", "left", "right"):
            cur_r = _fold_recall_mean(cur_three, cls)
            s0_r = _fold_recall_mean(s0_three, cls)
            if cur_r is not None and s0_r is not None:
                d[f"d_recall_{cls}"] = cur_r - s0_r
            else:
                d[f"d_recall_{cls}"] = None
        # left<->right confusion rate delta
        cur_lr = _fold_lr_confusion_mean(cur_three)
        s0_lr = _fold_lr_confusion_mean(s0_three)
        if cur_lr is not None and s0_lr is not None:
            d["d_lr_confusion"] = cur_lr - s0_lr
        else:
            d["d_lr_confusion"] = None
    else:
        d["d_three_acc_paper"] = None
        d["d_three_f1_macro"] = None
        d["d_three_balacc_maj"] = None
        d["d_lr_confusion"] = None

    # Task deltas
    if cur_task and s0_task:
        d["d_task_acc_paper"] = (
            float(cur_task["test_acc_paper_mean"])
            - float(s0_task["test_acc_paper_mean"])
        )
        d["d_task_balacc_maj"] = (
            float(cur_task["test_balacc_maj_mean"])
            - float(s0_task["test_balacc_maj_mean"])
        )
    else:
        d["d_task_acc_paper"] = None
        d["d_task_balacc_maj"] = None

    return d


def _fold_recall_mean(summary: dict, cls: str) -> float | None:
    """从 folds 的 test_trial_metrics 取 recall_{cls} 的均值。"""
    vals = []
    for f in summary.get("folds", []):
        m = f.get("test_trial_metrics") or {}
        v = m.get(f"recall_{cls}")
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return float(np.mean(vals))


def _fold_lr_confusion_mean(summary: dict) -> float | None:
    """从 folds 的 test_trial_metrics.cm 取 left<->right 混淆率均值。"""
    vals = []
    for f in summary.get("folds", []):
        m = f.get("test_trial_metrics") or {}
        cm = m.get("cm")
        if cm:
            r = extract_lr_confusion_rate(cm)
            if not np.isnan(r):
                vals.append(r)
    if not vals:
        return None
    return float(np.mean(vals))


# ================================================================
# 5. 成功/失败/护栏判定 (section3.1 + section3.2)
# ================================================================

def determine_conclusion(
    deltas: dict,
    cur_three: dict | None,
    s0_three: dict | None,
    is_smoke: bool = False,
) -> str:
    """按方案 section3 判定结论。

    section3.1 主成功线（Three）:
      弱成功: dAcc_paper >= +0.010 且 dF1-macro >= 0
      达标成功: dAcc_paper >= +0.015
      强成功: dAcc_paper >= +0.020，或 Acc>=+0.015 且 left<->right 混淆明显下降

    section3.2 Task 护栏:
      dTask Acc_paper >= -0.010；若 Three 达标但 Task 掉 >1pp -> Three专用成功
      BalAcc_maj 显著变差（偏 idle）-> 阴性

    Returns: 弱成功 / 达标 / 强成功 / 阴性 / 护栏失败 / Three专用成功 / 无S0基线 / 冒烟(不判定)
    """
    if is_smoke:
        return "冒烟(不判定)"

    d_three = deltas.get("d_three_acc_paper")
    d_f1 = deltas.get("d_three_f1_macro")
    d_task = deltas.get("d_task_acc_paper")
    d_bal = deltas.get("d_three_balacc_maj")
    d_lr = deltas.get("d_lr_confusion")

    if d_three is None or s0_three is None:
        return "无S0基线"

    # section3.2 BalAcc_maj 显著变差（偏 idle）-> 阴性，即使 Acc_paper 微升
    if d_bal is not None and d_bal < -0.010:
        return "阴性"

    # section3.2 Task 护栏
    task_guardrail_ok = True
    if d_task is not None and d_task < -0.010:
        task_guardrail_ok = False

    # section3.1 主成功线
    is_weak = d_three >= 0.010 and (d_f1 is not None and d_f1 >= 0)
    is_pass = d_three >= 0.015
    is_strong = d_three >= 0.020

    # 强成功补充：dAcc>=0.015 且 left<->right 混淆明显下降
    if not is_strong and is_pass:
        # "混淆明显下降" = d_lr_confusion < -0.010 (混淆率下降 1pp 以上)
        if d_lr is not None and d_lr < -0.010:
            is_strong = True

    if is_strong:
        return "Three专用成功" if not task_guardrail_ok else "强成功"
    if is_pass:
        return "Three专用成功" if not task_guardrail_ok else "达标"
    if is_weak:
        return "Three专用成功" if not task_guardrail_ok else "弱成功"

    # Three 无弱成功，检查 Task 护栏是否也失败
    if not task_guardrail_ok:
        return "护栏失败"

    return "阴性"


# ================================================================
# 6. 结果登记表写入 (section6 模板)
# ================================================================

REGISTRY_HEADERS = [
    "臂",
    "Task Acc_paper",
    "Task BalAcc_maj",
    "Three Acc_paper",
    "Three F1-macro",
    "Three BalAcc_maj",
    "Recall idle",
    "Recall left",
    "Recall right",
    "L<->R混淆率",
    "窗级BalAcc",
    "参数量",
    "相对S0",
    "dThree vs S0",
    "结论",
]


def _ms(d: dict | None, key: str) -> str:
    """格式化 mean+/-std。"""
    if not d:
        return "-"
    mean = d.get(f"{key}_mean")
    std = d.get(f"{key}_std")
    if mean is None:
        return "-"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f}+/-{std:.4f}"


def _fold_field_ms(folds: list[dict], field: str) -> str:
    """从 folds 的 test_trial_metrics 取某字段的 mean+/-std。"""
    vals = []
    for f in folds:
        m = f.get("test_trial_metrics") or {}
        v = m.get(field)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return "-"
    a = np.asarray(vals)
    return f"{a.mean():.4f}+/-{a.std():.4f}"


def _fold_lr_confusion_ms(folds: list[dict]) -> str:
    """从 folds 的 test_trial_metrics.cm 计算 left<->right 混淆率 mean+/-std。"""
    vals = []
    for f in folds:
        m = f.get("test_trial_metrics") or {}
        cm = m.get("cm")
        if cm:
            r = extract_lr_confusion_rate(cm)
            if not np.isnan(r):
                vals.append(r)
    if not vals:
        return "-"
    a = np.asarray(vals)
    return f"{a.mean():.4f}+/-{a.std():.4f}"


_REGISTRY_MARKER = "dThree vs S0"


def ensure_registry_table(registry_path: Path) -> None:
    """确保登记表存在且包含标准表头。

    - 文件不存在：创建带表头的新表
    - 文件存在但无标准表头：追加标准表头段
    - 文件存在且有标准表头：不做任何操作
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        "",
        "## 自动登记表（方案 section3 + section6）",
        "",
        "> 每臂一行；正式五折 mean+/-std；dThree vs S0 为主决策列。",
        "> 成功线：弱成功 dAcc>=+0.010 | 达标 dAcc>=+0.015 | 强成功 dAcc>=+0.020",
        "> Task护栏：dTask Acc_paper >= -0.010",
        "> 带 * 标记为冒烟运行（非正式五折）",
        "",
        "| " + " | ".join(REGISTRY_HEADERS) + " |",
        "|" + "|".join(["---"] * len(REGISTRY_HEADERS)) + "|",
        "",
    ]

    if not registry_path.exists():
        full = [
            "# 结果登记表（09 旁路 - Shallow 网络结构增强 - Three 主攻）",
            "",
        ] + header_lines
        registry_path.write_text("\n".join(full), encoding="utf-8")
        return

    # 检查是否已有标准表头
    content = registry_path.read_text(encoding="utf-8")
    if _REGISTRY_MARKER in content:
        return  # 已有表头，不重复添加

    # 文件存在但无标准表头 → 追加
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")


def append_registry_row(
    registry_path: Path,
    arm_name: str,
    task_summary: dict | None,
    three_summary: dict | None,
    n_params: int,
    s0_n_params: int | None,
    deltas: dict,
    conclusion: str,
) -> None:
    """向登记表追加一行。"""
    ensure_registry_table(registry_path)

    folds_three = three_summary.get("folds", []) if three_summary else []
    folds_task = task_summary.get("folds", []) if task_summary else []

    # 判断是否冒烟
    is_smoke = False
    if task_summary:
        mf = task_summary.get("max_folds", 5)
        if mf not in (0, 5):
            is_smoke = True

    arm_display = f"{arm_name}*" if is_smoke else arm_name

    d_three = deltas.get("d_three_acc_paper")

    row = [
        arm_display,
        _ms(task_summary, "test_acc_paper"),
        _ms(task_summary, "test_balacc_maj"),
        _ms(three_summary, "test_acc_paper"),
        _ms(three_summary, "test_f1_macro_maj"),
        _ms(three_summary, "test_balacc_maj"),
        _fold_field_ms(folds_three, "recall_idle"),
        _fold_field_ms(folds_three, "recall_left"),
        _fold_field_ms(folds_three, "recall_right"),
        _fold_lr_confusion_ms(folds_three),
        _ms(three_summary, "test_window_balacc"),
        str(n_params),
        f"x{n_params / s0_n_params:.2f}" if s0_n_params else "-",
        f"{d_three:+.4f}" if d_three is not None else "-",
        conclusion,
    ]

    with open(registry_path, "a", encoding="utf-8") as f:
        f.write("| " + " | ".join(row) + " |\n")


# ================================================================
# 7. 主入口：训练后调用
# ================================================================

def update_registry(
    arm_name: str,
    task_summary: dict | None,
    three_summary: dict | None,
    n_params: int,
    out_root_tag_dir: Path,
    registry_path: Path,
    is_s0: bool = False,
) -> None:
    """训练结束后调用：计算 delta、判定结论、写入登记表。

    对于 S0 自身，写入基线行（delta=0，结论=基线(S0)）。
    对于冒烟运行（max_folds!=5），标记但不判定成功/失败。
    """
    # 判断是否冒烟
    is_smoke = False
    if task_summary:
        mf = task_summary.get("max_folds", 5)
        if mf not in (0, 5):
            is_smoke = True

    if is_s0:
        append_registry_row(
            registry_path,
            arm_name,
            task_summary,
            three_summary,
            n_params,
            n_params,
            {
                "d_three_acc_paper": 0.0,
                "d_three_f1_macro": 0.0,
                "d_three_balacc_maj": 0.0,
                "d_task_acc_paper": 0.0,
                "d_task_balacc_maj": 0.0,
                "d_lr_confusion": 0.0,
            },
            "基线(S0)" if not is_smoke else "冒烟(不判定)",
        )
        print(f"[registry] S0 行已写入 -> {registry_path}")
        return

    # 加载 S0 基线
    s0_meta = find_s0_meta(out_root_tag_dir)
    s0_task = s0_meta.get("task") if s0_meta else None
    s0_three = s0_meta.get("three") if s0_meta else None
    s0_n_params = s0_meta.get("n_params") if s0_meta else None

    if s0_task is None and s0_three is None:
        print("[registry] 警告：未找到 S0 正式五折基线，delta 和结论标记为无S0基线")
        deltas = {
            "d_three_acc_paper": None,
            "d_three_f1_macro": None,
            "d_three_balacc_maj": None,
            "d_task_acc_paper": None,
            "d_task_balacc_maj": None,
            "d_lr_confusion": None,
        }
        conclusion = "无S0基线"
    else:
        deltas = compute_deltas(task_summary, three_summary, s0_task, s0_three)
        conclusion = determine_conclusion(
            deltas, three_summary, s0_three, is_smoke=is_smoke
        )

    append_registry_row(
        registry_path,
        arm_name,
        task_summary,
        three_summary,
        n_params,
        s0_n_params,
        deltas,
        conclusion,
    )

    # 控制台打印关键决策信息
    d_three = deltas.get("d_three_acc_paper")
    d_task = deltas.get("d_task_acc_paper")
    d_f1 = deltas.get("d_three_f1_macro")
    d_lr = deltas.get("d_lr_confusion")
    print(f"[registry] {arm_name} 已写入登记表")
    print(f"  dThree_AccPaper={d_three:+.4f}" if d_three is not None else "  dThree_AccPaper=N/A")
    print(f"  dThree_F1macro={d_f1:+.4f}" if d_f1 is not None else "  dThree_F1macro=N/A")
    print(f"  dTask_AccPaper={d_task:+.4f}" if d_task is not None else "  dTask_AccPaper=N/A")
    print(f"  dLR_confusion={d_lr:+.4f}" if d_lr is not None else "  dLR_confusion=N/A")
    print(f"  结论={conclusion}")
