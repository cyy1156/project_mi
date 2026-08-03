"""五折 MD 明细（Task BalAcc 早停）。"""

from __future__ import annotations


def _f(x, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _task_fold_lines(r: dict) -> list[str]:
    m = r.get("test_metrics") or {}
    fold = r.get("fold", "?")
    lines = [
        f"#### Fold {fold}",
        "",
        f"- 早停/结束轮次（stopped_epoch）：`{r.get('stopped_epoch')}`",
        f"- 验证最优轮次（best_epoch）：`{r.get('best_epoch')}`",
        f"- Val 选模分数（Balanced Acc）：`{_f(r.get('best_val_balanced_accuracy'))}`",
        f"- Val F1（最优 checkpoint 时，附报）：`{_f(r.get('best_val_f1'))}`",
        f"- Val loss（最优时）：`{_f(r.get('best_val_loss'))}`",
        "",
        "**Test（overall）**",
        f"- Accuracy：`{_f(m.get('accuracy'))}`",
        f"- Recall：`{_f(m.get('recall'))}`",
        f"- Specificity：`{_f(m.get('specificity'))}`",
        f"- Precision：`{_f(m.get('precision'))}`",
        f"- F1：`{_f(m.get('f1'))}`",
        f"- Balanced Acc：`{_f(m.get('balanced_accuracy'))}`",
        f"- 混淆矩阵：TP=`{m.get('tp')}` TN=`{m.get('tn')}` FP=`{m.get('fp')}` FN=`{m.get('fn')}`",
        "",
    ]
    return lines


def task_fold_md_lines(folds: list[dict]) -> list[str]:
    lines = [
        "### Task 各折明细",
        "",
        "说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。",
        "",
    ]
    for r in folds:
        lines.extend(_task_fold_lines(r))
    return lines


def three_fold_md_lines(folds: list[dict]) -> list[str]:
    """兼容桩：供 importlib 加载 baselines_single / Self_development 旧脚本时导入。

    baselines_1s 本身只跑 Task，不会调用此函数写 MD。
    """
    _ = folds
    return []
