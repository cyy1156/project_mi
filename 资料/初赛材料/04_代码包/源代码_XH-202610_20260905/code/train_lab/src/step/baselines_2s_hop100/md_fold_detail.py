"""五折 MD 明细（Task / Three；Val BalAcc 早停 + batch balance）。"""

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


def _three_fold_lines(r: dict) -> list[str]:
    m = r.get("test_metrics") or {}
    fold = r.get("fold", "?")
    cm = m.get("cm")
    lines = [
        f"#### Fold {fold}",
        "",
        f"- 早停/结束轮次（stopped_epoch）：`{r.get('stopped_epoch')}`",
        f"- 验证最优轮次（best_epoch）：`{r.get('best_epoch')}`",
        f"- Val 选模分数（Balanced Acc = Recall-macro）：`{_f(r.get('best_val_balanced_accuracy'))}`",
        f"- Val F1-macro（最优 checkpoint 时，附报）：`{_f(r.get('best_val_f1_macro'))}`",
        f"- Val loss（最优时）：`{_f(r.get('best_val_loss'))}`",
        "",
        "**Test（overall）**",
        f"- Accuracy：`{_f(m.get('accuracy'))}`",
        f"- Balanced Acc：`{_f(m.get('balanced_accuracy', m.get('recall_macro')))}`",
        f"- F1-macro：`{_f(m.get('f1_macro'))}`",
        f"- Precision-macro：`{_f(m.get('precision_macro'))}`",
        f"- Recall-macro：`{_f(m.get('recall_macro'))}`",
        f"- Recall idle/left/right：`{_f(m.get('recall_idle'))}` / "
        f"`{_f(m.get('recall_left'))}` / `{_f(m.get('recall_right'))}`",
        f"- Precision idle/left/right：`{_f(m.get('precision_idle'))}` / "
        f"`{_f(m.get('precision_left'))}` / `{_f(m.get('precision_right'))}`",
        f"- F1 idle/left/right：`{_f(m.get('f1_idle'))}` / "
        f"`{_f(m.get('f1_left'))}` / `{_f(m.get('f1_right'))}`",
    ]
    if cm is not None and len(cm) >= 3:
        lines.extend(
            [
                "- 混淆矩阵（行=真实, 列=预测）：",
                "```",
                "         pred0  pred1  pred2",
                f"  true0  {int(cm[0][0]):5d}  {int(cm[0][1]):5d}  {int(cm[0][2]):5d}",
                f"  true1  {int(cm[1][0]):5d}  {int(cm[1][1]):5d}  {int(cm[1][2]):5d}",
                f"  true2  {int(cm[2][0]):5d}  {int(cm[2][1]):5d}  {int(cm[2][2]):5d}",
                "```",
            ]
        )
    lines.append("")
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
    lines = [
        "### Three 各折明细",
        "",
        "说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；"
        "训练 **batch balance**（三类 inverse-freq）。",
        "",
    ]
    for r in folds:
        lines.extend(_three_fold_lines(r))
    return lines
