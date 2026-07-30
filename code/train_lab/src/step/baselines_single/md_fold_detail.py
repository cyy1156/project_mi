"""五折实验记录 MD：各折明细（测试指标 + 早停/最优轮次）。

仅供 baselines_single 各 baseline_*.py 在写 MD 时调用；不改训练逻辑。
"""

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
        f"- Val F1（最优）：`{_f(r.get('best_val_f1'))}`",
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
    by_ds = r.get("test_metrics_by_dataset") or {}
    extras = [(k, v) for k, v in by_ds.items() if k != "overall" and isinstance(v, dict)]
    if extras:
        lines.append("**Test（按数据集前缀）**")
        for name, dm in extras:
            lines.append(
                f"- `{name}`：Acc=`{_f(dm.get('accuracy'))}` F1=`{_f(dm.get('f1'))}` "
                f"BalAcc=`{_f(dm.get('balanced_accuracy'))}`"
            )
        lines.append("")
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
        f"- Val F1-macro（最优）：`{_f(r.get('best_val_f1_macro'))}`",
        f"- Val loss（最优时）：`{_f(r.get('best_val_loss'))}`",
        "",
        "**Test（overall）**",
        f"- Accuracy：`{_f(m.get('accuracy'))}`",
        f"- F1-macro：`{_f(m.get('f1_macro'))}`",
        f"- Recall-macro：`{_f(m.get('recall_macro'))}`",
        f"- Recall idle/left/right：`{_f(m.get('recall_idle'))}` / "
        f"`{_f(m.get('recall_left'))}` / `{_f(m.get('recall_right'))}`",
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
    by_ds = r.get("test_metrics_by_dataset") or {}
    extras = [(k, v) for k, v in by_ds.items() if k != "overall" and isinstance(v, dict)]
    if extras:
        lines.append("**Test（按数据集前缀）**")
        for name, dm in extras:
            lines.append(
                f"- `{name}`：Acc=`{_f(dm.get('accuracy'))}` "
                f"F1m=`{_f(dm.get('f1_macro'))}`"
            )
        lines.append("")
    return lines


def task_fold_md_lines(folds: list[dict]) -> list[str]:
    """Task 五折明细 MD 行（含标题）。"""
    lines = [
        "### Task 各折明细",
        "",
        "说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；"
        "`best_epoch` 为验证集最优并保存权重的轮次。",
        "",
    ]
    for r in folds:
        lines.extend(_task_fold_lines(r))
    return lines


def three_fold_md_lines(folds: list[dict]) -> list[str]:
    """Three 五折明细 MD 行（含标题）。"""
    lines = [
        "### Three 各折明细",
        "",
        "说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；"
        "`best_epoch` 为验证集最优并保存权重的轮次。",
        "",
    ]
    for r in folds:
        lines.extend(_three_fold_lines(r))
    return lines
