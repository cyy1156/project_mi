"""五折 MD 明细（Acc_paper 早停 · Three-only）。"""

from __future__ import annotations


def _f(x, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def three_fold_md_lines(folds: list[dict]) -> list[str]:
    lines = [
        "### Three 分折明细",
        "",
        "说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。",
        "",
    ]
    for r in folds:
        m = r.get("test_trial_metrics") or {}
        mw = r.get("test_window_metrics") or {}
        cm = m.get("cm")
        lines.extend(
            [
                f"#### Fold {r.get('fold', '?')}",
                "",
                f"- stopped_epoch：`{r.get('stopped_epoch')}` | best_epoch：`{r.get('best_epoch')}`",
                f"- Val Acc_paper（早停）：`{_f(r.get('best_val_acc_paper'))}`",
                f"- Val BalAcc_maj（附报）：`{_f(r.get('best_val_balacc_maj'))}`",
                f"- Val loss（最优时）：`{_f(r.get('best_val_loss'))}`",
                "",
                "**Test 试次级**",
                f"- Acc_paper：`{_f(m.get('acc_paper'))}`",
                f"- BalAcc_maj：`{_f(m.get('balanced_accuracy'))}`",
                f"- Acc_majority：`{_f(m.get('acc_majority'))}`",
                f"- F1-macro（众数）：`{_f(m.get('f1_macro'))}`",
                f"- Recall-macro：`{_f(m.get('recall_macro'))}`",
                f"- Recall idle/left/right：`{_f(m.get('recall_idle'))}` / "
                f"`{_f(m.get('recall_left'))}` / `{_f(m.get('recall_right'))}`",
                f"- n_trials：`{m.get('n_trials')}`",
                "",
                "**Test 窗级（附报）**",
                f"- BalAcc：`{_f(mw.get('balanced_accuracy'))}` | "
                f"F1m：`{_f(mw.get('f1_macro'))}` | Acc：`{_f(mw.get('accuracy'))}`",
            ]
        )
        if cm is not None and len(cm) >= 3:
            lines.extend(
                [
                    "- 混淆矩阵 试次级众数（行=真实, 列=预测）：",
                    "```",
                    "         pred0  pred1  pred2",
                    f"  true0  {int(cm[0][0]):5d}  {int(cm[0][1]):5d}  {int(cm[0][2]):5d}",
                    f"  true1  {int(cm[1][0]):5d}  {int(cm[1][1]):5d}  {int(cm[1][2]):5d}",
                    f"  true2  {int(cm[2][0]):5d}  {int(cm[2][1]):5d}  {int(cm[2][2]):5d}",
                    "```",
                ]
            )
        wcm = mw.get("cm")
        if wcm is not None and len(wcm) >= 3:
            lines.extend(
                [
                    "- 混淆矩阵 窗级（行=真实, 列=预测）：",
                    "```",
                    "         pred0  pred1  pred2",
                    f"  true0  {int(wcm[0][0]):5d}  {int(wcm[0][1]):5d}  {int(wcm[0][2]):5d}",
                    f"  true1  {int(wcm[1][0]):5d}  {int(wcm[1][1]):5d}  {int(wcm[1][2]):5d}",
                    f"  true2  {int(wcm[2][0]):5d}  {int(wcm[2][1]):5d}  {int(wcm[2][2]):5d}",
                    "```",
                ]
            )
        lines.append("")
    return lines
