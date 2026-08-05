"""五折 MD 明细（Acc_paper 早停）。"""

from __future__ import annotations


def _f(x, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def task_fold_md_lines(folds: list[dict]) -> list[str]:
    lines = ["### Task 分折明细", ""]
    for r in folds:
        m = r.get("test_trial_metrics") or {}
        mw = r.get("test_window_metrics") or {}
        lines.extend(
            [
                f"#### Fold {r.get('fold', '?')}",
                "",
                f"- stopped_epoch：`{r.get('stopped_epoch')}` | best_epoch：`{r.get('best_epoch')}`",
                f"- Val Acc_paper（早停）：`{_f(r.get('best_val_acc_paper'))}`",
                f"- Val BalAcc_maj（附报）：`{_f(r.get('best_val_balacc_maj'))}`",
                "",
                "**Test 试次级**",
                f"- Acc_paper：`{_f(m.get('acc_paper'))}`",
                f"- BalAcc_maj：`{_f(m.get('balanced_accuracy'))}`",
                f"- Acc_majority：`{_f(m.get('acc_majority'))}`",
                f"- n_trials：`{m.get('n_trials')}`",
                "",
                "**Test 窗级（附报）**",
                f"- BalAcc：`{_f(mw.get('balanced_accuracy'))}` | F1：`{_f(mw.get('f1'))}` | Acc：`{_f(mw.get('accuracy'))}`",
                "",
            ]
        )
    return lines


def three_fold_md_lines(folds: list[dict]) -> list[str]:
    lines = ["### Three 分折明细", ""]
    for r in folds:
        m = r.get("test_trial_metrics") or {}
        mw = r.get("test_window_metrics") or {}
        lines.extend(
            [
                f"#### Fold {r.get('fold', '?')}",
                "",
                f"- stopped_epoch：`{r.get('stopped_epoch')}` | best_epoch：`{r.get('best_epoch')}`",
                f"- Val Acc_paper（早停）：`{_f(r.get('best_val_acc_paper'))}`",
                f"- Val BalAcc_maj（附报）：`{_f(r.get('best_val_balacc_maj'))}`",
                "",
                "**Test 试次级**",
                f"- Acc_paper：`{_f(m.get('acc_paper'))}`",
                f"- BalAcc_maj：`{_f(m.get('balanced_accuracy'))}`",
                f"- F1-macro（众数）：`{_f(m.get('f1_macro'))}`",
                f"- Rec idle/left/right：`{_f(m.get('recall_idle'))}` / "
                f"`{_f(m.get('recall_left'))}` / `{_f(m.get('recall_right'))}`",
                f"- n_trials：`{m.get('n_trials')}`",
                "",
                "**Test 窗级（附报）**",
                f"- BalAcc：`{_f(mw.get('balanced_accuracy'))}` | F1m：`{_f(mw.get('f1_macro'))}`",
                "",
            ]
        )
    return lines
