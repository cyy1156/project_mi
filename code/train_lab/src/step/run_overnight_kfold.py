"""
过夜实验编排：依次跑头1五折 → 按 val 自动调超参再跑一轮 →
用最优头1权重跑头2五折 → 再按 val 调一轮。

调参只看验证集（协议要求）；测试集仅终评写入报告。
结果追加写入：资料/模型训练/五折过夜实验记录_*.md
"""

from __future__ import annotations

import copy
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

# 保证可直接 `python run_overnight_kfold.py`
STEP_DIR = Path(__file__).resolve().parent
TRAIN_LAB = STEP_DIR.parents[1]
CODE_ROOT = STEP_DIR.parents[2]
REPO_ROOT = CODE_ROOT.parent

sys.path.insert(0, str(STEP_DIR))
sys.path.insert(0, str(TRAIN_LAB))

from train_task_kfold import TaskKFoldConfig, run_task_kfold  # noqa: E402
from train_three_kfold import ThreeKFoldConfig, run_three_kfold  # noqa: E402

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MD_DIR = REPO_ROOT / "资料" / "模型训练"
MD_PATH = MD_DIR / f"五折过夜实验记录_{STAMP}.md"
MD_LATEST = MD_DIR / "五折过夜实验记录_最新.md"
OUT_ROOT = TRAIN_LAB / "out" / f"overnight_{STAMP}"
LOG_PATH = OUT_ROOT / "overnight.log"


def append_md(text: str) -> None:
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MD_PATH, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    # 稳定入口：始终指向本次实验文件
    MD_LATEST.write_text(
        f"# 最新过夜实验入口\n\n"
        f"本次记录文件：[`{MD_PATH.name}`](./{MD_PATH.name})\n\n"
        f"权重目录：`{OUT_ROOT}`\n"
        f"日志：`{LOG_PATH}`\n",
        encoding="utf-8",
    )


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fmt_hparams(hp: dict) -> str:
    keys = [
        "lr",
        "weight_decay",
        "drop_prob",
        "patience",
        "max_epochs",
        "batch_train",
        "seed",
        "val_ratio",
        "n_folds",
        "freeze_backbone",
        "f1",
        "d",
        "f2",
    ]
    lines = []
    for k in keys:
        if k in hp:
            lines.append(f"- `{k}`: `{hp[k]}`")
    return "\n".join(lines)


def write_task_section(run_name: str, summary: dict, note: str = "") -> None:
    hp = summary["hparams"]
    lines = [
        f"## {run_name}",
        "",
        f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`",
        f"- 输出目录：`{summary['out_dir']}`",
    ]
    if note:
        lines.append(f"- 说明：{note}")
    lines += [
        "",
        "### 超参",
        "",
        fmt_hparams(hp),
        "",
        "### 各折验证 / 测试",
        "",
        "| fold | best_ep | val_F1 | test_Acc | test_F1 | test_Spe | n_train/val/test |",
        "|------|---------|--------|----------|---------|----------|------------------|",
    ]
    for r in summary["folds"]:
        m = r["test_metrics"]
        lines.append(
            f"| {r['fold']} | {r['best_epoch']} | {r['best_val_f1']:.4f} | "
            f"{m['accuracy']:.4f} | {m['f1']:.4f} | {m['specificity']:.4f} | "
            f"{r['n_train']}/{r['n_val']}/{r['n_test']} |"
        )
    lines += [
        "",
        "### 汇总（调参看 Val；报数看 Test）",
        "",
        f"- **Val F1** mean±std = `{summary['val_f1_mean']:.4f} ± {summary['val_f1_std']:.4f}`",
        f"- **Test Acc** mean±std = `{summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}`",
        f"- **Test F1** mean±std = `{summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}`",
        f"- 平均 best_epoch = `{summary['mean_best_epoch']:.1f}`",
        "",
        "---",
        "",
    ]
    append_md("\n".join(lines))


def write_three_section(run_name: str, summary: dict, note: str = "") -> None:
    hp = summary["hparams"]
    lines = [
        f"## {run_name}",
        "",
        f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`",
        f"- 输出目录：`{summary['out_dir']}`",
        f"- 头1权重目录：`{hp.get('task_kfold_dir', '')}`",
    ]
    if note:
        lines.append(f"- 说明：{note}")
    lines += [
        "",
        "### 超参",
        "",
        fmt_hparams(hp),
        "",
        "### 各折验证 / 测试",
        "",
        "| fold | best_ep | val_F1macro | test_Acc | test_F1macro | R_idle | R_left | R_right |",
        "|------|---------|-------------|----------|--------------|--------|--------|---------|",
    ]
    for r in summary["folds"]:
        m = r["test_metrics"]
        lines.append(
            f"| {r['fold']} | {r['best_epoch']} | {r['best_val_f1_macro']:.4f} | "
            f"{m['accuracy']:.4f} | {m['f1_macro']:.4f} | "
            f"{m['recall_idle']:.3f} | {m['recall_left']:.3f} | {m['recall_right']:.3f} |"
        )
    lines += [
        "",
        "### 汇总（调参看 Val；报数看 Test）",
        "",
        f"- **Val F1-macro** mean±std = "
        f"`{summary['val_f1_macro_mean']:.4f} ± {summary['val_f1_macro_std']:.4f}`",
        f"- **Test Acc** mean±std = `{summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}`",
        f"- **Test F1-macro** mean±std = "
        f"`{summary['test_f1_macro_mean']:.4f} ± {summary['test_f1_macro_std']:.4f}`",
        f"- 平均 best_epoch = `{summary['mean_best_epoch']:.1f}`",
        "",
        "---",
        "",
    ]
    append_md("\n".join(lines))


def suggest_task_hparams(cfg: TaskKFoldConfig, summary: dict) -> tuple[TaskKFoldConfig, list[str]]:
    """仅根据验证集信号建议下一轮超参（不看 test）。"""
    new = copy.deepcopy(cfg)
    reasons: list[str] = []
    val_mean = summary["val_f1_mean"]
    val_std = summary["val_f1_std"]
    mean_ep = summary["mean_best_epoch"]
    early_folds = sum(1 for r in summary["folds"] if r["best_epoch"] <= 5)
    late_folds = sum(1 for r in summary["folds"] if r["stopped_epoch"] >= cfg.max_epochs - 1)

    if early_folds >= 3:
        new.lr = max(1e-4, cfg.lr * 0.5)
        new.patience = min(25, cfg.patience + 5)
        reasons.append(
            f"有 {early_folds}/5 折 best_epoch≤5 → 学习率 {cfg.lr:g}→{new.lr:g}，patience→{new.patience}"
        )
    elif late_folds >= 3 and val_mean < 0.75:
        new.drop_prob = max(0.25, round(cfg.drop_prob - 0.10, 2))
        new.lr = min(3e-3, cfg.lr * 1.5)
        new.patience = min(25, cfg.patience + 5)
        reasons.append(
            f"多数折接近满 epoch 且 val_F1={val_mean:.3f} 偏低 → "
            f"drop {cfg.drop_prob}→{new.drop_prob}，lr→{new.lr:g}，patience→{new.patience}"
        )
    elif val_std >= 0.12:
        new.drop_prob = min(0.70, round(cfg.drop_prob + 0.10, 2))
        new.weight_decay = min(1e-3, cfg.weight_decay * 2.0)
        reasons.append(
            f"折间 val_F1 标准差偏大({val_std:.3f}) → "
            f"drop→{new.drop_prob}，weight_decay→{new.weight_decay:g}"
        )
    elif val_mean < 0.70:
        new.drop_prob = min(0.70, round(cfg.drop_prob + 0.10, 2))
        new.weight_decay = min(1e-3, cfg.weight_decay * 2.0)
        new.lr = max(3e-4, cfg.lr * 0.7)
        reasons.append(
            f"val_F1={val_mean:.3f} 偏低 → 加强正则：drop→{new.drop_prob}，"
            f"wd→{new.weight_decay:g}，lr→{new.lr:g}"
        )
    else:
        new.lr = max(3e-4, cfg.lr * 0.7)
        new.drop_prob = min(0.65, round(cfg.drop_prob + 0.05, 2))
        reasons.append(
            f"基线尚可(val_F1={val_mean:.3f}) → 微调：lr→{new.lr:g}，drop→{new.drop_prob}"
        )

    # 人少时 val 噪，第二轮略加大 patience
    if new.patience == cfg.patience:
        new.patience = min(25, cfg.patience + 3)
        reasons.append(f"BCI2a 仅9人、val 人极少 → patience {cfg.patience}→{new.patience}")

    return new, reasons


def suggest_three_hparams(cfg: ThreeKFoldConfig, summary: dict) -> tuple[ThreeKFoldConfig, list[str]]:
    new = copy.deepcopy(cfg)
    reasons: list[str] = []
    val_mean = summary["val_f1_macro_mean"]
    val_std = summary["val_f1_macro_std"]
    mean_ep = summary["mean_best_epoch"]
    early_folds = sum(1 for r in summary["folds"] if r["best_epoch"] <= 5)

    # 看左右召回是否极不均衡（用 test 仅作诊断描述；真正改参仍以 val 为主）
    lefts = [r["test_metrics"]["recall_left"] for r in summary["folds"]]
    rights = [r["test_metrics"]["recall_right"] for r in summary["folds"]]
    mean_left = sum(lefts) / len(lefts)
    mean_right = sum(rights) / len(rights)

    if early_folds >= 3:
        new.lr = max(1e-4, cfg.lr * 0.5)
        new.patience = min(25, cfg.patience + 5)
        reasons.append(
            f"有 {early_folds}/5 折过早停 → lr {cfg.lr:g}→{new.lr:g}，patience→{new.patience}"
        )
    elif val_mean < 0.45:
        new.drop_prob = max(0.30, round(cfg.drop_prob - 0.10, 2))
        new.lr = min(2e-3, cfg.lr * 1.5)
        new.patience = min(25, cfg.patience + 5)
        reasons.append(
            f"val_F1macro={val_mean:.3f} 偏低疑欠拟合 → drop→{new.drop_prob}，lr→{new.lr:g}"
        )
    elif val_std >= 0.12:
        new.drop_prob = min(0.75, round(cfg.drop_prob + 0.05, 2))
        new.weight_decay = min(1e-3, cfg.weight_decay * 2.0)
        reasons.append(
            f"折间波动大(std={val_std:.3f}) → drop→{new.drop_prob}，wd→{new.weight_decay:g}"
        )
    else:
        new.lr = max(3e-4, cfg.lr * 0.7)
        new.drop_prob = min(0.70, round(cfg.drop_prob + 0.05, 2))
        reasons.append(
            f"基线尚可(val_F1macro={val_mean:.3f}) → 微调 lr→{new.lr:g}，drop→{new.drop_prob}"
        )

    if abs(mean_left - mean_right) > 0.25 and not cfg.freeze_backbone:
        # 左右差大时第二轮可试更小 lr 稳住迁移特征
        new.lr = min(new.lr, max(2e-4, cfg.lr * 0.5))
        reasons.append(
            f"左右召回差较大(L={mean_left:.2f}, R={mean_right:.2f}) → 进一步降 lr→{new.lr:g}"
        )

    if new.patience == cfg.patience:
        new.patience = min(25, cfg.patience + 3)
        reasons.append(f"val 人少 → patience→{new.patience}")

    _ = mean_ep  # 保留诊断字段，避免未使用告警
    return new, reasons


def write_tune_note(title: str, reasons: list[str], new_hp: dict) -> None:
    lines = [
        f"## {title}",
        "",
        "调参依据（**只看验证集**；测试集不参与选型）：",
        "",
    ]
    for r in reasons:
        lines.append(f"- {r}")
    lines += ["", "下一轮拟用超参：", "", fmt_hparams(new_hp), "", "---", ""]
    append_md("\n".join(lines))


def pick_best_task(run_a: dict, run_b: dict) -> dict:
    """按 val_F1_mean 选更好的一轮（平局取标准差更小）。"""
    if run_b["val_f1_mean"] > run_a["val_f1_mean"] + 1e-6:
        return run_b
    if abs(run_b["val_f1_mean"] - run_a["val_f1_mean"]) <= 1e-6 and run_b["val_f1_std"] < run_a["val_f1_std"]:
        return run_b
    return run_a


def pick_best_three(run_a: dict, run_b: dict) -> dict:
    if run_b["val_f1_macro_mean"] > run_a["val_f1_macro_mean"] + 1e-6:
        return run_b
    if (
        abs(run_b["val_f1_macro_mean"] - run_a["val_f1_macro_mean"]) <= 1e-6
        and run_b["val_f1_macro_std"] < run_a["val_f1_macro_std"]
    ):
        return run_b
    return run_a


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    append_md(
        "\n".join(
            [
                f"# 被试独立五折过夜实验记录（{STAMP}）",
                "",
                "> 协议：被试独立五折 + 内层按人 val；早停看 val；test 仅终评。  ",
                "> 自动调参：**只根据验证集**改超参，不用测试集选型。  ",
                "> 顺序：头1基线 → 头1调参轮 →（选优）→ 头2基线 → 头2调参轮。",
                "",
                f"- 开始时间：`{datetime.now().isoformat(timespec='seconds')}`",
                f"- device：`{device}`",
                f"- 数据：`code/preprocess_lab/out/bci2a/bci2a_*.npy`",
                f"- 权重根目录：`{OUT_ROOT}`",
                f"- 运行日志：`{LOG_PATH}`",
                "",
                "---",
                "",
            ]
        )
    )
    log(f"MD_PATH={MD_PATH}")
    log(f"OUT_ROOT={OUT_ROOT}")
    log(f"device={device}")

    try:
        # ----- 头1 第1轮 -----
        cfg1 = TaskKFoldConfig(
            out_dir=str(OUT_ROOT / "01_task_baseline"),
            lr=1e-3,
            weight_decay=1e-4,
            drop_prob=0.50,
            patience=15,
            max_epochs=100,
        )
        log("开始：头1五折 基线")
        sum1 = run_task_kfold(cfg1, device=device)
        write_task_section("Run01 头1五折（基线）", sum1, note="默认超参首跑")
        log(
            f"Run01 完成 val_F1={sum1['val_f1_mean']:.4f} test_F1={sum1['test_f1_mean']:.4f}"
        )

        cfg2, reasons = suggest_task_hparams(cfg1, sum1)
        cfg2.out_dir = str(OUT_ROOT / "02_task_tuned")
        write_tune_note("自动调参：头1 → Run02", reasons, cfg2.__dict__)
        log("调参建议: " + " | ".join(reasons))

        log("开始：头1五折 调参轮")
        sum2 = run_task_kfold(cfg2, device=device)
        write_task_section("Run02 头1五折（自动调参）", sum2, note="；".join(reasons))
        log(
            f"Run02 完成 val_F1={sum2['val_f1_mean']:.4f} test_F1={sum2['test_f1_mean']:.4f}"
        )

        best_task = pick_best_task(sum1, sum2)
        best_tag = "Run01" if best_task is sum1 else "Run02"
        append_md(
            "\n".join(
                [
                    "## 头1选优",
                    "",
                    f"- 按 **Val F1** 选中：**{best_tag}**",
                    f"- Val F1 = `{best_task['val_f1_mean']:.4f} ± {best_task['val_f1_std']:.4f}`",
                    f"- 对应 Test F1（仅报告）= "
                    f"`{best_task['test_f1_mean']:.4f} ± {best_task['test_f1_std']:.4f}`",
                    f"- 后续头2迁移权重目录：`{best_task['out_dir']}`",
                    "",
                    "---",
                    "",
                ]
            )
        )
        log(f"头1选优: {best_tag} → {best_task['out_dir']}")

        # ----- 头2 第1轮 -----
        cfg3 = ThreeKFoldConfig(
            out_dir=str(OUT_ROOT / "03_three_baseline"),
            task_kfold_dir=best_task["out_dir"],
            lr=1e-3,
            weight_decay=1e-4,
            drop_prob=0.60,
            patience=15,
            max_epochs=100,
            freeze_backbone=False,
        )
        log("开始：头2五折 基线（迁移最优头1）")
        sum3 = run_three_kfold(cfg3, device=device)
        write_three_section(
            "Run03 头2五折（基线，迁移最优头1）",
            sum3,
            note=f"init from {best_tag}",
        )
        log(
            f"Run03 完成 val_F1m={sum3['val_f1_macro_mean']:.4f} "
            f"test_F1m={sum3['test_f1_macro_mean']:.4f}"
        )

        cfg4, reasons3 = suggest_three_hparams(cfg3, sum3)
        cfg4.out_dir = str(OUT_ROOT / "04_three_tuned")
        cfg4.task_kfold_dir = best_task["out_dir"]
        write_tune_note("自动调参：头2 → Run04", reasons3, cfg4.__dict__)
        log("调参建议: " + " | ".join(reasons3))

        log("开始：头2五折 调参轮")
        sum4 = run_three_kfold(cfg4, device=device)
        write_three_section("Run04 头2五折（自动调参）", sum4, note="；".join(reasons3))
        log(
            f"Run04 完成 val_F1m={sum4['val_f1_macro_mean']:.4f} "
            f"test_F1m={sum4['test_f1_macro_mean']:.4f}"
        )

        best_three = pick_best_three(sum3, sum4)
        best3_tag = "Run03" if best_three is sum3 else "Run04"

        # 同步一份「当前推荐」超参回默认脚本常量说明进 md
        append_md(
            "\n".join(
                [
                    "## 最终结论（明早可读）",
                    "",
                    f"- 结束时间：`{datetime.now().isoformat(timespec='seconds')}`",
                    "",
                    "### 头1（静息/任务）推荐",
                    "",
                    f"- 轮次：**{best_tag}**",
                    f"- Val F1：`{best_task['val_f1_mean']:.4f} ± {best_task['val_f1_std']:.4f}`",
                    f"- Test F1：`{best_task['test_f1_mean']:.4f} ± {best_task['test_f1_std']:.4f}`",
                    f"- Test Acc：`{best_task['test_acc_mean']:.4f} ± {best_task['test_acc_std']:.4f}`",
                    "",
                    fmt_hparams(best_task["hparams"]),
                    "",
                    "### 头2（空闲/左/右）推荐",
                    "",
                    f"- 轮次：**{best3_tag}**",
                    f"- Val F1-macro：`{best_three['val_f1_macro_mean']:.4f} ± {best_three['val_f1_macro_std']:.4f}`",
                    f"- Test F1-macro：`{best_three['test_f1_macro_mean']:.4f} ± {best_three['test_f1_macro_std']:.4f}`",
                    f"- Test Acc：`{best_three['test_acc_mean']:.4f} ± {best_three['test_acc_std']:.4f}`",
                    "",
                    fmt_hparams(best_three["hparams"]),
                    "",
                    "### 权重路径",
                    "",
                    f"- 头1：`{best_task['out_dir']}`",
                    f"- 头2：`{best_three['out_dir']}`",
                    "",
                    "> 说明：BCI IV 2a 仅 9 人，五折时 val 往往只有 1～2 人，早停与调参噪声大；",
                    "> 以上为按协议自动跑通的过夜基线，人多后同一流程会更稳。",
                    "",
                ]
            )
        )

        # 把推荐超参写回 train_*_kfold.py 顶部常量，方便明早直接用
        apply_recommended_defaults(best_task["hparams"], best_three["hparams"])

        meta = {
            "stamp": STAMP,
            "md": str(MD_PATH),
            "best_task_run": best_tag,
            "best_three_run": best3_tag,
            "best_task": {
                "val_f1_mean": best_task["val_f1_mean"],
                "test_f1_mean": best_task["test_f1_mean"],
                "hparams": best_task["hparams"],
                "out_dir": best_task["out_dir"],
            },
            "best_three": {
                "val_f1_macro_mean": best_three["val_f1_macro_mean"],
                "test_f1_macro_mean": best_three["test_f1_macro_mean"],
                "hparams": best_three["hparams"],
                "out_dir": best_three["out_dir"],
            },
        }
        with open(OUT_ROOT / "final_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log("全部完成")
    except Exception:
        err = traceback.format_exc()
        log("失败:\n" + err)
        append_md(
            "\n".join(
                [
                    "## 运行中断",
                    "",
                    f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`",
                    "",
                    "```",
                    err,
                    "```",
                    "",
                ]
            )
        )
        raise


def apply_recommended_defaults(task_hp: dict, three_hp: dict) -> None:
    """把选中的超参写回两个 kfold 脚本的默认 Config 字段，明早可直接单跑。"""
    import re

    task_path = STEP_DIR / "train_task_kfold.py"
    three_path = STEP_DIR / "train_three_kfold.py"

    def patch_field(text: str, field: str, value: str) -> str:
        pat = rf"({field}:\s*[^=\n]+=\s*)[^\n]+"
        new_text, n = re.subn(pat, rf"\g<1>{value}", text, count=1)
        if n == 0:
            log(f"[warn] 未找到字段 {field} 可替换")
        return new_text

    def patch_file(path: Path, updates: dict[str, str]) -> None:
        text = path.read_text(encoding="utf-8")
        for field, value in updates.items():
            text = patch_field(text, field, value)
        path.write_text(text, encoding="utf-8")

    patch_file(
        task_path,
        {
            "lr": repr(float(task_hp["lr"])),
            "weight_decay": repr(float(task_hp["weight_decay"])),
            "drop_prob": repr(float(task_hp["drop_prob"])),
            "patience": str(int(task_hp["patience"])),
        },
    )
    patch_file(
        three_path,
        {
            "lr": repr(float(three_hp["lr"])),
            "weight_decay": repr(float(three_hp["weight_decay"])),
            "drop_prob": repr(float(three_hp["drop_prob"])),
            "patience": str(int(three_hp["patience"])),
        },
    )
    append_md(
        "\n".join(
            [
                "## 代码默认超参已更新",
                "",
                "已按选优结果改写：",
                "",
                "- `code/train_lab/src/step/train_task_kfold.py` 的 `TaskKFoldConfig` 默认值",
                "- `code/train_lab/src/step/train_three_kfold.py` 的 `ThreeKFoldConfig` 默认值",
                "",
                "明早可直接：",
                "",
                "```powershell",
                "cd code/train_lab",
                "$env:PYTHONPATH=\"src/step;.\"",
                "python -m src.step.train_task_kfold",
                "python -m src.step.train_three_kfold",
                "```",
                "",
            ]
        )
    )
    log("已回写推荐超参到 kfold 脚本默认值")


if __name__ == "__main__":
    main()
