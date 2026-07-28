"""
补跑三分类 4 点网格 + 规则微调，覆盖写入指定 matrix 目录。

保留原 matrix 中的 Task 最优；Three 用新网格结果覆盖。

默认目标: code/train_lab/out/matrix_20260727_203340

示例::

    cd D:\\cyy\\MI\\code\\train_lab\\src\\step
    D:\\cyy\\MI\\.venv\\Scripts\\python.exe run_three_grid_backfill.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

STEP_DIR = Path(__file__).resolve().parent
TRAIN_LAB = STEP_DIR.parents[1]

sys.path.insert(0, str(STEP_DIR))

from models import BUILTIN_NAMES, get_spec  # noqa: E402
from run_overnight_kfold import (  # noqa: E402
    pick_best_three,
    suggest_three_hparams,
)
from train_three_kfold import ThreeKFoldConfig, run_three_kfold  # noqa: E402

DEFAULT_MATRIX = TRAIN_LAB / "out" / "matrix_20260727_203340"


def _fmt_hp(hp: dict) -> str:
    keys = ("lr", "weight_decay", "drop_prob", "patience", "batch_train", "max_epochs")
    return ", ".join(f"{k}={hp[k]}" for k in keys if k in hp)


def _three_grid_cfgs(model_name: str, data_tag: str) -> list[ThreeKFoldConfig]:
    common = dict(
        model_name=model_name,
        data_tag=data_tag,
        init_from_task=False,
        task_kfold_dir="",
        freeze_backbone=False,
        max_epochs=100,
    )
    return [
        ThreeKFoldConfig(**common, lr=1e-3, weight_decay=1e-4, drop_prob=0.50, patience=15),
        ThreeKFoldConfig(**common, lr=7e-4, weight_decay=1e-4, drop_prob=0.55, patience=18),
        ThreeKFoldConfig(**common, lr=5e-4, weight_decay=2e-4, drop_prob=0.60, patience=20),
        ThreeKFoldConfig(**common, lr=1.5e-3, weight_decay=5e-5, drop_prob=0.40, patience=15),
    ]


def _cfg_from_hp(model_name: str, data_tag: str, hp: dict, out_dir: str) -> ThreeKFoldConfig:
    return ThreeKFoldConfig(
        model_name=model_name,
        data_tag=data_tag,
        init_from_task=False,
        task_kfold_dir="",
        n_folds=int(hp.get("n_folds", 5)),
        val_ratio=float(hp.get("val_ratio", 0.2)),
        seed=int(hp.get("seed", 42)),
        max_epochs=int(hp.get("max_epochs", 100)),
        patience=int(hp["patience"]),
        batch_train=int(hp.get("batch_train", 32)),
        batch_eval=int(hp.get("batch_eval", 64)),
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
        drop_prob=float(hp["drop_prob"]),
        f1=int(hp.get("f1", 8)),
        d=int(hp.get("d", 2)),
        f2=int(hp.get("f2", 16)),
        model_kwargs=hp.get("model_kwargs"),
        freeze_backbone=False,
        out_dir=out_dir,
    )


def run_one_model(
    model: str,
    data_tag: str,
    old_meta: dict,
    device: torch.device,
) -> dict:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = TRAIN_LAB / "out" / "baseline" / model / data_tag / f"three_grid_backfill_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "backfill.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"model={model} out_root={out_root}")
    grid_sums = []
    for gi, gcfg in enumerate(_three_grid_cfgs(model, data_tag), start=1):
        gcfg.out_dir = str(out_root / f"03_three_grid_{gi}")
        log(f"Three 网格 {gi}/4 lr={gcfg.lr} drop={gcfg.drop_prob} wd={gcfg.weight_decay}")
        gsum = run_three_kfold(gcfg, device=device)
        grid_sums.append(gsum)
        log(
            f"Grid{gi} done val_F1m={gsum['val_f1_macro_mean']:.4f} "
            f"test_F1m={gsum['test_f1_macro_mean']:.4f}"
        )

    best_grid = grid_sums[0]
    for g in grid_sums[1:]:
        best_grid = pick_best_three(best_grid, g)
    log(
        f"网格最优 Val F1m={best_grid['val_f1_macro_mean']:.4f} "
        f"dir={best_grid['out_dir']}"
    )

    cfg_base = _cfg_from_hp(
        model, data_tag, best_grid["hparams"], str(out_root / "03_three_baseline")
    )
    # 不重复训网格最优；直接当作 Run03
    sum3 = dict(best_grid)

    cfg4, reasons = suggest_three_hparams(cfg_base, sum3)
    cfg4.model_name = model
    cfg4.data_tag = data_tag
    cfg4.init_from_task = False
    cfg4.out_dir = str(out_root / "04_three_tuned")
    log("调参: " + " | ".join(reasons))
    sum4 = run_three_kfold(cfg4, device=device)
    log(
        f"Tune done val_F1m={sum4['val_f1_macro_mean']:.4f} "
        f"test_F1m={sum4['test_f1_macro_mean']:.4f}"
    )

    best_three = pick_best_three(sum3, sum4)
    best3_tag = "Run03_grid" if best_three is sum3 else "Run04_tuned"
    log(f"Three 选优: {best3_tag} Val={best_three['val_f1_macro_mean']:.4f}")

    # 保留原 Task；覆盖 Three
    new_meta = dict(old_meta)
    new_meta["three_grid_backfill"] = {
        "stamp": stamp,
        "out_root": str(out_root),
        "best_three_run": best3_tag,
        "grid_vals": [g["val_f1_macro_mean"] for g in grid_sums],
        "tune_reasons": reasons,
    }
    new_meta["best_three_run"] = best3_tag
    new_meta["best_three"] = {
        "val_f1_macro_mean": best_three["val_f1_macro_mean"],
        "test_f1_macro_mean": best_three["test_f1_macro_mean"],
        "test_acc_mean": best_three.get("test_acc_mean"),
        "hparams": best_three["hparams"],
        "out_dir": best_three["out_dir"],
    }
    new_meta["weight_transfer"] = False
    new_meta["classifier"] = "native"
    (out_root / "final_meta.json").write_text(
        json.dumps(new_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return new_meta


def write_matrix(matrix_dir: Path, metas: list[dict], data_tag: str) -> None:
    stamp_note = datetime.now().isoformat(timespec="seconds")
    lines = [
        f"# 五基线搜参矩阵 {matrix_dir.name}",
        "",
        "## 协议",
        "",
        "- 每个模型：Task 网格+调参 → **Three 网格+调参**（本文件 Three 已补跑网格）",
        f"- data: `{data_tag}`",
        "- weight_transfer: `False`",
        "- classifier: `native`",
        "- 选型只看 **Overall Val**；Test 仅终评",
        f"- models: `{', '.join(m['model_name'] for m in metas)}`",
        f"- Three 网格补跑更新时间: `{stamp_note}`",
        "",
        "## 汇总表",
        "",
        "| model | family | task_best_hp | task_Val_F1 | task_Test_F1 | "
        "three_best_hp | three_Val_F1m | three_Test_F1m |",
        "|-------|--------|--------------|-------------|--------------|"
        "----------------|---------------|----------------|",
    ]
    all_rows = []
    for meta in metas:
        model = meta["model_name"]
        fam = get_spec(model).family
        bt = meta["best_task"]
        b3 = meta["best_three"]
        lines.append(
            f"| {model} | {fam} | `{_fmt_hp(bt.get('hparams') or {})}` | "
            f"{bt['val_f1_mean']:.4f} | {bt['test_f1_mean']:.4f} | "
            f"`{_fmt_hp(b3.get('hparams') or {})}` | "
            f"{b3['val_f1_macro_mean']:.4f} | {b3['test_f1_macro_mean']:.4f} |"
        )
        (matrix_dir / f"{model}_final_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_rows.append(
            {
                "model": model,
                "family": fam,
                "weight_transfer": False,
                "classifier": "native",
                "task": {
                    "val_f1_mean": bt["val_f1_mean"],
                    "test_f1_mean": bt["test_f1_mean"],
                    "hparams": bt.get("hparams"),
                    "out_dir": bt.get("out_dir"),
                },
                "three": {
                    "val_f1_macro_mean": b3["val_f1_macro_mean"],
                    "test_f1_macro_mean": b3["test_f1_macro_mean"],
                    "hparams": b3.get("hparams"),
                    "out_dir": b3.get("out_dir"),
                },
                "three_grid_backfill": meta.get("three_grid_backfill"),
            }
        )

    lines += [
        "",
        "## 路径说明",
        "",
        f"- 本表: `{matrix_dir / 'overview.md'}`",
        f"- JSON: `{matrix_dir / 'all_results.json'}`",
        "- Three 补跑权重: `code/train_lab/out/baseline/<model>/"
        + data_tag
        + "/three_grid_backfill_*/`",
        "",
    ]
    (matrix_dir / "overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (matrix_dir / "all_results.json").write_text(
        json.dumps(
            {
                "stamp": matrix_dir.name,
                "data": data_tag,
                "weight_transfer": False,
                "classifier": "native",
                "three_grid_backfill": True,
                "updated_at": stamp_note,
                "results": all_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (matrix_dir / "config_snapshot.json").write_text(
        json.dumps(
            {
                "matrix": matrix_dir.name,
                "data": data_tag,
                "weight_transfer": False,
                "three_has_grid": True,
                "models": list(BUILTIN_NAMES),
                "updated_at": stamp_note,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="补跑 Three 网格并覆盖 matrix 结果")
    p.add_argument("--matrix-dir", default=str(DEFAULT_MATRIX))
    p.add_argument("--data", default="merged_2s")
    p.add_argument(
        "--models",
        default=",".join(BUILTIN_NAMES),
        help="默认五个基线",
    )
    args = p.parse_args()

    matrix_dir = Path(args.matrix_dir)
    if not matrix_dir.is_dir():
        raise SystemExit(f"matrix 目录不存在: {matrix_dir}")

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} matrix={matrix_dir}", flush=True)

    # 备份旧 overview
    overview = matrix_dir / "overview.md"
    if overview.exists():
        bak = matrix_dir / f"overview_before_three_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        bak.write_text(overview.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"已备份旧 overview → {bak.name}", flush=True)

    new_metas: list[dict] = []
    for i, model in enumerate(models, start=1):
        meta_path = matrix_dir / f"{model}_final_meta.json"
        if not meta_path.exists():
            raise SystemExit(f"缺少 {meta_path}（需要保留其中的 Task 结果）")
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"\n===== [{i}/{len(models)}] {model} Three 网格补跑 =====\n", flush=True)
        new_meta = run_one_model(model, args.data, old_meta, device)
        new_metas.append(new_meta)

        # 每完成一个模型就回写：已补跑的用新结果，未跑的暂保留旧 Three
        done = {m["model_name"]: m for m in new_metas}
        merged = []
        for m in models:
            if m in done:
                merged.append(done[m])
            else:
                merged.append(
                    json.loads((matrix_dir / f"{m}_final_meta.json").read_text(encoding="utf-8"))
                )
        write_matrix(matrix_dir, merged, args.data)

    print(f"\n全部完成，已覆盖写入 {matrix_dir}", flush=True)


if __name__ == "__main__":
    main()
