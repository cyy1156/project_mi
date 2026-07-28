"""
一次跑完五个内置基线：各自按文档流程搜最佳参数。

实验流程（每个模型独立一遍，互不套用超参）::

    Task:  4 点小网格 → Val 选优 → 规则微调 1 轮 → 再按 Val 选最终 Task 超参
    Three: 从零初始化（默认不迁权重）→ 基线 → 规则微调 → Val 选最终 Three 超参
    报数:  Test 只写入报告，不参与选型
    标注:  weight_transfer=false, classifier=native

五个基线::

    eegnet, shallow, deep, eegtcnet, conformer

PyCharm 运行配置::

    Script path:       .../code/train_lab/src/step/run_five_baselines.py
    Working directory: .../code/train_lab/src/step
    Interpreter:       D:/cyy/MI/.venv/Scripts/python.exe
    Parameters:        （可空；或 --data merged_2s）

命令行::

    cd D:\\cyy\\MI\\code\\train_lab\\src\\step
    D:\\cyy\\MI\\.venv\\Scripts\\python.exe run_five_baselines.py
    D:\\cyy\\MI\\.venv\\Scripts\\python.exe run_five_baselines.py --data merged_2s
    D:\\cyy\\MI\\.venv\\Scripts\\python.exe run_five_baselines.py --models eegnet,shallow

结果::

    code/train_lab/out/matrix_<stamp>/overview.md
    code/train_lab/out/matrix_<stamp>/<model>_final_meta.json
    各模型权重: code/train_lab/out/baseline/<model>/<data>/overnight_*/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parent
TRAIN_LAB = STEP_DIR.parents[1]

sys.path.insert(0, str(STEP_DIR))
from models import BUILTIN_NAMES, get_spec, list_models  # noqa: E402

PY = Path(sys.executable)
DEFAULT_MODELS = ",".join(BUILTIN_NAMES)  # 五个基线


def _fmt_hp(hp: dict) -> str:
    keys = ("lr", "weight_decay", "drop_prob", "patience", "batch_train", "max_epochs")
    parts = [f"{k}={hp[k]}" for k in keys if k in hp]
    return ", ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(
        description="五个基线按文档流程各自搜最佳参数（Task/Three 独立、不迁权重）"
    )
    p.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=f"逗号分隔，默认五基线: {DEFAULT_MODELS}",
    )
    p.add_argument("--data", default="merged_2s", help="merged_2s | bci2a_2s | stieger_2s")
    p.add_argument(
        "--init-from-task",
        action="store_true",
        help="仅历史对照：三分类迁二分类主干（默认关闭）",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某一模型失败时继续跑后面的模型",
    )
    args = p.parse_args()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    known = set(list_models())
    for m in models:
        if m not in known:
            raise SystemExit(f"未知模型 {m!r}；已注册: {sorted(known)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    matrix_dir = TRAIN_LAB / "out" / f"matrix_{stamp}"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    overview = matrix_dir / "overview.md"
    results_json = matrix_dir / "all_results.json"

    header = [
        f"# 五基线搜参矩阵 {stamp}",
        "",
        "## 协议",
        "",
        "- 每个模型：Task 网格+调参 → Three 独立训练网格+调参",
        f"- data: `{args.data}`",
        f"- weight_transfer: `{bool(args.init_from_task)}`（默认 false）",
        "- classifier: `native`",
        "- 选型只看 **Overall Val**；Test 仅终评",
        f"- models: `{', '.join(models)}`",
        "",
        "## 汇总表",
        "",
        "| model | family | task_best_hp | task_Val_F1 | task_Test_F1 | "
        "three_best_hp | three_Val_F1m | three_Test_F1m |",
        "|-------|--------|--------------|-------------|--------------|"
        "----------------|---------------|----------------|",
    ]
    overview.write_text("\n".join(header) + "\n", encoding="utf-8")

    print("=" * 60, flush=True)
    print(f"五基线搜参开始  stamp={stamp}", flush=True)
    print(f"models={models}", flush=True)
    print(f"data={args.data}  weight_transfer={bool(args.init_from_task)}", flush=True)
    print(f"结果目录: {matrix_dir}", flush=True)
    print("=" * 60, flush=True)

    all_rows: list[dict] = []
    failed: list[str] = []

    for i, model in enumerate(models, start=1):
        print(f"\n>>>>> [{i}/{len(models)}] {model} <<<<<\n", flush=True)
        cmd = [
            str(PY),
            str(STEP_DIR / "run_overnight_kfold.py"),
            "--model",
            model,
            "--data",
            args.data,
            "--no-writeback",
        ]
        if args.init_from_task:
            cmd.append("--init-from-task")

        proc = subprocess.run(cmd, cwd=str(STEP_DIR))
        if proc.returncode != 0:
            failed.append(model)
            with open(overview, "a", encoding="utf-8") as f:
                f.write(f"| {model} | - | FAIL | - | - | FAIL | - | - |\n")
            msg = f"{model} overnight 失败 exit={proc.returncode}"
            if args.continue_on_error:
                print(f"[WARN] {msg}，继续下一个", flush=True)
                continue
            raise SystemExit(msg)

        base = TRAIN_LAB / "out" / "baseline" / model / args.data
        metas = sorted(
            base.glob("overnight_*/final_meta.json"),
            key=lambda x: x.stat().st_mtime,
        )
        if not metas:
            raise SystemExit(f"未找到 {base}/overnight_*/final_meta.json")

        meta = json.loads(metas[-1].read_text(encoding="utf-8"))
        fam = get_spec(model).family
        bt = meta["best_task"]
        b3 = meta["best_three"]
        task_hp = _fmt_hp(bt.get("hparams") or {})
        three_hp = _fmt_hp(b3.get("hparams") or {})

        row_md = (
            f"| {model} | {fam} | `{task_hp}` | "
            f"{bt['val_f1_mean']:.4f} | {bt['test_f1_mean']:.4f} | "
            f"`{three_hp}` | "
            f"{b3['val_f1_macro_mean']:.4f} | {b3['test_f1_macro_mean']:.4f} |"
        )
        with open(overview, "a", encoding="utf-8") as f:
            f.write(row_md + "\n")

        row = {
            "model": model,
            "family": fam,
            "weight_transfer": bool(args.init_from_task),
            "classifier": "native",
            "task": {
                "val_f1_mean": bt["val_f1_mean"],
                "test_f1_mean": bt["test_f1_mean"],
                "test_acc_mean": bt.get("test_acc_mean"),
                "hparams": bt.get("hparams"),
                "out_dir": bt.get("out_dir"),
            },
            "three": {
                "val_f1_macro_mean": b3["val_f1_macro_mean"],
                "test_f1_macro_mean": b3["test_f1_macro_mean"],
                "test_acc_mean": b3.get("test_acc_mean"),
                "hparams": b3.get("hparams"),
                "out_dir": b3.get("out_dir"),
            },
            "overnight_meta": str(metas[-1]),
        }
        all_rows.append(row)
        (matrix_dir / f"{model}_final_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[OK] {model}: Task Val F1={bt['val_f1_mean']:.4f} | "
            f"Three Val F1m={b3['val_f1_macro_mean']:.4f}",
            flush=True,
        )

    footer = [
        "",
        "## 路径说明",
        "",
        f"- 本表: `{overview}`",
        f"- JSON: `{results_json}`",
        "- 各模型过夜目录: `code/train_lab/out/baseline/<model>/"
        + args.data
        + "/overnight_*/`",
        "",
    ]
    if failed:
        footer += ["## 失败模型", "", ", ".join(failed), ""]

    with open(overview, "a", encoding="utf-8") as f:
        f.write("\n".join(footer))

    payload = {
        "stamp": stamp,
        "data": args.data,
        "weight_transfer": bool(args.init_from_task),
        "classifier": "native",
        "models": models,
        "failed": failed,
        "results": all_rows,
    }
    results_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (matrix_dir / "config_snapshot.json").write_text(
        json.dumps(
            {
                "stamp": stamp,
                "data": args.data,
                "weight_transfer": bool(args.init_from_task),
                "models": models,
                "builtin_names": list(BUILTIN_NAMES),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60, flush=True)
    print(f"全部结束。汇总: {overview}", flush=True)
    if failed:
        print(f"失败: {failed}", flush=True)
        raise SystemExit(1)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
