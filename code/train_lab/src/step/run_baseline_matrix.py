"""
多模型基线矩阵：对每个模型各跑一遍过夜（Task/Three 独立搜参）。

示例::

    cd code/train_lab/src/step
    D:\\cyy\\MI\\.venv\\Scripts\\python.exe run_baseline_matrix.py --models eegnet,shallow,deep
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
REPO_ROOT = STEP_DIR.parents[3]

sys.path.insert(0, str(STEP_DIR))
from models import BUILTIN_NAMES, list_models  # noqa: E402

PY = Path(sys.executable)


def main() -> None:
    p = argparse.ArgumentParser(description="多模型基线矩阵过夜")
    p.add_argument(
        "--models",
        default=",".join(BUILTIN_NAMES),
        help="逗号分隔；默认五基线。也可只用 CNN: eegnet,shallow,deep",
    )
    p.add_argument("--data", default="merged_2s")
    p.add_argument("--init-from-task", action="store_true")
    p.add_argument("--no-writeback", action="store_true", default=True)
    args = p.parse_args()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    known = set(list_models())
    for m in models:
        if m not in known:
            raise SystemExit(f"未知模型 {m!r}；可选: {sorted(known)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    matrix_dir = TRAIN_LAB / "out" / f"matrix_{stamp}"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    overview = matrix_dir / "overview.md"
    rows = []

    lines = [
        f"# 基线矩阵 {stamp}",
        "",
        f"- data: `{args.data}`",
        f"- weight_transfer: `{bool(args.init_from_task)}`",
        f"- classifier: `native`",
        f"- models: `{', '.join(models)}`",
        "",
        "| model | family | task_Val_F1 | task_Test_F1 | three_Val_F1m | three_Test_F1m | task_out | three_out |",
        "|-------|--------|-------------|--------------|---------------|----------------|----------|-----------|",
    ]
    overview.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for model in models:
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
        print(f"\n===== MATRIX: {model} =====\n", flush=True)
        proc = subprocess.run(cmd, cwd=str(STEP_DIR))
        if proc.returncode != 0:
            with open(overview, "a", encoding="utf-8") as f:
                f.write(f"| {model} | - | FAIL | FAIL | FAIL | FAIL | - | - |\n")
            raise SystemExit(f"{model} overnight failed with code {proc.returncode}")

        # 找该模型最新 overnight final_meta
        base = TRAIN_LAB / "out" / "baseline" / model / args.data
        metas = sorted(base.glob("overnight_*/final_meta.json"), key=lambda x: x.stat().st_mtime)
        if not metas:
            raise SystemExit(f"未找到 {base}/overnight_*/final_meta.json")
        meta = json.loads(metas[-1].read_text(encoding="utf-8"))
        from models import get_spec

        fam = get_spec(model).family
        bt = meta["best_task"]
        b3 = meta["best_three"]
        row = (
            f"| {model} | {fam} | "
            f"{bt['val_f1_mean']:.4f} | {bt['test_f1_mean']:.4f} | "
            f"{b3['val_f1_macro_mean']:.4f} | {b3['test_f1_macro_mean']:.4f} | "
            f"`{bt['out_dir']}` | `{b3['out_dir']}` |"
        )
        with open(overview, "a", encoding="utf-8") as f:
            f.write(row + "\n")
        rows.append(meta)
        (matrix_dir / f"{model}_final_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    snap = {
        "stamp": stamp,
        "data": args.data,
        "weight_transfer": bool(args.init_from_task),
        "models": models,
        "builtin_reference": list(BUILTIN_NAMES),
    }
    (matrix_dir / "config_snapshot.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n矩阵完成 → {overview}", flush=True)


if __name__ == "__main__":
    main()
