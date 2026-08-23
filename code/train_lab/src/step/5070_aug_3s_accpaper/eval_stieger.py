"""G1 三端评测之二：Stieger eval_half 零样本（09-A0 口径）。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

import _bootstrap  # noqa: F401

from s25_config import (  # noqa: E402
    ANCHOR_09_A0_EVAL_HALF_THREE,
    ANCHOR_S07_ZEROSHOT_THREE,
    N_FOLDS,
    PROTOCOL,
    RESULTS_ROOT,
)
from data import iter_subject_streams  # noqa: E402
from stieger_eval import eval_subject_fold  # noqa: E402
from util_metrics import jsonable, mean_std  # noqa: E402
from s25_weights import resolve_weight_run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="方案25 Stieger eval_half 零样本")
    p.add_argument("--arm", default="G1", choices=("A0", "G1", "G2", "G3"))
    p.add_argument("--run-stamp", default="")
    p.add_argument("--train-device", default="5070", choices=("5070", "5090"))
    p.add_argument("--subjects", default="")
    p.add_argument("--tasks", default="task,three")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    folds = (0,) if args.smoke else tuple(range(N_FOLDS))
    device = torch.device(args.device)
    run_dir = resolve_weight_run(
        args.arm,
        run_stamp=args.run_stamp or None,
        train_device=args.train_device,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = RESULTS_ROOT / f"S25-{args.arm}_zeroshot" / f"{stamp}_eval_half"
    out_run.mkdir(parents=True, exist_ok=True)

    results = {
        "arm": f"S25-{args.arm}",
        "protocol": PROTOCOL,
        "eval_protocol": "eval_half_noz_unified",
        "weight_run": str(run_dir),
        "anchors": {
            "S07-01_full_trial_three": ANCHOR_S07_ZEROSHOT_THREE,
            "S09-A0_eval_half_three": ANCHOR_09_A0_EVAL_HALF_THREE,
        },
        "subjects": {},
    }
    md = [
        f"# S25-{args.arm} Stieger eval_half 零样本",
        "",
        f"- 时间：`{stamp}`",
        f"- 权重：`{run_dir}`",
        f"- 口径：noz_unified · 后半 trial · 无 EA/AdaBN",
        "",
    ]

    macro_three: list[float] = []

    for stream in iter_subject_streams(subjects=subjects):
        print(f"\n=== {stream.subject_id} ===", flush=True)
        if stream.X_noz is None:
            raise RuntimeError(f"{stream.subject_id}: 缺少 stieger_X_noz.npy")
        sub: dict = {"metrics": {}}
        md += [f"## {stream.subject_id}", ""]
        for head in tasks:
            rows = []
            for fold in folds:
                row = eval_subject_fold(
                    stream,
                    init_run=run_dir,
                    head=head,
                    fold=fold,
                    device=device,
                    batch_size=int(args.batch_size),
                )
                rows.append(row)
                print(
                    f"  {head} fold{fold}: Acc_paper={row['acc_paper']:.4f}",
                    flush=True,
                )
            mean, std = mean_std([r["acc_paper"] for r in rows])
            sub["metrics"][head] = {
                "acc_paper_mean": mean,
                "acc_paper_std": std,
                "folds": rows,
            }
            md += [f"### {head}", "", f"- Acc_paper：`{mean:.4f}±{std:.4f}`", ""]
            if head == "three":
                macro_three.append(mean)
        results["subjects"][stream.subject_id] = sub
        (out_run / f"{stream.subject_id}_summary.json").write_text(
            json.dumps(jsonable(sub), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if macro_three:
        m, s = mean_std(macro_three)
        results["macro_three_acc_paper"] = {
            "mean": m,
            "std": s,
            "delta_vs_09_A0": m - ANCHOR_09_A0_EVAL_HALF_THREE,
        }
        md += [
            "## Macro",
            "",
            f"- Three Acc_paper：`{m:.4f}±{s:.4f}`",
            f"- Δ vs 09-A0 ({ANCHOR_09_A0_EVAL_HALF_THREE:.4f})：`{(m - ANCHOR_09_A0_EVAL_HALF_THREE)*100:+.2f} pp`",
            "",
        ]

    (out_run / "summary.json").write_text(
        json.dumps(jsonable(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_run / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[done] {out_run}", flush=True)


if __name__ == "__main__":
    main()
