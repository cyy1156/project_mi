"""S07-01：S3 shallow × Stieger 全量零样本（无门控 ≡ H0 / Q0）。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from braindecode.models import ShallowFBCSPNet

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import DOCS_07, N_FOLDS, N_TIMES, PROTOCOL, RESULTS_ROOT
from data import iter_subject_streams
from infer import load_fold_model, predict_windows
from util_metrics import aggregate_windows_to_segments, jsonable, mean_std
from weights import resolve_openbmi_s3_run

RESULTS = RESULTS_ROOT / "S07-01_zeroshot"


def build_shallow(n_chans, n_times, n_outputs, drop_prob):
    return ShallowFBCSPNet(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, drop_prob=drop_prob
    )


def main() -> None:
    p = argparse.ArgumentParser(description="S07-01 Stieger 3s 零样本")
    p.add_argument("--subjects", default="", help="逗号分隔；空=全部")
    p.add_argument("--tasks", default="task,three")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--run-stamp", default="")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    folds = (0,) if args.smoke else tuple(range(N_FOLDS))
    device = torch.device(args.device)
    run_dir = resolve_openbmi_s3_run("shallow", run_stamp=args.run_stamp or None)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = RESULTS / f"{stamp}_shallow_zeroshot"
    out_run.mkdir(parents=True, exist_ok=True)

    streams = iter_subject_streams(subjects=subjects)
    results = {
        "arm": "S07-01",
        "protocol": PROTOCOL,
        "weight_run": str(run_dir),
        "n_times": N_TIMES,
        "subjects": {},
    }
    md = [
        f"# S07-01 Stieger 零样本 · shallow 3s",
        "",
        f"- 时间：`{stamp}`",
        f"- 权重：`{run_dir}`",
        f"- 门控：无（≡ H0 / Q0）",
        "",
    ]

    for stream in streams:
        print(f"\n=== {stream.subject_id} n={len(stream.X)} ===", flush=True)
        sub: dict = {"n_windows": int(len(stream.X)), "metrics": {}}
        md += [f"## {stream.subject_id}", ""]
        for head in tasks:
            n_classes = 2 if head == "task" else 3
            y = stream.y_task if head == "task" else stream.y_three
            rows = []
            for fold in folds:
                net = load_fold_model(
                    build_shallow, run_dir, head=head, fold=fold, device=device
                )
                pred = predict_windows(
                    net, stream.X, device, batch_size=int(args.batch_size)
                )
                del net
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                agg = aggregate_windows_to_segments(
                    y, pred, stream.seg_keys, n_classes=n_classes
                )
                m = agg["segment_metrics"]
                row = {
                    "fold": fold,
                    "acc_paper": float(m["acc_paper"]),
                    "balanced_accuracy": float(m.get("balanced_accuracy") or float("nan")),
                    "n_segments": int(m.get("n_segments") or 0),
                }
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
            md += [
                f"### {head}",
                "",
                f"- Acc_paper：`{mean:.4f}±{std:.4f}`",
                "",
            ]
        results["subjects"][stream.subject_id] = sub
        (out_run / f"{stream.subject_id}_summary.json").write_text(
            json.dumps(jsonable(sub), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (out_run / "summary.json").write_text(
        json.dumps(jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path = RESULTS / f"{stamp}_shallow_zeroshot.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_run / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[done] {md_path}", flush=True)
    print(f"[done] docs={DOCS_07}", flush=True)


if __name__ == "__main__":
    main()
