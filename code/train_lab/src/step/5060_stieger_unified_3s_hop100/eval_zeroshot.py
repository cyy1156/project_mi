"""S10-01 / S10-01b：零样本全量评测（--tw 3s|2s）。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import torch
from braindecode.models import ShallowFBCSPNet

import _bootstrap  # noqa: F401

from config import (
    N_FOLDS,
    N_TIMES,
    active_profile,
    apply_tw,
    protocol_repro,
    results_zeroshot_dir,
)
from data import iter_subject_streams
from infer import load_fold_model, predict_windows
from util_metrics import aggregate_windows_to_segments, jsonable, mean_std
from weights import resolve_openbmi_s3_run


def build_shallow(n_chans, n_times, n_outputs, drop_prob):
    return ShallowFBCSPNet(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, drop_prob=drop_prob
    )


def main() -> None:
    p = argparse.ArgumentParser(description="S10 零样本 Stieger")
    p.add_argument("--tw", default="3s", choices=("2s", "3s"))
    p.add_argument("--subjects", default="")
    p.add_argument("--tasks", default="task,three")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--run-stamp", default="")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()

    prof = apply_tw(args.tw)
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    folds = (0,) if args.smoke else tuple(range(N_FOLDS))
    device = torch.device(args.device)
    run_dir = resolve_openbmi_s3_run("shallow", run_stamp=args.run_stamp or None)

    results_root = results_zeroshot_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = results_root / f"{stamp}_shallow_zeroshot"
    out_run.mkdir(parents=True, exist_ok=True)

    results = {
        "arm": prof.arm_zeroshot,
        "tw": prof.tw,
        "protocol": protocol_repro(),
        "weight_run": str(run_dir),
        "n_times": N_TIMES,
        "data_tag": prof.data_tag,
        "subjects": {},
    }
    md = [
        f"# {prof.arm_zeroshot} Stieger 零样本 · Tw={prof.tw}",
        "",
        f"- 时间：`{stamp}`",
        f"- 权重：`{run_dir}`",
        f"- 数据：`{prof.data_dir}`",
        "",
    ]

    for stream in iter_subject_streams(subjects=subjects):
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
                rows.append(
                    {
                        "fold": fold,
                        "acc_paper": float(m["acc_paper"]),
                    }
                )
                print(
                    f"  {head} f{fold}: acc={m['acc_paper']:.4f}",
                    flush=True,
                )
            mean, std = mean_std([r["acc_paper"] for r in rows])
            sub["metrics"][head] = {
                "acc_paper_mean": mean,
                "acc_paper_std": std,
                "folds": rows,
            }
            md += [f"### {head}", f"- Acc_paper: **{mean:.4f}±{std:.4f}**", ""]
        results["subjects"][stream.subject_id] = sub
        (out_run / f"{stream.subject_id}_summary.json").write_text(
            json.dumps(jsonable(sub), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    macro = {}
    for head in tasks:
        vals = [
            float(s["metrics"][head]["acc_paper_mean"])
            for s in results["subjects"].values()
            if head in s.get("metrics", {})
        ]
        m, s = mean_std(vals)
        macro[head] = {"mean": m, "std": s, "per_subject": vals}
    results["macro"] = macro
    (out_run / "summary.json").write_text(
        json.dumps(jsonable(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = results_root / f"{stamp}_{prof.arm_zeroshot}.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_run / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[done] {report}", flush=True)


if __name__ == "__main__":
    main()
