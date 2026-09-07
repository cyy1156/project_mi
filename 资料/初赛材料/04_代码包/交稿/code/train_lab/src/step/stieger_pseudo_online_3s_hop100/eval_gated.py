"""S07-03 / S07-06：在线门控 H0–H3。

--mode zeroshot → S07-03（S3 权重 · 全量）
--mode ft       → S07-06（只读半程 FT ckpt · 后半）
"""

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

from config import FT_WEIGHT_ROOT, N_FOLDS, N_TIMES, OPENBMI_CHANS, RESULTS_ROOT
from data import iter_subject_streams
from data_split import assert_no_leakage, build_cue_split, window_mask_for_cues
from infer import load_fold_model, load_ft_fold, predict_windows
from util_metrics import (
    aggregate_windows_to_segments_gated,
    build_gate_keeps,
    gate_stats,
    jsonable,
    mean_std,
)
from weights import resolve_openbmi_s3_run


def build_shallow(n_chans, n_times, n_outputs, drop_prob):
    return ShallowFBCSPNet(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, drop_prob=drop_prob
    )


def main() -> None:
    p = argparse.ArgumentParser(description="S07 在线门控 H0–H3")
    p.add_argument("--mode", choices=("zeroshot", "ft"), default="zeroshot")
    p.add_argument("--subjects", default="")
    p.add_argument("--gates", default="H0,H1,H2,H3")
    p.add_argument("--tasks", default="task,three")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--run-stamp", default="", help="S3 零样本 run；ft 模式忽略")
    p.add_argument(
        "--ft-stamp",
        default="",
        help="S07-02 FT 权重 stamp（目录名）；默认取最新",
    )
    p.add_argument("--g3-top-p", type=float, default=0.5)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    gates = tuple(g.strip() for g in args.gates.split(",") if g.strip())
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    folds = (0,) if args.smoke else tuple(range(N_FOLDS))
    device = torch.device(args.device)

    if args.mode == "zeroshot":
        arm = "S07-03"
        results_dir = RESULTS_ROOT / "S07-03_gate"
        run_dir = resolve_openbmi_s3_run("shallow", run_stamp=args.run_stamp or None)
        ft_root = None
    else:
        arm = "S07-06"
        results_dir = RESULTS_ROOT / "S07-06_ft_gated"
        run_dir = None
        base = FT_WEIGHT_ROOT / "shallow_stieger_ft_half_balbatch_accpaper"
        if args.ft_stamp:
            ft_root = base / args.ft_stamp
        else:
            cands = sorted(
                [p for p in base.iterdir() if p.is_dir()],
                key=lambda p: p.name,
                reverse=True,
            ) if base.is_dir() else []
            if not cands:
                raise FileNotFoundError(
                    f"未找到 FT 权重，请先跑 ft_half.py → {base}"
                )
            ft_root = cands[0]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = results_dir / f"{stamp}_shallow_{args.mode}_gate"
    out_run.mkdir(parents=True, exist_ok=True)

    results = {
        "arm": arm,
        "mode": args.mode,
        "weight_run": str(run_dir) if run_dir else None,
        "ft_root": str(ft_root) if ft_root else None,
        "gates": list(gates),
        "n_times": N_TIMES,
        "channels": list(OPENBMI_CHANS),
        "subjects": {},
    }
    md = [
        f"# {arm} Stieger 在线门控 · {args.mode}",
        "",
        f"- 时间：`{stamp}`",
        f"- 权重：`{run_dir or ft_root}`",
        f"- 门控：`{', '.join(gates)}`（H1=ERD≤−15 & lat≥8；质量窗无 z-score）",
        f"- cue 配对：Task tid / Rest=tid+1 → cue_id 共用",
        "",
    ]

    for stream in iter_subject_streams(subjects=subjects):
        print(f"\n=== {stream.subject_id} ===", flush=True)
        if stream.X_noz is None or not stream.meta.get("has_x_noz"):
            raise RuntimeError(
                f"{stream.subject_id}: 缺少 stieger_X_noz.npy，请重跑 "
                "batch_3s_hop100（含 X_noz）"
            )

        eval_mask = np.ones(len(stream.X), dtype=bool)
        if args.mode == "ft":
            split = build_cue_split(stream)
            assert_no_leakage(split)
            eval_mask = window_mask_for_cues(stream.cue_ids, split.eval_cues)
            if int(eval_mask.sum()) == 0:
                raise RuntimeError(f"{stream.subject_id}: 后半无窗")

        idx = np.flatnonzero(eval_mask)
        X = stream.X[idx]
        X_noz = stream.X_noz[idx]
        y_task = stream.y_task[idx]
        y_three = stream.y_three[idx]
        segs = stream.segs[idx]
        seg_keys = stream.seg_keys[idx]
        cue_ids = stream.cue_ids[idx]

        keeps = build_gate_keeps(
            X_noz,
            y_three,
            cue_ids,  # 同 cue 的 REST 作 MI 基线
            segs,
            ch_names=list(OPENBMI_CHANS),
            top_p=float(args.g3_top_p),
            modes=gates,
        )
        sub_pack: dict = {
            "n_windows_eval": int(len(X)),
            "gates": {},
            "metrics": {},
        }
        for g, k in keeps.items():
            st = gate_stats(k, segs, seg_keys)
            sub_pack["gates"][g] = st
            print(
                f"  {g}: kept={st['n_kept']}/{st['n_windows']} "
                f"mi_abstain={st['n_mi_segments_abstain']}/{st['n_mi_segments']}",
                flush=True,
            )

        md += [f"## {stream.subject_id}", ""]
        for head in tasks:
            n_classes = 2 if head == "task" else 3
            y = y_task if head == "task" else y_three
            per_gate: dict[str, list[dict]] = {g: [] for g in gates}
            for fold in folds:
                if args.mode == "zeroshot":
                    net = load_fold_model(
                        build_shallow, run_dir, head=head, fold=fold, device=device
                    )
                else:
                    net = load_ft_fold(
                        build_shallow,
                        ft_subject_root=ft_root / stream.subject_id,
                        head=head,
                        fold=fold,
                        device=device,
                    )
                pred = predict_windows(
                    net, X, device, batch_size=int(args.batch_size)
                )
                del net
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                for g in gates:
                    agg = aggregate_windows_to_segments_gated(
                        y, pred, seg_keys, keeps[g], n_classes=n_classes
                    )
                    m = agg["segment_metrics"]
                    row = {
                        "fold": fold,
                        "acc_paper": float(m["acc_paper"])
                        if m.get("acc_paper") is not None
                        else float("nan"),
                        "abstain_rate": float(m["abstain_rate"]),
                        "abstain_as_wrong_acc": float(m["abstain_as_wrong_acc"])
                        if m.get("abstain_as_wrong_acc") is not None
                        else float("nan"),
                        "n_segments_scored": int(m["n_segments_scored"]),
                        "n_segments_all": int(m["n_segments_all"]),
                    }
                    per_gate[g].append(row)
                    print(
                        f"  {head} fold{fold} {g}: "
                        f"acc={row['acc_paper']:.4f} abstain={row['abstain_rate']:.3f}",
                        flush=True,
                    )

            h0_mean = mean_std([r["acc_paper"] for r in per_gate.get("H0", [])])[0]
            summ_g = {}
            for g, rows in per_gate.items():
                mean, std = mean_std([r["acc_paper"] for r in rows])
                ab_m, ab_s = mean_std([r["abstain_rate"] for r in rows])
                summ_g[g] = {
                    "acc_paper_mean": mean,
                    "acc_paper_std": std,
                    "delta_vs_H0": float(mean - h0_mean)
                    if np.isfinite(mean) and np.isfinite(h0_mean)
                    else float("nan"),
                    "abstain_rate_mean": ab_m,
                    "abstain_rate_std": ab_s,
                    "folds": rows,
                }
            sub_pack["metrics"][head] = summ_g
            md += [
                f"### {head}",
                "",
                "| 代号 | Acc_paper | vs H0 Δ | abstain |",
                "|------|-----------|---------|---------|",
            ]
            for g in gates:
                s = summ_g[g]
                md.append(
                    f"| {g} | {s['acc_paper_mean']:.4f}±{s['acc_paper_std']:.4f} | "
                    f"{s['delta_vs_H0']:+.4f} | "
                    f"{s['abstain_rate_mean']:.3f}±{s['abstain_rate_std']:.3f} |"
                )
            md.append("")
            if head == "three" and "H1" in summ_g:
                d = summ_g["H1"]["delta_vs_H0"]
                tip = (
                    f"H1 ΔThree={d:+.4f} → 达 +0.02"
                    if np.isfinite(d) and d >= 0.02
                    else f"H1 ΔThree={d:+.4f} → 未达主成功线"
                )
                md += [f"- **决策**：{tip}", ""]
                sub_pack["metrics"]["three_decision"] = tip

        results["subjects"][stream.subject_id] = sub_pack
        (out_run / f"{stream.subject_id}_summary.json").write_text(
            json.dumps(jsonable(sub_pack), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (out_run / "summary.json").write_text(
        json.dumps(jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path = results_dir / f"{stamp}_shallow_{args.mode}_gate.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_run / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[done] {md_path}", flush=True)


if __name__ == "__main__":
    main()
