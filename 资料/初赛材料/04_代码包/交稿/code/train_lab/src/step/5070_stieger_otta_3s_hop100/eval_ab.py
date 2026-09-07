"""S09 A/B 臂评测：后半 trial · 因果 OTTA 回放。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from braindecode.models import ShallowFBCSPNet
from scipy.stats import wilcoxon

import _bootstrap  # noqa: F401

from arms import ArmSpec, get_arm, parse_arms
from config import (
    ADABN_PREDICT_FIRST,
    ADABN_VERSION,
    ANCHOR_A0,
    ANCHOR_B0,
    FT_STAMP_DEFAULT,
    FT_WEIGHT_ROOT,
    INPUT_PIPELINE,
    N_FOLDS,
    N_TIMES,
    OPENBMI_CHANS,
    PROTOCOL,
    PROTOCOL_VERSION,
    RESULTS_ROOT,
)
from data import iter_subject_streams
from data_split import assert_no_leakage, build_cue_split
from infer import load_fold_model, load_ft_fold
from otta_infer import build_eval_tensors, predict_eval_pack
from paired_results import load_paired_summary
from ref_cov import load_ref_cov_src
from util_metrics import (
    aggregate_windows_to_segments,
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


def _resolve_ft_root(ft_stamp: str) -> Path:
    base = FT_WEIGHT_ROOT / "shallow_stieger_ft_half_balbatch_accpaper"
    stamp = ft_stamp or FT_STAMP_DEFAULT
    root = base / stamp
    if not root.is_dir():
        raise FileNotFoundError(f"FT 权重不存在: {root}")
    return root


def _combine_keep(gate_keep: np.ndarray, conf_keep: np.ndarray) -> np.ndarray:
    return np.asarray(gate_keep & conf_keep, dtype=bool)


def _anchor_for(arm: ArmSpec) -> dict | None:
    if arm.readonly_anchor == "S07-01":
        return ANCHOR_A0
    if arm.readonly_anchor == "S07-05":
        return ANCHOR_B0
    return None


def _paired_delta(
    cur_per: list[float],
    anchor_arm: str,
    *,
    head: str = "three",
) -> dict | None:
    paired = load_paired_summary(anchor_arm, head=head)
    if not paired:
        return None
    base = paired.get("per_subject") or []
    if len(base) != len(cur_per):
        return None
    deltas = [c - b for c, b in zip(cur_per, base)]
    d_mean = float(np.mean(deltas))
    n_pass = sum(1 for d in deltas if d >= 0.02)
    try:
        p = float(wilcoxon(deltas).pvalue)
    except Exception:
        p = float("nan")
    return {
        f"{head}_vs_{anchor_arm}": d_mean,
        "paired_anchor": anchor_arm,
        "paired_baseline_run": paired.get("path"),
        "n_pass_ge_0.02": n_pass,
        "wilcoxon_p": p,
    }


def eval_one_arm(
    arm: ArmSpec,
    *,
    subjects: list[str] | None,
    tasks: list[str],
    folds: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    ft_stamp: str,
    run_stamp: str,
    conf_tau: float | None,
    g3_top_p: float,
    r_ref_src: np.ndarray | None,
    ref_meta: dict,
    adabn_version: str,
) -> Path:
    results_dir = RESULTS_ROOT / arm.results_subdir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = results_dir / f"{stamp}_shallow_{arm.arm_id}"
    out_run.mkdir(parents=True, exist_ok=True)

    run_dir = None
    ft_root = None
    if arm.weight == "zeroshot":
        run_dir = resolve_openbmi_s3_run("shallow", run_stamp=run_stamp or None)
    else:
        ft_root = _resolve_ft_root(ft_stamp)

    tau = conf_tau if conf_tau is not None else arm.conf_tau
    gates = (arm.gate,) if arm.gate != "H0" else ("H0",)

    results = {
        "arm": arm.arm_id,
        "protocol": PROTOCOL,
        "spec": {
            "protocol_version": PROTOCOL_VERSION,
            "input_pipeline": INPUT_PIPELINE,
            "weight": arm.weight,
            "ea_ref": arm.ea_ref,
            "adabn": arm.adabn,
            "adabn_version": adabn_version,
            "adabn_predict_first": ADABN_PREDICT_FIRST,
            "gate": arm.gate,
            "conf_tau": tau,
            "eval_protocol": "eval_half_causal",
            "paired_anchor": arm.anchor,
            "ablation_vs": list(arm.ablation_vs),
            "readonly_anchor": arm.readonly_anchor,
        },
        "weight_run": str(run_dir) if run_dir else None,
        "ft_root": str(ft_root) if ft_root else None,
        "ea_ref_meta": ref_meta,
        "subjects": {},
    }
    md = [
        f"# S09-{arm.arm_id} OTTA 回放",
        "",
        f"- 时间：`{stamp}`",
        f"- 权重：`{run_dir or ft_root}`",
        f"- 管线：{INPUT_PIPELINE} · EA={arm.ea_ref} AdaBN={arm.adabn} ({adabn_version}, predict_first={ADABN_PREDICT_FIRST})",
        "",
    ]

    per_sub_acc: dict[str, dict[str, list[float]]] = {}

    for stream in iter_subject_streams(subjects=subjects):
        print(f"\n=== {arm.arm_id} {stream.subject_id} ===", flush=True)
        if stream.X_noz is None or not stream.meta.get("has_x_noz"):
            raise RuntimeError(f"{stream.subject_id}: 缺少 stieger_X_noz.npy")

        split = build_cue_split(stream)
        assert_no_leakage(split)
        pack = build_eval_tensors(
            stream, split, ea_ref=arm.ea_ref, r_ref_src=r_ref_src
        )
        if "ea_pipeline_meta" not in results:
            results["ea_pipeline_meta"] = pack.get("ea_meta")

        keeps = {}
        if arm.gate != "H0":
            keeps = build_gate_keeps(
                pack["X_noz_eval"],
                pack["y_three"],
                pack["cue_ids"],
                pack["segs"],
                ch_names=list(OPENBMI_CHANS),
                top_p=float(g3_top_p),
                modes=gates,
            )

        sub_pack: dict = {
            "n_windows_eval": pack["n_eval"],
            "split": {
                "n_train_cues": split.n_train,
                "n_eval_cues": split.n_eval,
            },
            "metrics": {},
            "latency_ms_mean": [],
        }
        per_sub_acc[stream.subject_id] = {}

        md += [f"## {stream.subject_id}", ""]

        for head in tasks:
            n_classes = 2 if head == "task" else 3
            y = pack["y_task"] if head == "task" else pack["y_three"]
            fold_rows = []
            for fold in folds:
                if arm.weight == "zeroshot":
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
                pr = predict_eval_pack(
                    net,
                    pack,
                    device,
                    adabn=arm.adabn,
                    conf_tau=tau,
                    batch_size=batch_size,
                    adabn_version=adabn_version,
                )
                del net
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                if not np.isnan(pr["latency_mean_ms"]):
                    sub_pack["latency_ms_mean"].append(pr["latency_mean_ms"])

                if arm.gate != "H0":
                    gk = keeps[arm.gate]
                    keep = _combine_keep(gk, pr["conf_keep"])
                    agg = aggregate_windows_to_segments_gated(
                        y, pr["pred"], pack["seg_keys"], keep, n_classes=n_classes
                    )
                    gst = gate_stats(gk, pack["segs"], pack["seg_keys"])
                    sub_pack.setdefault("gate_stats", {})[arm.gate] = gst
                elif tau is not None:
                    agg = aggregate_windows_to_segments_gated(
                        y,
                        pr["pred"],
                        pack["seg_keys"],
                        pr["conf_keep"],
                        n_classes=n_classes,
                    )
                else:
                    agg = aggregate_windows_to_segments(
                        y, pr["pred"], pack["seg_keys"], n_classes=n_classes
                    )

                m = agg["segment_metrics"]
                row = {
                    "fold": fold,
                    "acc_paper": float(m["acc_paper"])
                    if m.get("acc_paper") is not None
                    else float("nan"),
                    "abstain_rate": float(m.get("abstain_rate") or 0.0),
                    "n_segments_scored": int(m.get("n_segments_scored") or 0),
                }
                fold_rows.append(row)
                print(
                    f"  {head} f{fold}: acc={row['acc_paper']:.4f} "
                    f"abstain={row['abstain_rate']:.3f}",
                    flush=True,
                )

            mean, std = mean_std([r["acc_paper"] for r in fold_rows])
            ab_m, ab_s = mean_std([r["abstain_rate"] for r in fold_rows])
            sub_pack["metrics"][head] = {
                "acc_paper_mean": mean,
                "acc_paper_std": std,
                "abstain_rate_mean": ab_m,
                "abstain_rate_std": ab_s,
                "folds": fold_rows,
            }
            per_sub_acc[stream.subject_id][head] = [r["acc_paper"] for r in fold_rows]
            md += [
                f"### {head}",
                "",
                f"- Acc_paper: **{mean:.4f}±{std:.4f}**",
                f"- abstain: {ab_m:.3f}±{ab_s:.3f}",
                "",
            ]

        if sub_pack["latency_ms_mean"]:
            sub_pack["latency_ms_mean_fold"] = float(
                np.mean(sub_pack["latency_ms_mean"])
            )
        results["subjects"][stream.subject_id] = sub_pack
        (out_run / f"{stream.subject_id}_summary.json").write_text(
            json.dumps(jsonable(sub_pack), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    macro = {}
    for head in tasks:
        vals = []
        for sid, heads in per_sub_acc.items():
            rows = heads.get(head, [])
            if rows:
                vals.append(float(np.nanmean(rows)))
        m, s = mean_std(vals)
        macro[head] = {"mean": m, "std": s, "per_subject": vals}

    anchor = _anchor_for(arm)
    delta_block: dict = {}
    if "three" in macro:
        cur_three = macro["three"]["per_subject"]
        if arm.anchor:
            main = _paired_delta(cur_three, arm.anchor, head="three")
            if main:
                delta_block["primary"] = main
        for ab_arm in arm.ablation_vs:
            ab = _paired_delta(cur_three, ab_arm, head="three")
            if ab:
                delta_block[f"ablation_vs_{ab_arm}"] = ab
        if not delta_block.get("primary") and anchor:
            a_mean = anchor["three"][0]
            d = macro["three"]["mean"] - a_mean
            delta_block["readonly_S07"] = {
                "three_vs_readonly_S07": d,
                "note": "S07 全量 trial，仅参考，不作主判定",
                "readonly_three": a_mean,
            }

    results["macro"] = macro
    results["delta"] = delta_block
    (out_run / "summary.json").write_text(
        json.dumps(jsonable(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if macro:
        md += ["## Macro", ""]
        for head in tasks:
            m = macro[head]
            md.append(f"- **{head}**: {m['mean']:.4f}±{m['std']:.4f}")
        if delta_block:
            primary = delta_block.get("primary") or delta_block.get("readonly_S07") or {}
            key = next(
                (k for k in primary if k.startswith("three_vs_")),
                "three_vs_paired_anchor",
            )
            md += [
                "",
                f"- ΔThree vs {arm.anchor or 'S07'}: {primary.get(key, float('nan')):+.4f}",
            ]
            if primary.get("n_pass_ge_0.02") is not None:
                md += [
                    f"- 达标数(≥+0.02): {primary.get('n_pass_ge_0.02', '—')}",
                    f"- Wilcoxon p: {primary.get('wilcoxon_p', float('nan')):.4g}",
                ]
            for k, v in delta_block.items():
                if k.startswith("ablation_vs_"):
                    ab_key = next((x for x in v if x.startswith("three_vs_")), "")
                    md.append(
                        f"- 消融 ΔThree vs {v.get('paired_anchor')}: "
                        f"{v.get(ab_key, float('nan')):+.4f}"
                    )
        md.append("")

    report = results_dir / f"{stamp}_shallow_{arm.arm_id}.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_run / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[done] {report}", flush=True)
    return out_run


def main() -> None:
    p = argparse.ArgumentParser(description="S09 OTTA A/B 臂评测")
    p.add_argument("--arm", default="", help="单臂代号如 A3")
    p.add_argument("--arms", default="", help="多臂逗号分隔，覆盖 --arm")
    p.add_argument("--subjects", default="")
    p.add_argument("--tasks", default="task,three")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--run-stamp", default="")
    p.add_argument("--ft-stamp", default="")
    p.add_argument("--conf-tau", type=float, default=None)
    p.add_argument("--g3-top-p", type=float, default=0.5)
    p.add_argument("--rebuild-ref", action="store_true")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()

    arm_text = args.arms or args.arm
    if not arm_text:
        arm_text = "A0,A1,A2,A3,B0,B1,B2,B3,B4"
    arms = parse_arms(arm_text)

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    folds = (0,) if args.smoke else tuple(range(N_FOLDS))
    device = torch.device(args.device)

    need_src = any(a.ea_ref == "src" for a in arms)
    r_ref_src, ref_meta = (None, {})
    if need_src:
        r_ref_src, ref_meta = load_ref_cov_src(
            rebuild=bool(args.rebuild_ref), smoke=bool(args.smoke)
        )

    for arm in arms:
        eval_one_arm(
            arm,
            subjects=subjects,
            tasks=tasks,
            folds=folds,
            device=device,
            batch_size=int(args.batch_size),
            ft_stamp=args.ft_stamp,
            run_stamp=args.run_stamp,
            conf_tau=args.conf_tau,
            g3_top_p=float(args.g3_top_p),
            r_ref_src=r_ref_src,
            ref_meta=ref_meta,
            adabn_version=ADABN_VERSION,
        )


if __name__ == "__main__":
    main()
