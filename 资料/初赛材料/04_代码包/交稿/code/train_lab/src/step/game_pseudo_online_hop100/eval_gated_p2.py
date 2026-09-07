"""P2：游戏伪在线 · 在线质量门控（H0–H3）。

默认与 04 对齐：OpenBMI Acc_paper shallow + 通道重排。
历史对照：`--weight-domain bci2a`（对齐 01，门控阴性）。

方案：`资料/伪在线实验/03_旁路_teachable质量门控/方案.md`
产物写入：`资料/伪在线实验/03_旁路_teachable质量门控/results/`（不覆盖 01）
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

try:
    from braindecode.models import EEGNet
except ImportError:
    from braindecode.models import EEGNetv4 as EEGNet

from channel_remap import OPENBMI_CHANS, remap_windows_to_openbmi
from config import DEFAULT_SESSIONS, N_FOLDS, REPO_ROOT, SESSIONS_ROOT
from gated_segment_metrics import aggregate_windows_to_segments_gated
from infer import load_fold_model, predict_windows
from online_gate import build_gate_keeps, gate_stats
from stream import build_eval_stream
from weights import resolve_accpaper_run, resolve_openbmi_accpaper_run

DOCS_P2 = REPO_ROOT / "资料" / "伪在线实验" / "03_旁路_teachable质量门控"
RESULTS = DOCS_P2 / "results"

EEGNET_F1, EEGNET_D, EEGNET_F2 = 8, 2, 16


def build_shallow(n_chans, n_times, n_outputs, drop_prob):
    return ShallowFBCSPNet(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, drop_prob=drop_prob
    )


def build_eegnet(n_chans, n_times, n_outputs, drop_prob):
    return EEGNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        F1=EEGNET_F1,
        D=EEGNET_D,
        F2=EEGNET_F2,
        drop_prob=drop_prob,
    )


BUILDERS = {"shallow": build_shallow, "eegnet": build_eegnet}


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=0))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def main() -> None:
    p = argparse.ArgumentParser(description="P2 游戏会话在线质量门控")
    p.add_argument("--model", choices=tuple(BUILDERS), default="shallow")
    p.add_argument(
        "--weight-domain",
        choices=("openbmi", "bci2a"),
        default="openbmi",
        help="默认 openbmi（与 04 对齐）；bci2a=历史对照（对齐 01）",
    )
    p.add_argument("--sessions", default=",".join(DEFAULT_SESSIONS))
    p.add_argument("--gates", default="H0,H1,H2,H3")
    p.add_argument("--tasks", default="task,three")
    p.add_argument("--smoke", action="store_true", help="仅 fold0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--run-stamp", default="")
    p.add_argument("--g3-top-p", type=float, default=0.5)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.weight_domain == "openbmi" and args.model != "shallow":
        raise SystemExit("openbmi 域 P2 当前仅支持 --model shallow（与 04 对齐）")

    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    gates = tuple(g.strip() for g in args.gates.split(",") if g.strip())
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    folds = (0,) if args.smoke else tuple(range(N_FOLDS))
    device = torch.device(args.device)
    build_model = BUILDERS[args.model]

    if args.weight_domain == "openbmi":
        run_dir = resolve_openbmi_accpaper_run(
            args.model, run_stamp=args.run_stamp or None
        )
    else:
        run_dir = resolve_accpaper_run(args.model, run_stamp=args.run_stamp or None)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"P2_{args.weight_domain}_gate"
    out_run = RESULTS / f"{stamp}_{args.model}_{tag}"
    out_run.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "phase": "P2",
        "model": args.model,
        "weight_domain": args.weight_domain,
        "weight_run": str(run_dir),
        "aligned_with": "04" if args.weight_domain == "openbmi" else "01",
        "gates": list(gates),
        "sessions": sessions,
        "device": str(device),
        "note": (
            "REST always kept; MI gated by online ERD/laterality on no-zscore windows; "
            + (
                "model input remapped game→OpenBMI"
                if args.weight_domain == "openbmi"
                else "model input = game channel order"
            )
        ),
        "subjects": {},
        "summary": {},
    }

    md = [
        f"# P2 游戏在线质量门控 · {args.model} · {args.weight_domain}",
        "",
        f"- 时间：`{stamp}`",
        f"- 权重域：`{args.weight_domain}`"
        + ("（**与 04 对齐**）" if args.weight_domain == "openbmi" else "（历史对照 · 对齐 01）"),
        f"- 权重：`{run_dir}`",
        f"- 门控：`{', '.join(gates)}`（H1≈P1-G2 阈值）",
        f"- 质量窗：**无** z-score；推理窗：有 z-score"
        + (
            f"；通道重排 → `{', '.join(OPENBMI_CHANS)}`"
            if args.weight_domain == "openbmi"
            else "（游戏通道序）"
        ),
        "",
    ]

    for ses_name in sessions:
        print(f"\n=== {ses_name} ===", flush=True)
        ses_dir = SESSIONS_ROOT / ses_name
        stream = build_eval_stream(ses_dir, zscore_windows=True)
        stream_q = build_eval_stream(ses_dir, zscore_windows=False)
        assert len(stream.X) == len(stream_q.X)
        assert np.array_equal(stream.trial_ids, stream_q.trial_ids)
        assert list(stream.segs) == list(stream_q.segs)

        ch_names = list(stream.meta.get("channels") or [])
        if args.weight_domain == "openbmi":
            X_model = remap_windows_to_openbmi(stream.X, ch_names)
            print(
                f"  remap: {ch_names} -> {OPENBMI_CHANS} X={tuple(X_model.shape)}",
                flush=True,
            )
        else:
            X_model = stream.X

        keeps = build_gate_keeps(
            stream_q.X,
            stream.y_three,
            stream.trial_ids,
            stream.segs,
            ch_names=ch_names,
            top_p=float(args.g3_top_p),
            modes=gates,
        )
        subj = stream.subject_id
        sub_pack: dict = {
            "session": ses_name,
            "n_windows": int(len(stream.X)),
            "src_channels": ch_names,
            "model_channels": list(OPENBMI_CHANS)
            if args.weight_domain == "openbmi"
            else ch_names,
            "gates": {},
        }
        for g, k in keeps.items():
            st = gate_stats(k, stream.segs, stream.seg_keys)
            sub_pack["gates"][g] = st
            print(
                f"  {g}: kept={st['n_kept']}/{st['n_windows']} "
                f"mi_kept={st['n_mi_kept']} mi_seg_abstain={st['n_mi_segments_abstain']}/"
                f"{st['n_mi_segments']} ({st['mi_segment_abstain_rate']:.2f})",
                flush=True,
            )

        md += [f"## {subj} / {ses_name}", ""]
        if args.weight_domain == "openbmi":
            md += [
                f"- 源通道：`{', '.join(ch_names)}`",
                f"- 模型通道：`{', '.join(OPENBMI_CHANS)}`",
                "",
            ]
        task_summ: dict = {}

        for head in tasks:
            n_classes = 2 if head == "task" else 3
            y = stream.y_task if head == "task" else stream.y_three
            per_gate: dict[str, list[dict]] = {g: [] for g in gates}

            for fold in folds:
                print(f"[{subj}] {args.model} {head} fold{fold}", flush=True)
                net = load_fold_model(
                    build_model, run_dir, head=head, fold=fold, device=device
                )
                pred = predict_windows(
                    net, X_model, device, batch_size=int(args.batch_size)
                )
                del net
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                for g in gates:
                    agg = aggregate_windows_to_segments_gated(
                        y, pred, stream.seg_keys, keeps[g], n_classes=n_classes
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
                        "balanced_accuracy": float(m.get("balanced_accuracy") or float("nan")),
                        "recall_left": float(m["recall_left"])
                        if m.get("recall_left") is not None
                        else float("nan"),
                        "recall_right": float(m["recall_right"])
                        if m.get("recall_right") is not None
                        else float("nan"),
                    }
                    per_gate[g].append(row)
                    print(
                        f"  {g}: acc_paper={row['acc_paper']:.4f} "
                        f"abstain={row['abstain_rate']:.3f} "
                        f"scored={row['n_segments_scored']}/{row['n_segments_all']}",
                        flush=True,
                    )

            h0_mean = _mean_std([r["acc_paper"] for r in per_gate.get("H0", [])])[0]
            summ_g = {}
            for g, rows in per_gate.items():
                mean, std = _mean_std([r["acc_paper"] for r in rows])
                ab_m, ab_s = _mean_std([r["abstain_rate"] for r in rows])
                aw_m, _ = _mean_std([r["abstain_as_wrong_acc"] for r in rows])
                summ_g[g] = {
                    "acc_paper_mean": mean,
                    "acc_paper_std": std,
                    "delta_vs_H0": float(mean - h0_mean)
                    if np.isfinite(mean) and np.isfinite(h0_mean)
                    else float("nan"),
                    "abstain_rate_mean": ab_m,
                    "abstain_rate_std": ab_s,
                    "abstain_as_wrong_acc_mean": aw_m,
                    "folds": rows,
                }
            task_summ[head] = summ_g

            md += [
                f"### {head}",
                "",
                "| 代号 | Acc_paper | vs H0 Δ | abstain 率 | abstain计错Acc | scored段/折 |",
                "|------|-----------|---------|------------|----------------|-------------|",
            ]
            for g in gates:
                s = summ_g[g]
                md.append(
                    f"| {g} | {s['acc_paper_mean']:.4f}±{s['acc_paper_std']:.4f} | "
                    f"{s['delta_vs_H0']:+.4f} | "
                    f"{s['abstain_rate_mean']:.3f}±{s['abstain_rate_std']:.3f} | "
                    f"{s['abstain_as_wrong_acc_mean']:.4f} | "
                    f"{_mean_std([float(r['n_segments_scored']) for r in s['folds']])[0]:.1f} |"
                )
            md.append("")

            # 决策短评
            if head == "three" and "H1" in summ_g and "H0" in summ_g:
                d = summ_g["H1"]["delta_vs_H0"]
                ab = summ_g["H1"]["abstain_rate_mean"]
                if d >= 0.02:
                    tip = f"H1 ΔThree={d:+.4f} → 达 +0.02 成功线"
                elif np.isfinite(d) and d >= 0 and ab < 0.40:
                    tip = f"H1 ΔThree={d:+.4f} abstain={ab:.2f} → 可看混淆/附报"
                else:
                    tip = f"H1 ΔThree={d:+.4f} abstain={ab:.2f} → 未达主成功线"
                md += [f"- **决策（{subj} Three）**：{tip}", ""]
                task_summ[f"{head}_decision"] = tip

        results["subjects"][subj] = {**sub_pack, "metrics": task_summ}
        (out_run / f"{subj}_summary.json").write_text(
            json.dumps(_jsonable(results["subjects"][subj]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 跨被试决策摘要（分列，不夺冠）
    md += ["## 汇总决策", ""]
    for subj, pack in results["subjects"].items():
        tip = pack.get("metrics", {}).get("three_decision")
        if tip:
            md.append(f"- **{subj}**：{tip}")
    md.append("")
    md.append("成功线（方案）：相对 H0 Acc_paper ≥ +0.02（分人），或混淆改善且 abstain<40%。")
    md.append("")

    (out_run / "summary.json").write_text(
        json.dumps(_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path = RESULTS / f"{stamp}_{args.model}_{tag}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_run / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[done] {md_path}", flush=True)
    print(f"[done] {out_run / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
