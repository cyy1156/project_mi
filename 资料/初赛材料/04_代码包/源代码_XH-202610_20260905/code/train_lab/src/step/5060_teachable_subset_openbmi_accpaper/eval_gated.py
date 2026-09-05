"""P1：OpenBMI 离线质量门控复评（投票前丢窗；空试次 abstain）。

对照伪在线方案：`资料/伪在线实验/03_旁路_teachable质量门控/方案.md`
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
CODE_ROOT = HERE.parents[3]
TRAIN_LAB = CODE_ROOT / "train_lab"
PRE_ROOT = CODE_ROOT / "preprocess_lab"
REPO = CODE_ROOT.parent
HOP100 = STEP / "baselines_2s_hop100"
OLD = STEP / "baselines_single"

for p in (STEP, PRE_ROOT, HOP100, OLD):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
_here = str(HERE)
if _here in sys.path:
    sys.path.remove(_here)
sys.path.insert(0, _here)

from data_paths import resolve_data
from eval_subset import (
    IndexDS,
    _jsonable,
    _mean_std,
    build_model,
    predict,
    resolve_run_dir,
)
from gated_metrics import aggregate_windows_to_trials_gated, build_g3_keep_mask
from perf_loader import apply_runtime_threads, configure_cuda_backends, make_loader
from shared_hparams import SHARED
from src.common.steps.split_subjects import iter_subject_kfold
from teachable_io import load_masks, resolve_teachable_paths

RESULTS_ROOT = REPO / "资料" / "伪在线实验" / "03_旁路_teachable质量门控" / "results"


def _mi_or_rest_keep(y_three: np.ndarray, mi_quality: np.ndarray) -> np.ndarray:
    """REST 全留；MI 仅质量过门窗保留。"""
    y_three = np.asarray(y_three).astype(int)
    rest = y_three == 0
    return rest | (np.asarray(mi_quality, dtype=bool) & ~rest)


def eval_fold_all_gates(
    *,
    model_name: str,
    task: str,
    fold: int,
    run_dir: Path,
    X,
    y,
    y_three,
    subjects,
    trial_ids,
    test_mask: np.ndarray,
    gate_keeps: dict[str, np.ndarray],
    device,
    hp,
    x_path: str,
) -> dict[str, dict]:
    n_classes = 2 if task == "task" else 3
    ckpt_name = "best_task.pt" if task == "task" else "best_three.pt"
    ckpt_path = run_dir / task / f"fold{fold}" / ckpt_name
    model = build_model(model_name, n_classes, hp.n_times_expected, hp.drop_prob).to(device)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)

    indices = np.flatnonzero(np.asarray(test_mask, dtype=bool)).astype(np.int64)
    loader = make_loader(
        IndexDS(X, y, indices, x_path=x_path),
        batch_size=hp.batch_eval,
        shuffle=False,
        num_workers=hp.num_workers,
        pin_memory=hp.pin_memory,
        persistent_workers=hp.persistent_workers and hp.num_workers > 0,
        prefetch_factor=hp.prefetch_factor,
    )
    yt, yp = predict(model, loader, device)
    subs = subjects[indices]
    tids = trial_ids[indices]

    out = {}
    for gname, gkeep_full in gate_keeps.items():
        keep = np.asarray(gkeep_full[indices], dtype=bool)
        agg = aggregate_windows_to_trials_gated(
            yt, yp, subs, tids, keep, n_classes=n_classes
        )
        m = agg["metrics"]
        out[gname] = {
            "acc_paper": float(m["acc_paper"]) if m["acc_paper"] is not None else float("nan"),
            "abstain_rate": float(m["abstain_rate"]),
            "abstain_as_wrong_acc": float(m["abstain_as_wrong_acc"])
            if m.get("abstain_as_wrong_acc") is not None
            else float("nan"),
            "n_trials_all": int(m["n_trials_all"]),
            "n_trials_scored": int(m["n_trials_scored"]),
            "n_trials_abstain": int(m["n_trials_abstain"]),
            "n_windows": int(m["n_windows"]),
            "n_windows_kept": int(m["n_windows_kept"]),
            "recall_left": float(m.get("recall_left", float("nan")))
            if m.get("recall_left") is not None
            else float("nan"),
            "recall_right": float(m.get("recall_right", float("nan")))
            if m.get("recall_right") is not None
            else float("nan"),
            "cm": m.get("cm"),
        }
        print(
            f"[{model_name}/{task}/fold{fold}/{gname}] "
            f"acc_paper={out[gname]['acc_paper']} "
            f"abstain={out[gname]['abstain_rate']:.3f} "
            f"scored={out[gname]['n_trials_scored']}/{out[gname]['n_trials_all']}",
            flush=True,
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="P1 OpenBMI 质量门控复评")
    p.add_argument("--model", choices=("shallow", "eegnet"), default="shallow")
    p.add_argument("--run-dir", default="")
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--tasks", default="task,three")
    p.add_argument("--gates", default="G0,G1,G2,G3", help="逗号分隔 G0–G3")
    p.add_argument("--g3-top-p", type=float, default=0.5)
    p.add_argument("--teachable-mask", default="")
    p.add_argument("--skip-g3", action="store_true", help="跳过 G3（较慢）")
    args = p.parse_args()

    hp = SHARED
    apply_runtime_threads(hp.torch_num_threads)
    configure_cuda_backends(
        cudnn_benchmark=hp.cudnn_benchmark, deterministic=hp.deterministic
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, mp = resolve_teachable_paths(None, args.teachable_mask or None)
    data_dir, prefix = resolve_data(hp.data_tag)
    x_path = str(data_dir / f"{prefix}_X.npy")
    X = np.load(x_path, mmap_mode="r")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    trial_ids = np.load(data_dir / f"{prefix}_trial_id.npy")
    y_task = np.load(data_dir / f"{prefix}_y_task.npy")
    y_three = np.load(data_dir / f"{prefix}_y_three.npy")
    masks = load_masks(len(X), mp)

    gates_wanted = [g.strip() for g in args.gates.split(",") if g.strip()]
    if args.skip_g3 and "G3" in gates_wanted:
        gates_wanted = [g for g in gates_wanted if g != "G3"]

    print("[gate] building keep masks …", flush=True)
    gate_keeps: dict[str, np.ndarray] = {}
    if "G0" in gates_wanted:
        gate_keeps["G0"] = np.ones(len(X), dtype=bool)
    if "G1" in gates_wanted:
        gate_keeps["G1"] = _mi_or_rest_keep(y_three, masks["high_lat_eval"])
    if "G2" in gates_wanted:
        gate_keeps["G2"] = _mi_or_rest_keep(y_three, masks["teachable"])
    if "G3" in gates_wanted:
        print(f"[gate] G3 top_p={args.g3_top_p}（全库试次内相对门控，较慢）…", flush=True)
        gate_keeps["G3"] = build_g3_keep_mask(
            X, y_three, trial_ids, top_p=float(args.g3_top_p)
        )
        print(f"  G3 kept={int(gate_keeps['G3'].sum())}/{len(X)}", flush=True)

    for g, k in gate_keeps.items():
        print(
            f"  {g}: kept_windows={int(k.sum())} "
            f"mi_kept={int((k & (y_three > 0)).sum())} "
            f"rest_kept={int((k & (y_three == 0)).sum())}",
            flush=True,
        )

    run_dir = resolve_run_dir(args.model, args.run_dir or None)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = (
        TRAIN_LAB
        / "out"
        / "5060_teachable_subset_openbmi_accpaper"
        / f"gate_p1_{args.model}"
        / stamp
    )
    out_run.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    fold_iter = list(
        iter_subject_kfold(
            subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
        )
    )
    if args.max_folds > 0:
        fold_iter = fold_iter[: args.max_folds]

    results: dict = {
        "phase": "P1",
        "model": args.model,
        "run_dir": str(run_dir),
        "teachable_mask": str(mp),
        "gates": list(gate_keeps.keys()),
        "g3_top_p": float(args.g3_top_p),
        "device": str(device),
        "note": "REST windows always kept; MI filtered by gate; empty trial=abstain",
        "folds": {},
        "summary": {},
    }

    for task in tasks:
        y = y_task if task == "task" else y_three
        per_gate: dict[str, list[dict]] = {g: [] for g in gate_keeps}
        for fd in fold_iter:
            fold = int(fd["fold"])
            pack = eval_fold_all_gates(
                model_name=args.model,
                task=task,
                fold=fold,
                run_dir=run_dir,
                X=X,
                y=y,
                y_three=y_three,
                subjects=subjects,
                trial_ids=trial_ids,
                test_mask=fd["masks"]["test"],
                gate_keeps=gate_keeps,
                device=device,
                hp=hp,
                x_path=x_path,
            )
            results["folds"].setdefault(task, {})[f"fold{fold}"] = pack
            for g, row in pack.items():
                per_gate[g].append(row)

        summ = {}
        g0_mean = _mean_std([x["acc_paper"] for x in per_gate.get("G0", [])])[0]
        for g, rows in per_gate.items():
            mean, std = _mean_std([x["acc_paper"] for x in rows])
            ab_m, ab_s = _mean_std([x["abstain_rate"] for x in rows])
            aw_m, _ = _mean_std([x["abstain_as_wrong_acc"] for x in rows])
            summ[g] = {
                "acc_paper_mean": mean,
                "acc_paper_std": std,
                "delta_vs_G0": float(mean - g0_mean)
                if np.isfinite(mean) and np.isfinite(g0_mean)
                else float("nan"),
                "abstain_rate_mean": ab_m,
                "abstain_rate_std": ab_s,
                "abstain_as_wrong_acc_mean": aw_m,
                "n_trials_scored_mean": _mean_std(
                    [float(x["n_trials_scored"]) for x in rows]
                )[0],
            }
        results["summary"][task] = summ

        # 决策（Three）
        if task == "three" and "G0" in summ:
            advice = []
            for g in ("G1", "G2"):
                if g not in summ:
                    continue
                d = summ[g]["delta_vs_G0"]
                if d >= 0.03:
                    advice.append(f"{g} ΔThree={d:+.4f} → 达 P1 成功线，可开 P2")
                else:
                    advice.append(f"{g} ΔThree={d:+.4f} → 未达 +0.03")
            results["summary"]["three_decision"] = advice

    (out_run / "summary.json").write_text(
        json.dumps(_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    res_json = RESULTS_ROOT / f"{stamp}_{args.model}_P1_gate.json"
    res_json.write_text(
        json.dumps(_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        f"# P1 OpenBMI 质量门控 · {args.model}",
        "",
        f"- 时间：`{stamp}`",
        f"- 权重：`{run_dir}`",
        f"- mask：`{mp}`",
        f"- 约定：REST 窗一律保留；MI 按门控；空试次 abstain（不计 Acc_paper 分母）",
        "",
    ]
    for task in tasks:
        lines += [
            f"## {task}",
            "",
            "| 代号 | Acc_paper | vs G0 Δ | abstain 率 | abstain计错Acc | scored试次/折 |",
            "|------|-----------|---------|------------|----------------|---------------|",
        ]
        for g in gate_keeps:
            s = results["summary"][task][g]
            lines.append(
                f"| {g} | {s['acc_paper_mean']:.4f}±{s['acc_paper_std']:.4f} | "
                f"{s['delta_vs_G0']:+.4f} | "
                f"{s['abstain_rate_mean']:.3f}±{s['abstain_rate_std']:.3f} | "
                f"{s['abstain_as_wrong_acc_mean']:.4f} | "
                f"{s['n_trials_scored_mean']:.1f} |"
            )
        lines.append("")
    if results["summary"].get("three_decision"):
        lines += ["### Three 决策", ""]
        for a in results["summary"]["three_decision"]:
            lines.append(f"- {a}")
        lines.append("")

    md_path = RESULTS_ROOT / f"{stamp}_{args.model}_P1_gate.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {out_run / 'summary.json'}", flush=True)
    print(f"[done] {md_path}", flush=True)


if __name__ == "__main__":
    main()
