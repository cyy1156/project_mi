"""S10-C1：B3 + 每 trial 伪标签全量微调（Q3）。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import ShallowFBCSPNet
from torch.optim import Adam

import _bootstrap  # noqa: F401

from config import (
    ADABN_PREDICT_FIRST,
    ADABN_VERSION,
    FT_STAMP_DEFAULT,
    FT_WEIGHT_ROOT,
    INPUT_PIPELINE,
    N_FOLDS,
    N_TIMES,
    PROTOCOL_OTTA,
    PROTOCOL_VERSION,
    RESULTS_ROOT,
)
from data import iter_subject_streams
from data_split import assert_no_leakage, build_cue_split
from adabn import (
    ADABN_VERSION_DEFAULT,
    freeze_all_params,
    restore_bn_running,
    snapshot_bn_running,
    stream_predict_adabn,
)
from infer import load_ft_fold
from otta_infer import build_eval_tensors
from paired_results import load_paired_summary, subject_acc
from util_metrics import aggregate_windows_to_segments, jsonable, mean_std


C1_CONF_TAU = 0.6
C1_MIN_CONF_FRAC = 0.5
C1_LR = 1e-4


def build_shallow(n_chans, n_times, n_outputs, drop_prob):
    return ShallowFBCSPNet(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, drop_prob=drop_prob
    )


def _resolve_ft_root(ft_stamp: str) -> Path:
    base = FT_WEIGHT_ROOT / "shallow_stieger_ft_half_balbatch_accpaper"
    root = base / (ft_stamp or FT_STAMP_DEFAULT)
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def _trial_order(cue_ids: np.ndarray) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for c in cue_ids.tolist():
        c = int(c)
        if c not in seen:
            seen.add(c)
            order.append(c)
    return order


def _majority(preds: np.ndarray) -> int | None:
    if len(preds) == 0:
        return None
    cnt = Counter(int(p) for p in preds.tolist())
    top = cnt.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return None
    return int(top[0][0])


def _thirds_acc(
    y: np.ndarray,
    pred: np.ndarray,
    seg_keys: np.ndarray,
    trial_order: list[int],
    cue_ids: np.ndarray,
) -> dict[str, float]:
    """按 eval cue 顺序三等分，段级 Acc_paper。"""
    n = len(trial_order)
    if n < 3:
        return {"first": float("nan"), "mid": float("nan"), "last": float("nan")}
    b1 = n // 3
    b2 = 2 * n // 3
    chunks = [trial_order[:b1], trial_order[b1:b2], trial_order[b2:]]
    names = ["first", "mid", "last"]
    out = {}
    for name, cues in zip(names, chunks):
        allow = set(cues)
        m = np.asarray([int(c) in allow for c in cue_ids], dtype=bool)
        if not m.any():
            out[name] = float("nan")
            continue
        agg = aggregate_windows_to_segments(
            y[m], pred[m], seg_keys[m], n_classes=3
        )
        out[name] = float(agg["segment_metrics"]["acc_paper"])
    return out


@torch.no_grad()
def _predict_trial_windows(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    sp = stream_predict_adabn(
        model,
        X,
        device,
        update_bn=True,
        version=ADABN_VERSION_DEFAULT,
        predict_first=ADABN_PREDICT_FIRST,
    )
    return sp.pred, sp.prob_max


def _finetune_step(
    model: nn.Module,
    X: np.ndarray,
    label: int,
    device: torch.device,
) -> None:
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    opt = Adam(model.parameters(), lr=C1_LR)
    crit = nn.CrossEntropyLoss()
    for w in X:
        xb = w[0] if w.ndim == 4 else w
        if xb.ndim == 3 and xb.shape[0] == 1:
            xb = xb[0]
        batch = torch.from_numpy(np.asarray(xb[None, ...], dtype=np.float32)).to(device)
        tgt = torch.tensor([int(label)], device=device, dtype=torch.long)
        opt.zero_grad()
        logits = model(batch)
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        loss = crit(logits, tgt)
        loss.backward()
        opt.step()
    freeze_all_params(model)


def run_c1_subject(
    stream,
    *,
    ft_root: Path,
    head: str,
    fold: int,
    device: torch.device,
    a0_three: float | None,
    b3_three: float | None,
) -> dict:
    split = build_cue_split(stream)
    assert_no_leakage(split)
    pack = build_eval_tensors(stream, split, ea_ref="cal", r_ref_src=None)

    net = load_ft_fold(
        build_shallow,
        ft_subject_root=ft_root / stream.subject_id,
        head=head,
        fold=fold,
        device=device,
    )
    bn0 = snapshot_bn_running(net)
    freeze_all_params(net)

    y = pack["y_three"] if head == "three" else pack["y_task"]
    n_classes = 3 if head == "three" else 2
    X_all = pack["X_eval"]
    cue_ids = pack["cue_ids"]
    seg_keys = pack["seg_keys"]

    trial_order = _trial_order(cue_ids)
    all_pred = np.full(len(X_all), -1, dtype=np.int64)
    n_updates = 0
    n_skip = 0

    for cue in trial_order:
        m = cue_ids == cue
        idx = np.flatnonzero(m)
        X_t = X_all[idx]
        pred_t, pmax_t = _predict_trial_windows(net, X_t, device)
        all_pred[idx] = pred_t

        conf = pmax_t >= C1_CONF_TAU
        if conf.mean() < C1_MIN_CONF_FRAC:
            n_skip += 1
            continue
        pseudo = _majority(pred_t[conf])
        if pseudo is None or pseudo >= n_classes:
            n_skip += 1
            continue
        _finetune_step(net, X_t[conf], pseudo, device)
        n_updates += 1

    restore_bn_running(net, bn0)
    agg = aggregate_windows_to_segments(y, all_pred, seg_keys, n_classes=n_classes)
    acc = float(agg["segment_metrics"]["acc_paper"])
    thirds = _thirds_acc(y, all_pred, seg_keys, trial_order, cue_ids)
    collapse = False
    collapse_reasons: list[str] = []
    if np.isfinite(thirds["first"]) and np.isfinite(thirds["last"]):
        if thirds["last"] < thirds["first"] - 0.03:
            collapse = True
            collapse_reasons.append("drift_last_vs_first")
    if head == "three":
        if a0_three is not None and acc < a0_three:
            collapse = True
            collapse_reasons.append("below_S09_A0_eval_half")
        if b3_three is not None and acc < b3_three - 0.005:
            collapse = True
            collapse_reasons.append("below_S09_B3_eval_half")

    return {
        "acc_paper": acc,
        "n_trial_updates": n_updates,
        "n_trial_skip": n_skip,
        "thirds": thirds,
        "collapse": collapse,
        "collapse_reasons": collapse_reasons,
        "ref_a0_three": a0_three,
        "ref_b3_three": b3_three,
        "segment_metrics": agg["segment_metrics"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="S10-C1 伪标签在线 FT")
    p.add_argument("--subjects", default="")
    p.add_argument("--tasks", default="three")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--ft-stamp", default="")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    folds = (0,) if args.smoke else tuple(range(N_FOLDS))
    device = torch.device(args.device)
    ft_root = _resolve_ft_root(args.ft_stamp)

    paired_a0 = load_paired_summary("A0", head="three")
    paired_b3 = load_paired_summary("B3", head="three")

    results_dir = RESULTS_ROOT / "S10-C1"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = results_dir / f"{stamp}_shallow_C1"
    out_run.mkdir(parents=True, exist_ok=True)

    results = {
        "arm": "C1",
        "protocol": PROTOCOL_OTTA,
        "ft_root": str(ft_root),
        "spec": {
            "protocol_version": PROTOCOL_VERSION,
            "input_pipeline": INPUT_PIPELINE,
            "base": "B3",
            "ea_ref": "cal",
            "adabn_version": ADABN_VERSION,
            "adabn_predict_first": ADABN_PREDICT_FIRST,
            "eval_protocol": "eval_half_causal",
            "collapse_refs": {
                "A0_eval_half": paired_a0.get("path") if paired_a0 else None,
                "B3_eval_half": paired_b3.get("path") if paired_b3 else None,
            },
        },
        "c1_hparams": {
            "conf_tau": C1_CONF_TAU,
            "min_conf_frac": C1_MIN_CONF_FRAC,
            "lr": C1_LR,
            "base": "B3",
        },
        "subjects": {},
    }
    md = [f"# S10-C1 伪标签全量 FT · {stamp}", ""]

    for stream in iter_subject_streams(subjects=subjects):
        print(f"\n=== C1 {stream.subject_id} ===", flush=True)
        sub: dict = {}
        for head in tasks:
            rows = []
            for fold in folds:
                row = run_c1_subject(
                    stream,
                    ft_root=ft_root,
                    head=head,
                    fold=fold,
                    device=device,
                    a0_three=subject_acc(paired_a0, stream.subject_id),
                    b3_three=subject_acc(paired_b3, stream.subject_id),
                )
                row["fold"] = fold
                rows.append(row)
                print(
                    f"  {head} f{fold}: acc={row['acc_paper']:.4f} "
                    f"updates={row['n_trial_updates']} collapse={row['collapse']}",
                    flush=True,
                )
            mean, std = mean_std([r["acc_paper"] for r in rows])
            n_coll = sum(1 for r in rows if r["collapse"])
            sub[head] = {"mean": mean, "std": std, "folds": rows, "n_collapse_folds": n_coll}
        results["subjects"][stream.subject_id] = sub
        (out_run / f"{stream.subject_id}_summary.json").write_text(
            json.dumps(jsonable(sub), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if "three" in sub:
            t = sub["three"]
            md += [
                f"## {stream.subject_id}",
                f"- Three: {t['mean']:.4f}±{t['std']:.4f} collapse_folds={t['n_collapse_folds']}",
                "",
            ]

    (out_run / "summary.json").write_text(
        json.dumps(jsonable(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = results_dir / f"{stamp}_shallow_C1.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[done] {report}", flush=True)


if __name__ == "__main__":
    main()
