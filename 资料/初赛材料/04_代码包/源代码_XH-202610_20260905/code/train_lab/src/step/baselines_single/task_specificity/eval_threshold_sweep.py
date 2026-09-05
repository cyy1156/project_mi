"""对已保存的 best_task.pt 做 P(任务)≥τ 阈值扫描（不训练）。

默认在各折 val 上选 τ（Rec≥min_recall 中 BalAcc 最高），再到 test 报告。
当前实现：ShallowFBCSPNet + 与 baseline_shallow 相同的五折划分。

用法：
  python eval_threshold_sweep.py --run_dir <.../run_xxx> --data merged_2s
  python eval_threshold_sweep.py --run_dir <.../run_xxx> --data merged_2s --min_recall 0.75
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import ShallowFBCSPNet
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent  # .../task_specificity
BASELINES_DIR = HERE.parent  # .../baselines_single
STEP_DIR = BASELINES_DIR.parent  # .../step
CODE_ROOT = HERE.parents[4]  # .../code
REPO_ROOT = CODE_ROOT.parent
PRE_ROOT = CODE_ROOT / "preprocess_lab"

for p in (HERE, BASELINES_DIR, STEP_DIR, PRE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared_hparams import SHARED
from data_paths import resolve_data
from dataset import ArrayTaskDataset
from metrics import binary_task_metrics
from src.common.steps.split_subjects import (
    iter_subject_kfold,
    iter_subject_kfold_stratified_by_dataset,
)


@torch.no_grad()
def collect_proba(model: nn.Module, loader: DataLoader, device: torch.device):
    """返回 y_true (N,), p_task (N,) = softmax 的任务类概率。"""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        ps.append(prob)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def metrics_at_tau(y_true: np.ndarray, p_task: np.ndarray, tau: float) -> dict:
    y_pred = (p_task >= tau).astype(int)
    return binary_task_metrics(y_true, y_pred)


def sweep_taus(y_true, p_task, taus=None) -> list[dict]:
    if taus is None:
        taus = [round(float(x), 2) for x in np.arange(0.30, 0.81, 0.05)]
    rows = []
    for tau in taus:
        m = metrics_at_tau(y_true, p_task, tau)
        rows.append(
            {
                "tau": float(tau),
                "specificity": float(m["specificity"]),
                "recall": float(m["recall"]),
                "balanced_accuracy": float(m["balanced_accuracy"]),
                "f1": float(m["f1"]),
                "accuracy": float(m["accuracy"]),
            }
        )
    return rows


def pick_tau_on_val(
    rows: list[dict],
    *,
    min_recall: float = 0.75,
    min_spec: float = 0.40,
) -> dict:
    """优先：Rec≥min_recall 且 Spec≥min_spec 中 BalAcc 最高；
    否则 Rec≥min_recall 中 BalAcc 最高；再否则全局 BalAcc 最高。
    """
    both = [r for r in rows if r["recall"] >= min_recall and r["specificity"] >= min_spec]
    if both:
        return max(both, key=lambda r: r["balanced_accuracy"])
    ok_rec = [r for r in rows if r["recall"] >= min_recall]
    if ok_rec:
        return max(ok_rec, key=lambda r: r["balanced_accuracy"])
    return max(rows, key=lambda r: r["balanced_accuracy"])


def iter_folds(subjects: np.ndarray, data_tag: str):
    hp = SHARED
    if data_tag.startswith("merged"):
        return iter_subject_kfold_stratified_by_dataset(
            subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
        )
    return iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    )


def build_model(n_times: int, drop_prob: float) -> nn.Module:
    return ShallowFBCSPNet(
        n_chans=8,
        n_outputs=2,
        n_times=n_times,
        drop_prob=drop_prob,
    )


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def run_sweep(
    run_dir: Path,
    data_tag: str,
    *,
    min_recall: float,
    min_spec: float,
    device: torch.device,
) -> dict:
    data_dir, prefix = resolve_data(data_tag)
    X = np.load(data_dir / f"{prefix}_X.npy")
    y = np.load(data_dir / f"{prefix}_y_task.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    assert len(X) == len(y) == len(subjects)

    fold_infos = list(iter_folds(subjects, data_tag))
    fold_rows = []
    for info in fold_infos:
        fold = int(info["fold"])
        masks = info["masks"]
        ckpt_path = run_dir / "task" / f"fold{fold}" / "best_task.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        hp = ckpt.get("hparams") or {}
        drop_prob = float(hp.get("drop_prob", SHARED.drop_prob))
        # ArrayTaskDataset 会把 (N,1,8,T) → (N,8,T)
        if X.ndim == 4:
            n_times = int(X.shape[-1])
        else:
            n_times = int(X.shape[-1])

        model = build_model(n_times=n_times, drop_prob=drop_prob).to(device)
        model.load_state_dict(ckpt["model"])

        val_loader = DataLoader(
            ArrayTaskDataset(X[masks["val"]], y[masks["val"]]),
            batch_size=SHARED.batch_eval,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            ArrayTaskDataset(X[masks["test"]], y[masks["test"]]),
            batch_size=SHARED.batch_eval,
            shuffle=False,
            num_workers=0,
        )

        y_val, p_val = collect_proba(model, val_loader, device)
        y_te, p_te = collect_proba(model, test_loader, device)
        rows_val = sweep_taus(y_val, p_val)
        rows_te = sweep_taus(y_te, p_te)
        chosen = pick_tau_on_val(rows_val, min_recall=min_recall, min_spec=min_spec)
        m_te_at_chosen = metrics_at_tau(y_te, p_te, chosen["tau"])
        m_te_at_05 = metrics_at_tau(y_te, p_te, 0.5)

        # val 上是否存在双过关 τ
        val_pass = [
            r
            for r in rows_val
            if r["recall"] >= min_recall and r["specificity"] >= min_spec
        ]
        te_pass = [
            r
            for r in rows_te
            if r["recall"] >= min_recall and r["specificity"] >= min_spec
        ]

        fold_rows.append(
            {
                "fold": fold,
                "ckpt": str(ckpt_path),
                "model_name": ckpt.get("model_name"),
                "val_curve": rows_val,
                "test_curve": rows_te,
                "chosen_on_val": chosen,
                "test_at_chosen_tau": {
                    "tau": float(chosen["tau"]),
                    "specificity": float(m_te_at_chosen["specificity"]),
                    "recall": float(m_te_at_chosen["recall"]),
                    "balanced_accuracy": float(m_te_at_chosen["balanced_accuracy"]),
                    "f1": float(m_te_at_chosen["f1"]),
                    "accuracy": float(m_te_at_chosen["accuracy"]),
                },
                "test_at_tau05": {
                    "tau": 0.5,
                    "specificity": float(m_te_at_05["specificity"]),
                    "recall": float(m_te_at_05["recall"]),
                    "balanced_accuracy": float(m_te_at_05["balanced_accuracy"]),
                    "f1": float(m_te_at_05["f1"]),
                    "accuracy": float(m_te_at_05["accuracy"]),
                },
                "val_has_pass_tau": bool(val_pass),
                "test_has_pass_tau": bool(te_pass),
                "best_test_pass_tau": (
                    max(te_pass, key=lambda r: r["balanced_accuracy"]) if te_pass else None
                ),
            }
        )
        c = fold_rows[-1]["test_at_chosen_tau"]
        print(
            f"fold{fold}: val→τ={chosen['tau']:.2f} "
            f"(val Spec={chosen['specificity']:.3f} Rec={chosen['recall']:.3f} "
            f"Bal={chosen['balanced_accuracy']:.3f}) | "
            f"test@τ Spec={c['specificity']:.3f} Rec={c['recall']:.3f} "
            f"Bal={c['balanced_accuracy']:.3f} | "
            f"test@0.5 Spec={m_te_at_05['specificity']:.3f} "
            f"Rec={m_te_at_05['recall']:.3f} Bal={m_te_at_05['balanced_accuracy']:.3f}",
            flush=True,
        )

    def agg(key_path: str):
        # key_path like test_at_chosen_tau.specificity
        parts = key_path.split(".")
        vals = []
        for r in fold_rows:
            cur = r
            for p in parts:
                cur = cur[p]
            vals.append(float(cur))
        m, s = _mean_std(vals)
        return {"mean": m, "std": s, "values": vals}

    summary = {
        "run_dir": str(run_dir),
        "data_tag": data_tag,
        "min_recall": min_recall,
        "min_spec": min_spec,
        "stamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_folds": len(fold_rows),
        "agg_test_at_chosen_tau": {
            "specificity": agg("test_at_chosen_tau.specificity"),
            "recall": agg("test_at_chosen_tau.recall"),
            "balanced_accuracy": agg("test_at_chosen_tau.balanced_accuracy"),
            "f1": agg("test_at_chosen_tau.f1"),
            "tau": agg("test_at_chosen_tau.tau"),
        },
        "agg_test_at_tau05": {
            "specificity": agg("test_at_tau05.specificity"),
            "recall": agg("test_at_tau05.recall"),
            "balanced_accuracy": agg("test_at_tau05.balanced_accuracy"),
            "f1": agg("test_at_tau05.f1"),
        },
        "folds_val_has_pass": sum(1 for r in fold_rows if r["val_has_pass_tau"]),
        "folds_test_has_pass": sum(1 for r in fold_rows if r["test_has_pass_tau"]),
        "folds": fold_rows,
    }
    return summary


def write_md(summary: dict, out_md: Path) -> None:
    a = summary["agg_test_at_chosen_tau"]
    b = summary["agg_test_at_tau05"]
    lines = [
        f"# 阈值扫描（{summary['stamp']}）",
        "",
        f"- run_dir：`{summary['run_dir']}`",
        f"- data：`{summary['data_tag']}`",
        f"- 选 τ 规则：val 上优先 Spec≥{summary['min_spec']} 且 Rec≥{summary['min_recall']}，取 BalAcc 最大",
        "",
        "## 五折汇总（Test）",
        "",
        f"- **τ=0.5（训练默认）**：Spec `{b['specificity']['mean']:.4f}±{b['specificity']['std']:.4f}` | "
        f"Rec `{b['recall']['mean']:.4f}±{b['recall']['std']:.4f}` | "
        f"BalAcc `{b['balanced_accuracy']['mean']:.4f}±{b['balanced_accuracy']['std']:.4f}`",
        f"- **val 选定 τ 后**：τ `{a['tau']['mean']:.2f}±{a['tau']['std']:.2f}` | "
        f"Spec `{a['specificity']['mean']:.4f}±{a['specificity']['std']:.4f}` | "
        f"Rec `{a['recall']['mean']:.4f}±{a['recall']['std']:.4f}` | "
        f"BalAcc `{a['balanced_accuracy']['mean']:.4f}±{a['balanced_accuracy']['std']:.4f}`",
        f"- val 存在双过关 τ 的折数：`{summary['folds_val_has_pass']}/5`",
        f"- test 曲线上存在双过关 τ 的折数：`{summary['folds_test_has_pass']}/5`",
        "",
        "## 各折",
        "",
    ]
    for r in summary["folds"]:
        c = r["test_at_chosen_tau"]
        z = r["test_at_tau05"]
        lines.extend(
            [
                f"### Fold {r['fold']}",
                f"- val 选 τ=`{r['chosen_on_val']['tau']:.2f}` "
                f"(val Spec={r['chosen_on_val']['specificity']:.4f} "
                f"Rec={r['chosen_on_val']['recall']:.4f} "
                f"BalAcc={r['chosen_on_val']['balanced_accuracy']:.4f})",
                f"- test@选中τ：Spec=`{c['specificity']:.4f}` Rec=`{c['recall']:.4f}` "
                f"BalAcc=`{c['balanced_accuracy']:.4f}` F1=`{c['f1']:.4f}`",
                f"- test@0.5：Spec=`{z['specificity']:.4f}` Rec=`{z['recall']:.4f}` "
                f"BalAcc=`{z['balanced_accuracy']:.4f}`",
                f"- val 双过关τ：`{r['val_has_pass_tau']}` | test 曲线双过关τ：`{r['test_has_pass_tau']}`",
                "",
            ]
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Task 阈值扫描（Shallow best_task.pt）")
    p.add_argument("--run_dir", type=Path, required=True, help="含 task/fold*/best_task.pt")
    p.add_argument("--data", default=SHARED.data_tag)
    p.add_argument("--min_recall", type=float, default=0.75)
    p.add_argument("--min_spec", type=float, default=0.40)
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir 不存在: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} run_dir={run_dir}", flush=True)
    summary = run_sweep(
        run_dir,
        args.data,
        min_recall=args.min_recall,
        min_spec=args.min_spec,
        device=device,
    )

    out_json = run_dir / "threshold_sweep.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    stamp = summary["stamp"]
    model_name = Path(run_dir).parent.parent.name  # .../model/data/run_xxx
    records = REPO_ROOT / "资料" / "模型训练" / "runs" / f"{stamp}_{model_name}_tau_sweep"
    out_md = records / f"{model_name}_阈值扫描.md"
    write_md(summary, out_md)

    # 同步最新入口
    latest = REPO_ROOT / "资料" / "模型训练" / "五折实验记录_最新.md"
    rel = out_md.relative_to(REPO_ROOT / "资料" / "模型训练").as_posix()
    latest.write_text(
        f"# 最新实验入口\n\n本次记录：[`{rel}`](./{rel})\n\n"
        f"JSON：`{out_json}`\n",
        encoding="utf-8",
    )

    a = summary["agg_test_at_chosen_tau"]
    b = summary["agg_test_at_tau05"]
    print(
        f"\n[SWEEP] test@0.5 Spec {b['specificity']['mean']:.4f} Rec {b['recall']['mean']:.4f} "
        f"Bal {b['balanced_accuracy']['mean']:.4f}"
    )
    print(
        f"[SWEEP] test@val-τ Spec {a['specificity']['mean']:.4f} Rec {a['recall']['mean']:.4f} "
        f"Bal {a['balanced_accuracy']['mean']:.4f} | meanτ={a['tau']['mean']:.2f}"
    )
    print(f"json={out_json}")
    print(f"md={out_md}")


if __name__ == "__main__":
    main()
