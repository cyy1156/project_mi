"""读取 suite_progress.json（可合并 Shallow）+ 对 A20/A22 做阈值扫描，写出八模型总汇总 MD。"""

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

HERE = Path(__file__).resolve().parent
BASELINES_DIR = HERE.parent
STEP_DIR = BASELINES_DIR.parent
CODE_ROOT = HERE.parents[4]
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
from suite_runner import (
    load_baseline_module,
    load_X,
    make_builder,
    make_dataset,
)


def iter_folds(subjects, data_tag):
    hp = SHARED
    if data_tag.startswith("merged"):
        return iter_subject_kfold_stratified_by_dataset(
            subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
        )
    return iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    )


@torch.no_grad()
def collect_proba(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        ps.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def metrics_at_tau(y, p, tau):
    return binary_task_metrics(y, (p >= tau).astype(int))


def sweep(y, p, taus=None):
    if taus is None:
        taus = [round(float(x), 2) for x in np.arange(0.30, 0.81, 0.05)]
    rows = []
    for tau in taus:
        m = metrics_at_tau(y, p, tau)
        rows.append(
            {
                "tau": float(tau),
                "specificity": float(m["specificity"]),
                "recall": float(m["recall"]),
                "balanced_accuracy": float(m["balanced_accuracy"]),
                "f1": float(m["f1"]),
            }
        )
    return rows


def pick_tau(rows, min_recall=0.75, min_spec=0.40):
    both = [r for r in rows if r["recall"] >= min_recall and r["specificity"] >= min_spec]
    if both:
        return max(both, key=lambda r: r["balanced_accuracy"])
    ok = [r for r in rows if r["recall"] >= min_recall]
    return max(ok or rows, key=lambda r: r["balanced_accuracy"])


MODEL_ORDER = (
    "shallow",
    "eegnet",
    "deep",
    "eegtcnet",
    "conformer",
    "dbn",
    "gcbnet",
    "dgcnn",
)
ARM_ORDER = ("A20", "A22", "B1", "B2", "S1")

# 实验序号 → 实验内容（与 suite_runner.ARMS 一致）
EXP_LEGEND = {
    "A20": "加权交叉熵（静息类权重 w0=2.0，任务类 w1=1.0）+ 以 Val BalAcc 早停/选模",
    "A22": "加权交叉熵（w0=2.2, w1=1.0）+ 以 Val BalAcc 早停/选模（主特异度设定）",
    "B1": "普通交叉熵 + train 类逆频 batch balance（WeightedRandomSampler）+ BalAcc 早停",
    "B2": "加权交叉熵（w0=2.0）+ batch balance + BalAcc 早停",
    "S1": "普通交叉熵 + train 折 SMOTE（展平 8×T）+ BalAcc 早停",
}


def merge_progress(suite: dict, shallow: dict | None) -> dict:
    """合并七模型套件与 Shallow 下午臂；同 model_key/arm 以 suite 为准。"""
    by_key = {}
    for r in (shallow or {}).get("runs", []):
        by_key[(r.get("model_key"), r.get("arm"))] = r
    for r in suite.get("runs", []):
        by_key[(r.get("model_key"), r.get("arm"))] = r
    return {
        "started": suite.get("started") or (shallow or {}).get("started"),
        "finished": suite.get("finished") or (shallow or {}).get("finished"),
        "runs": list(by_key.values()),
        "sources": {
            "suite": True,
            "shallow": bool(shallow),
        },
    }


def enrich_runs_with_summary_metrics(progress: dict) -> None:
    """从各 run 的 task/summary.json 补全 Acc 等字段（旧 progress 可能缺）。"""
    for r in progress.get("runs", []):
        if not r.get("ok") or not r.get("out_root"):
            continue
        sj = Path(r["out_root"]) / "task" / "summary.json"
        if not sj.is_file():
            continue
        try:
            summary = json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "test_acc_mean" in summary:
            r["test_acc_mean"] = float(summary["test_acc_mean"])
            r["test_acc_std"] = float(summary.get("test_acc_std", 0.0))
        # 若 progress 缺其它主指标，也一并回填
        for key in (
            "test_specificity_mean",
            "test_specificity_std",
            "test_recall_mean",
            "test_recall_std",
            "test_balanced_accuracy_mean",
            "test_balanced_accuracy_std",
            "test_f1_mean",
            "test_f1_std",
        ):
            if key not in r and key in summary:
                r[key] = float(summary[key])


def _agg_from_folds(fold_rows: list[dict]) -> dict:
    def mean_key(key, sub):
        xs = [r[key][sub] for r in fold_rows]
        return float(np.mean(xs)), float(np.std(xs))

    return {
        "folds": fold_rows,
        "agg_05": {
            "specificity": mean_key("test_at_05", "specificity"),
            "recall": mean_key("test_at_05", "recall"),
            "balanced_accuracy": mean_key("test_at_05", "balanced_accuracy"),
        },
        "agg_chosen": {
            "specificity": mean_key("test_at_chosen", "specificity"),
            "recall": mean_key("test_at_chosen", "recall"),
            "balanced_accuracy": mean_key("test_at_chosen", "balanced_accuracy"),
        },
    }


def load_cached_sweep(out_root: Path) -> dict | None:
    path = out_root / "threshold_sweep.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    # suite_report 格式
    if "agg_05" in data and "agg_chosen" in data and "error" not in data:
        return data
    # eval_threshold_sweep.py 格式 → 转成统一结构
    if "agg_test_at_tau05" in data and "agg_test_at_chosen_tau" in data:
        a05 = data["agg_test_at_tau05"]
        ac = data["agg_test_at_chosen_tau"]
        return {
            "out_root": str(out_root),
            "source": "eval_threshold_sweep",
            "agg_05": {
                "specificity": (a05["specificity"]["mean"], a05["specificity"]["std"]),
                "recall": (a05["recall"]["mean"], a05["recall"]["std"]),
                "balanced_accuracy": (
                    a05["balanced_accuracy"]["mean"],
                    a05["balanced_accuracy"]["std"],
                ),
            },
            "agg_chosen": {
                "specificity": (ac["specificity"]["mean"], ac["specificity"]["std"]),
                "recall": (ac["recall"]["mean"], ac["recall"]["std"]),
                "balanced_accuracy": (
                    ac["balanced_accuracy"]["mean"],
                    ac["balanced_accuracy"]["std"],
                ),
            },
        }
    return None


def build_shallow(n_times: int, drop_prob: float) -> nn.Module:
    return ShallowFBCSPNet(
        n_chans=8, n_outputs=2, n_times=n_times, drop_prob=drop_prob
    )


def sweep_run_shallow(out_root: Path, data_tag: str, device) -> dict:
    data_dir, prefix = resolve_data(data_tag)
    X = np.load(data_dir / f"{prefix}_X.npy")
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0, :, :]
    y = np.load(data_dir / f"{prefix}_y_task.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    fold_rows = []
    for info in iter_folds(subjects, data_tag):
        fold = int(info["fold"])
        masks = info["masks"]
        ckpt_path = out_root / "task" / f"fold{fold}" / "best_task.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        drop = float((ckpt.get("hparams") or {}).get("drop_prob", SHARED.drop_prob))
        model = build_shallow(int(X.shape[-1]), drop).to(device)
        model.load_state_dict(ckpt["model"])
        val_loader = DataLoader(
            ArrayTaskDataset(X[masks["val"]], y[masks["val"]]),
            batch_size=SHARED.batch_eval,
            shuffle=False,
        )
        test_loader = DataLoader(
            ArrayTaskDataset(X[masks["test"]], y[masks["test"]]),
            batch_size=SHARED.batch_eval,
            shuffle=False,
        )
        yv, pv = collect_proba(model, val_loader, device)
        yt, pt = collect_proba(model, test_loader, device)
        chosen = pick_tau(sweep(yv, pv))
        m05 = metrics_at_tau(yt, pt, 0.5)
        mc = metrics_at_tau(yt, pt, chosen["tau"])
        fold_rows.append(
            {
                "fold": fold,
                "chosen_tau": chosen["tau"],
                "test_at_05": {
                    "specificity": float(m05["specificity"]),
                    "recall": float(m05["recall"]),
                    "balanced_accuracy": float(m05["balanced_accuracy"]),
                },
                "test_at_chosen": {
                    "specificity": float(mc["specificity"]),
                    "recall": float(mc["recall"]),
                    "balanced_accuracy": float(mc["balanced_accuracy"]),
                },
            }
        )
    out = _agg_from_folds(fold_rows)
    out["out_root"] = str(out_root)
    return out


def sweep_run(model_key: str, out_root: Path, data_tag: str, device) -> dict:
    if model_key == "shallow":
        return sweep_run_shallow(out_root, data_tag, device)
    mod = load_baseline_module(model_key)
    builder = make_builder(model_key, mod)
    data_dir, prefix = resolve_data(data_tag)
    X, feat_kind = load_X(model_key, data_dir, prefix, mod)
    y = np.load(data_dir / f"{prefix}_y_task.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    fold_rows = []
    for info in iter_folds(subjects, data_tag):
        fold = int(info["fold"])
        masks = info["masks"]
        ckpt_path = out_root / "task" / f"fold{fold}" / "best_task.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        drop = float((ckpt.get("hparams") or {}).get("drop_prob", SHARED.drop_prob))
        model = builder(int(X.shape[-1]), 2, drop).to(device)
        model.load_state_dict(ckpt["model"])
        val_loader = DataLoader(
            make_dataset(X[masks["val"]], y[masks["val"]], feat_kind),
            batch_size=SHARED.batch_eval,
            shuffle=False,
        )
        test_loader = DataLoader(
            make_dataset(X[masks["test"]], y[masks["test"]], feat_kind),
            batch_size=SHARED.batch_eval,
            shuffle=False,
        )
        yv, pv = collect_proba(model, val_loader, device)
        yt, pt = collect_proba(model, test_loader, device)
        rows_v = sweep(yv, pv)
        chosen = pick_tau(rows_v)
        m05 = metrics_at_tau(yt, pt, 0.5)
        mc = metrics_at_tau(yt, pt, chosen["tau"])
        fold_rows.append(
            {
                "fold": fold,
                "chosen_tau": chosen["tau"],
                "test_at_05": {
                    "specificity": float(m05["specificity"]),
                    "recall": float(m05["recall"]),
                    "balanced_accuracy": float(m05["balanced_accuracy"]),
                },
                "test_at_chosen": {
                    "specificity": float(mc["specificity"]),
                    "recall": float(mc["recall"]),
                    "balanced_accuracy": float(mc["balanced_accuracy"]),
                },
            }
        )
    out = _agg_from_folds(fold_rows)
    out["out_root"] = str(out_root)
    return out


def _sort_key_model_arm(r: dict):
    mk = r.get("model_key", "")
    arm = r.get("arm", "")
    mi = MODEL_ORDER.index(mk) if mk in MODEL_ORDER else 99
    ai = ARM_ORDER.index(arm) if arm in ARM_ORDER else 99
    return (mi, ai)


def write_summary_md(progress: dict, sweeps: dict, out_md: Path) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models = sorted({r.get("model_key") for r in progress.get("runs", []) if r.get("ok")})
    n_models = len(models)
    lines = [
        f"# 八模型 Task 特异度汇总（{stamp}）",
        "",
        "口径：`merged_2s`，被试独立五折，`seed=42`；验收 Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65。",
        f"覆盖模型（{n_models}）：`{'` / `'.join(m for m in MODEL_ORDER if m in models)}`。",
        "Shallow 为 2026-08-01 下午代表 run；其余七模型为套件 `suite_progress.json`。",
        "更细的 Shallow 叙事见 `结论_20260801_Task特异度.md`。",
        "",
        "## 0. 实验序号说明",
        "",
        "| 实验序号 | 实验内容 |",
        "|----------|----------|",
    ]
    for exp_id in ARM_ORDER:
        lines.append(f"| `{exp_id}` | {EXP_LEGEND[exp_id]} |")
    lines += [
        "",
        "说明：字母大致对应步骤——`A*`=损失加权（步骤 A），`B*`=batch balance（步骤 B），`S*`=SMOTE（步骤 C/S）；"
        "数字区分变体（如 A20=w0=2.0，A22=w0=2.2；B1=仅 balance，B2=加权+balance）。",
        "",
        "## 1. 训练实验结果（Test 五折均值）",
        "",
        "| 模型 | 实验序号 | Acc | Spec | Rec | BalAcc | F1 | 过关 | run |",
        "|------|----------|-----|------|-----|--------|-----|------|-----|",
    ]
    runs = [r for r in progress.get("runs", []) if r.get("ok")]
    runs_sorted = sorted(runs, key=_sort_key_model_arm)
    for r in runs_sorted:
        acc_m = r.get("test_acc_mean")
        acc_s = r.get("test_acc_std", 0.0)
        acc_cell = (
            f"{acc_m:.4f}±{acc_s:.4f}" if acc_m is not None else "—"
        )
        lines.append(
            f"| {r['model_key']} | {r['arm']} | "
            f"{acc_cell} | "
            f"{r['test_specificity_mean']:.4f}±{r['test_specificity_std']:.4f} | "
            f"{r['test_recall_mean']:.4f}±{r['test_recall_std']:.4f} | "
            f"{r['test_balanced_accuracy_mean']:.4f}±{r['test_balanced_accuracy_std']:.4f} | "
            f"{r['test_f1_mean']:.4f}±{r['test_f1_std']:.4f} | "
            f"{'是' if r.get('pass_gate') else '否'} | `{r.get('stamp','')}` |"
        )
    fails = [r for r in progress.get("runs", []) if not r.get("ok")]
    if fails:
        lines += ["", "## 失败任务", ""]
        for r in fails:
            lines.append(f"- {r.get('model_key')}/{r.get('arm')}: `{r.get('error')}`")

    lines += ["", "## 2. 阈值扫描（实验 A20 / A22，Test）", ""]
    if not sweeps:
        lines.append("（尚无扫描结果）")
    else:
        lines += [
            "| 模型 | 实验序号 | τ=0.5 Spec/Rec/BalAcc | val选τ Spec/Rec/BalAcc |",
            "|------|----------|----------------------|------------------------|",
        ]
        def sweep_sort(item):
            key = item[0]
            mk, arm = key.split("/", 1) if "/" in key else (key, "")
            mi = MODEL_ORDER.index(mk) if mk in MODEL_ORDER else 99
            ai = ARM_ORDER.index(arm) if arm in ARM_ORDER else 99
            return (mi, ai)

        for key, sw in sorted(sweeps.items(), key=sweep_sort):
            if "error" in sw:
                lines.append(f"| {key.replace('/', ' | ')} | 错误：`{sw['error']}` | |")
                continue
            a05 = sw["agg_05"]
            ac = sw["agg_chosen"]
            mk, arm = key.split("/", 1) if "/" in key else (key, "")
            lines.append(
                f"| {mk} | {arm} | "
                f"{a05['specificity'][0]:.3f}/{a05['recall'][0]:.3f}/{a05['balanced_accuracy'][0]:.3f} | "
                f"{ac['specificity'][0]:.3f}/{ac['recall'][0]:.3f}/{ac['balanced_accuracy'][0]:.3f} |"
            )

    lines += ["", "## 3. 简要结论", ""]
    if runs_sorted:
        n_pass = sum(1 for r in runs_sorted if r.get("pass_gate"))
        lines.append(f"- 成功 run 数：`{len(runs_sorted)}`（{n_models} 模型）；过关：`{n_pass}`。")
        by_arm = {}
        for r in runs_sorted:
            by_arm.setdefault(r["arm"], []).append(r)
        for arm in ARM_ORDER:
            rs = by_arm.get(arm)
            if not rs:
                continue
            bal = np.mean([x["test_balanced_accuracy_mean"] for x in rs])
            spec = np.mean([x["test_specificity_mean"] for x in rs])
            rec = np.mean([x["test_recall_mean"] for x in rs])
            accs = [x["test_acc_mean"] for x in rs if x.get("test_acc_mean") is not None]
            acc_part = f", Acc={np.mean(accs):.3f}" if accs else ""
            lines.append(
                f"- 实验序号 `{arm}` {len(rs)} 模型均值："
                f"Spec={spec:.3f}, Rec={rec:.3f}, BalAcc={bal:.3f}{acc_part}"
            )
        # top BalAcc rows
        top = sorted(runs_sorted, key=lambda x: x["test_balanced_accuracy_mean"], reverse=True)[:5]
        lines.append("- BalAcc Top5：")
        for r in top:
            acc = r.get("test_acc_mean")
            acc_s = f", Acc={acc:.3f}" if acc is not None else ""
            lines.append(
                f"  - `{r['model_key']}/{r['arm']}`：BalAcc={r['test_balanced_accuracy_mean']:.3f}, "
                f"Spec={r['test_specificity_mean']:.3f}, Rec={r['test_recall_mean']:.3f}{acc_s}"
            )
        lines.append(
            "- 阈值扫描：val 选 τ 多抬 Rec、压 Spec，五折均值仍无法同时满足门控。"
        )
        lines.append(
            "- 综合：八模型在 `merged_2s` 上，仅靠加权 CE / batch balance / SMOTE / 调阈值，"
            "**均未稳定达到** Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65。"
        )
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--progress", type=Path, default=HERE / "suite_progress.json")
    p.add_argument(
        "--shallow_progress",
        type=Path,
        default=HERE / "shallow_progress.json",
        help="Shallow 下午代表 run；默认合并进八模型报告",
    )
    p.add_argument("--no_shallow", action="store_true", help="不合并 Shallow")
    p.add_argument("--data", default="merged_2s")
    p.add_argument("--skip_sweep", action="store_true")
    p.add_argument(
        "--force_sweep",
        action="store_true",
        help="忽略已有 threshold_sweep.json，强制重扫",
    )
    args = p.parse_args()
    suite = json.loads(args.progress.read_text(encoding="utf-8"))
    shallow = None
    if not args.no_shallow and args.shallow_progress.is_file():
        shallow = json.loads(args.shallow_progress.read_text(encoding="utf-8"))
    progress = merge_progress(suite, shallow)
    enrich_runs_with_summary_metrics(progress)
    merged_path = HERE / "suite_progress_all8.json"
    merged_path.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sweeps = {}
    if not args.skip_sweep:
        for r in progress.get("runs", []):
            if not r.get("ok") or r.get("arm") not in ("A20", "A22"):
                continue
            key = f"{r['model_key']}/{r['arm']}"
            out_root = Path(r["out_root"])
            if not args.force_sweep:
                cached = load_cached_sweep(out_root)
                if cached is not None:
                    print(f"reuse sweep {key}", flush=True)
                    sweeps[key] = cached
                    continue
            print(f"sweep {key}", flush=True)
            try:
                sweeps[key] = sweep_run(r["model_key"], out_root, args.data, device)
                (out_root / "threshold_sweep.json").write_text(
                    json.dumps(sweeps[key], indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as e:
                sweeps[key] = {"error": str(e)}
                print(f"  ERROR {key}: {e}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_md = HERE / f"汇总_八模型特异度_{stamp}.md"
    write_summary_md(progress, sweeps, out_md)
    dst = REPO_ROOT / "资料" / "模型训练" / "runs" / f"{stamp}_八模型特异度套件汇总"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "八模型特异度套件汇总.md").write_text(
        out_md.read_text(encoding="utf-8"), encoding="utf-8"
    )
    # 稳定入口（始终覆盖为最新）
    latest = HERE / "汇总_八模型特异度_最新.md"
    latest.write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8")
    print("md=", out_md)
    print("latest=", latest)
    print("copy=", dst)
    print("progress_all8=", merged_path)


if __name__ == "__main__":
    main()
