"""方案 26 · E2 异构成员：向量特征五折 + prob dump。"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
PKG24 = STEP / "5090_baselines_openbmi_3s_hop100_accpaper"
PRE = STEP.parents[2] / "preprocess_lab"
for p in (str(STEP), str(PKG24), str(PRE), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_paths import resolve_data  # noqa: E402
from perf_loader import apply_runtime_threads, configure_cuda_backends, make_loader  # noqa: E402
from s26_hparams import OUT_ROOT_TAG, RecipeTrainHP  # noqa: E402
from s26_config import FE_CACHE  # noqa: E402
from trial_metrics import aggregate_windows_to_trials  # noqa: E402
from t0_sec import compute_window_t0_sec  # noqa: E402
from src.common.steps.split_subjects import iter_subject_kfold  # noqa: E402

from recipe_train import (  # noqa: E402
    is_improved,
    make_criterion,
    make_optimizer,
    make_scheduler,
    run_epoch_recipe,
)
from task_runner import (  # noqa: E402
    _indices_from_mask,
    _loader_kwargs,
    make_balanced_sampler,
    seed_everything,
)
from prob_dump import dump_rows_to_csv  # noqa: E402


class VectorFeatDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        return (
            torch.from_numpy(np.array(self.X[idx], dtype=np.float32)),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def load_openbmi(*, data_tag: str = "openbmi_3s_hop100"):
    data_dir, prefix = resolve_data(data_tag)
    x_path = data_dir / f"{prefix}_X.npy"
    X = np.load(x_path, mmap_mode="r")
    y_three = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    trial_ids = np.load(data_dir / f"{prefix}_trial_id.npy")
    return X, y_three, subjects, trial_ids, str(x_path)


def ensure_feature_cache(
    X_src: np.ndarray,
    x_path: str,
    *,
    cache_name: str,
    materialize_fn,
) -> np.ndarray:
    tag = Path(x_path).stem
    out = FE_CACHE / f"{cache_name}_{tag}.npy"
    materialize_fn(X_src, out)
    return np.load(out, mmap_mode="r")


def _eval_vector_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    subjects,
    trial_ids,
    mask: np.ndarray,
    device,
    hp: RecipeTrainHP,
    *,
    n_classes: int = 3,
) -> dict:
    idx = _indices_from_mask(mask)
    loader = make_loader(
        VectorFeatDataset(X, y, idx),
        batch_size=hp.batch_eval,
        shuffle=False,
        **_loader_kwargs(hp),
    )
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=hp.non_blocking)
            logits = model(xb)
            if logits.ndim > 2:
                logits = logits.reshape(logits.shape[0], -1)
            pred = logits.argmax(dim=1).cpu().numpy()
            yt.extend(yb.numpy().tolist())
            yp.extend(pred.tolist())
    yt_arr = np.asarray(yt, dtype=np.int64)
    yp_arr = np.asarray(yp, dtype=np.int64)
    subs = subjects[idx]
    tids = trial_ids[idx]
    trial = aggregate_windows_to_trials(yt_arr, yp_arr, subs, tids, n_classes=n_classes)
    return trial["metrics"]


@torch.no_grad()
def _collect_kan_prob_rows(
    model,
    X: np.ndarray,
    y: np.ndarray,
    subjects,
    trial_ids,
    mask: np.ndarray,
    device,
    hp: RecipeTrainHP,
    *,
    fold: int,
    split: str,
) -> list[dict]:
    idx = _indices_from_mask(mask)
    loader = make_loader(
        VectorFeatDataset(X, y, idx),
        batch_size=hp.batch_eval,
        shuffle=False,
        **_loader_kwargs(hp),
    )
    t0_all = compute_window_t0_sec(trial_ids)
    rows: list[dict] = []
    off = 0
    model.eval()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=hp.non_blocking)
        logits = model(xb)
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        bs = probs.shape[0]
        for i in range(bs):
            gi = int(idx[off + i])
            p = probs[i]
            rows.append(
                {
                    "subject": str(subjects[gi]),
                    "fold": fold,
                    "split": split,
                    "trial_id": int(trial_ids[gi]),
                    "t0_sec": float(t0_all[gi]),
                    "pred": int(p.argmax()),
                    "y": int(y[gi]),
                    "p_max": float(p.max()),
                    "p0": float(p[0]),
                    "p1": float(p[1]),
                    "p2": float(p[2]),
                }
            )
        off += bs
    return rows


def _collect_sklearn_prob_rows(
    clf,
    scaler: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,
    subjects,
    trial_ids,
    mask: np.ndarray,
    *,
    fold: int,
    split: str,
) -> list[dict]:
    idx = _indices_from_mask(mask)
    Xv = scaler.transform(np.array(X[idx], dtype=np.float64))
    probs = clf.predict_proba(Xv)
    t0_all = compute_window_t0_sec(trial_ids)
    rows: list[dict] = []
    for j, gi in enumerate(idx):
        gi = int(gi)
        p = probs[j]
        rows.append(
            {
                "subject": str(subjects[gi]),
                "fold": fold,
                "split": split,
                "trial_id": int(trial_ids[gi]),
                "t0_sec": float(t0_all[gi]),
                "pred": int(p.argmax()),
                "y": int(y[gi]),
                "p_max": float(p.max()),
                "p0": float(p[0]),
                "p1": float(p[1]),
                "p2": float(p[2]),
            }
        )
    return rows


def run_kan_e2a(
    *,
    model_name: str = "kan_bandpower_e2a",
    build_model,
    hp: RecipeTrainHP | None = None,
    max_folds: int = 0,
    max_epochs: int = 0,
) -> Path:
    hp = hp or RecipeTrainHP()
    if max_epochs > 0:
        from dataclasses import replace

        hp = replace(hp, max_epochs=max_epochs)
    apply_runtime_threads(hp.torch_num_threads)
    configure_cuda_backends(cudnn_benchmark=hp.cudnn_benchmark, deterministic=hp.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from fe_bandpower_3s import materialize_bandpower_cache

    X_src, y, subjects, trial_ids, x_path = load_openbmi(data_tag=hp.data_tag)
    X = ensure_feature_cache(
        X_src, x_path, cache_name="bandpower24_3s", materialize_fn=materialize_bandpower_cache
    )
    in_dim = int(X.shape[1])

    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_root = (
        Path(__file__).resolve().parents[3]
        / "out"
        / OUT_ROOT_TAG
        / model_name
        / hp.data_tag
        / stamp
        / "three"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    fold_results = []
    n_folds = hp.n_folds if max_folds <= 0 else min(max_folds, hp.n_folds)

    for fold_info in iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    ):
        fold = fold_info["fold"]
        if fold >= n_folds:
            break
        masks = fold_info["masks"]
        fold_dir = out_root / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        tr_idx = _indices_from_mask(masks["train"])
        va_idx = _indices_from_mask(masks["val"])
        seed_everything(
            hp.seed + fold,
            cudnn_benchmark=hp.cudnn_benchmark,
            deterministic=hp.deterministic,
        )
        g = torch.Generator()
        g.manual_seed(hp.seed + fold)
        train_loader = make_loader(
            VectorFeatDataset(X, y, tr_idx),
            batch_size=hp.batch_train,
            sampler=make_balanced_sampler(y[tr_idx], n_classes=3, generator=g),
            **_loader_kwargs(hp),
        )
        model = build_model(in_dim, 3, hp.drop_prob).to(device)
        criterion = make_criterion(hp)
        optimizer = make_optimizer(model, hp)
        scheduler = make_scheduler(optimizer, hp, total_epochs=hp.max_epochs)
        scaler = torch.cuda.amp.GradScaler() if hp.use_amp and device.type == "cuda" else None
        best_score, best_state, bad = -1.0, None, 0
        for ep in range(1, hp.max_epochs + 1):
            run_epoch_recipe(
                model, train_loader, criterion, optimizer, device, True, hp,
                non_blocking=hp.non_blocking, use_amp=hp.use_amp, scaler=scaler,
            )
            scheduler.step()
            val_m = _eval_vector_model(
                model, X, y, subjects, trial_ids, masks["val"], device, hp
            )
            score = float(val_m["acc_paper"])
            print(f"fold{fold} ep{ep:03d} val_AccPaper={score:.4f}")
            if is_improved(score, best_score, hp):
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
                torch.save({"model": best_state, "hparams": asdict(hp)}, fold_dir / "best_three.pt")
            else:
                bad += 1
                if bad >= hp.patience:
                    break
        assert best_state is not None
        model.load_state_dict(best_state)
        te_m = _eval_vector_model(
            model, X, y, subjects, trial_ids, masks["test"], device, hp
        )
        rows = _collect_kan_prob_rows(
            model, X, y, subjects, trial_ids, masks["val"], device, hp,
            fold=fold, split="val",
        ) + _collect_kan_prob_rows(
            model, X, y, subjects, trial_ids, masks["test"], device, hp,
            fold=fold, split="test",
        )
        dump_rows_to_csv(fold_dir / "prob_dump_three.csv", rows)
        fold_results.append({"fold": fold, "val_acc_paper": best_score, "test_acc_paper": te_m["acc_paper"]})

    vals = [r["val_acc_paper"] for r in fold_results]
    tests = [r["test_acc_paper"] for r in fold_results]
    summary = {
        "model_name": model_name,
        "arm": "E2a",
        "val_acc_paper_mean": float(np.mean(vals)),
        "val_acc_paper_std": float(np.std(vals)),
        "test_acc_paper_mean": float(np.mean(tests)),
        "test_acc_paper_std": float(np.std(tests)),
        "folds": fold_results,
        "hparams": asdict(hp),
    }
    out_root.parent.joinpath("meta.json").write_text(
        json.dumps({"stamp": stamp, "model": model_name, "arm": "E2a"}, indent=2),
        encoding="utf-8",
    )
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[E2a] test Acc_paper {summary['test_acc_paper_mean']:.4f}±{summary['test_acc_paper_std']:.4f}")
    return out_root


def run_riemann_e2b(
    *,
    model_name: str = "riemann_tangent_e2b",
    band_split: bool = False,
    max_folds: int = 0,
) -> Path:
    from fe_riemannian import materialize_riemann_cache

    hp = RecipeTrainHP()
    X_src, y, subjects, trial_ids, x_path = load_openbmi(data_tag=hp.data_tag)
    cache_name = "riemann72_3s" if band_split else "riemann36_3s"
    X = ensure_feature_cache(
        X_src,
        x_path,
        cache_name=cache_name,
        materialize_fn=lambda src, out: materialize_riemann_cache(src, out, band_split=band_split),
    )
    c_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_root = (
        Path(__file__).resolve().parents[3]
        / "out"
        / OUT_ROOT_TAG
        / model_name
        / hp.data_tag
        / stamp
        / "three"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    fold_results = []
    n_folds = hp.n_folds if max_folds <= 0 else min(max_folds, hp.n_folds)

    for fold_info in iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    ):
        fold = fold_info["fold"]
        if fold >= n_folds:
            break
        masks = fold_info["masks"]
        fold_dir = out_root / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        tr_idx = _indices_from_mask(masks["train"])
        va_idx = _indices_from_mask(masks["val"])
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(np.array(X[tr_idx], dtype=np.float64))
        X_va = scaler.transform(np.array(X[va_idx], dtype=np.float64))
        best_c, best_acc, best_clf = c_grid[0], -1.0, None
        for c in c_grid:
            clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
            clf.fit(X_tr, y[tr_idx])
            pred = clf.predict(X_va)
            trial = aggregate_windows_to_trials(
                y[va_idx], pred, subjects[va_idx], trial_ids[va_idx], n_classes=3
            )
            acc = float(trial["metrics"]["acc_paper"])
            if acc > best_acc:
                best_acc, best_c, best_clf = acc, c, clf
        assert best_clf is not None
        te_idx = _indices_from_mask(masks["test"])
        X_te = scaler.transform(np.array(X[te_idx], dtype=np.float64))
        te_pred = best_clf.predict(X_te)
        te_trial = aggregate_windows_to_trials(
            y[te_idx], te_pred, subjects[te_idx], trial_ids[te_idx], n_classes=3
        )
        rows = _collect_sklearn_prob_rows(
            best_clf, scaler, X, y, subjects, trial_ids, masks["val"],
            fold=fold, split="val",
        ) + _collect_sklearn_prob_rows(
            best_clf, scaler, X, y, subjects, trial_ids, masks["test"],
            fold=fold, split="test",
        )
        dump_rows_to_csv(fold_dir / "prob_dump_three.csv", rows)
        fold_results.append(
            {
                "fold": fold,
                "best_C": best_c,
                "val_acc_paper": best_acc,
                "test_acc_paper": float(te_trial["metrics"]["acc_paper"]),
            }
        )
        print(f"fold{fold} C={best_c} val={best_acc:.4f} test={te_trial['metrics']['acc_paper']:.4f}")

    vals = [r["val_acc_paper"] for r in fold_results]
    tests = [r["test_acc_paper"] for r in fold_results]
    summary = {
        "model_name": model_name,
        "arm": "E2b",
        "band_split": band_split,
        "val_acc_paper_mean": float(np.mean(vals)),
        "test_acc_paper_mean": float(np.mean(tests)),
        "folds": fold_results,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_root
