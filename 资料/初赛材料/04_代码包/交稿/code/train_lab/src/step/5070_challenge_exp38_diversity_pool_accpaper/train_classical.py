# -*- coding: utf-8 -*-
"""Exp38 D1：经典候选 fbcsp_lda / riemann_tsc（sklearn · 无 pyriemann）。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from mne.decoding import CSP
from scipy.linalg import eigh, logm
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_STEP = Path(__file__).resolve().parent
from exp38_config import OUT_ROOT_TAG, RUN_TAG, a59_step, exp38_out  # noqa: E402

_A59 = a59_step()
for p in (_A59, _STEP.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_paths import resolve_data  # noqa: E402
from loso import iter_loso6  # noqa: E402
from metrics_three import three_class_report  # noqa: E402


def _as_ct(X: np.ndarray) -> np.ndarray:
    """(N,1,C,T) or (N,C,T) → (N,C,T)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0]
    if X.ndim != 3:
        raise ValueError(X.shape)
    return X


def _bandpass_fft(x_ct: np.ndarray, fs: float, l_freq: float, h_freq: float) -> np.ndarray:
    """x (C,T) 简易 FFT 带通。"""
    c, t = x_ct.shape
    freqs = np.fft.rfftfreq(t, d=1.0 / fs)
    spec = np.fft.rfft(x_ct, axis=1)
    mask = (freqs >= l_freq) & (freqs <= h_freq)
    spec[:, ~mask] = 0
    return np.fft.irfft(spec, n=t, axis=1).astype(np.float64)


def _filter_bank(X: np.ndarray, fs: float = 250.0) -> list[np.ndarray]:
    bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
    out = []
    for lo, hi in bands:
        xb = np.stack([_bandpass_fft(X[i], fs, lo, hi) for i in range(len(X))], axis=0)
        out.append(xb)
    return out


class FilterBankCSPLDA:
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = _as_ct(X)
        y = np.asarray(y, dtype=np.int64)
        banks = _filter_bank(X)
        self.csps_ = []
        feats = []
        for xb in banks:
            csp = CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)
            # CSP expects (n_epochs, n_channels, n_times)
            f = csp.fit_transform(xb, y)
            self.csps_.append(csp)
            feats.append(f)
        F = np.concatenate(feats, axis=1)
        self.scaler_ = StandardScaler()
        self.clf_ = LinearDiscriminantAnalysis()
        Z = self.scaler_.fit_transform(F)
        self.clf_.fit(Z, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _as_ct(X)
        feats = [csp.transform(xb) for csp, xb in zip(self.csps_, _filter_bank(X))]
        F = np.concatenate(feats, axis=1)
        Z = self.scaler_.transform(F)
        return self.clf_.predict_proba(Z).astype(np.float32)


def _cov_vec(x_ct: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf cov → logm → 上三角向量。"""
    lw = LedoitWolf().fit(x_ct.T)
    c = lw.covariance_
    # 对称化 + 本征钳制保证 PD
    c = 0.5 * (c + c.T)
    w, v = eigh(c)
    w = np.clip(w, 1e-6, None)
    c = (v * w) @ v.T
    lm = logm(c).real
    iu = np.triu_indices(lm.shape[0])
    return lm[iu].astype(np.float64)


class RiemannTSC:
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = _as_ct(X)
        y = np.asarray(y, dtype=np.int64)
        F = np.stack([_cov_vec(X[i]) for i in range(len(X))], axis=0)
        self.scaler_ = StandardScaler()
        self.clf_ = LinearDiscriminantAnalysis()
        Z = self.scaler_.fit_transform(F)
        self.clf_.fit(Z, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _as_ct(X)
        F = np.stack([_cov_vec(X[i]) for i in range(len(X))], axis=0)
        Z = self.scaler_.transform(F)
        return self.clf_.predict_proba(Z).astype(np.float32)


MODELS = {
    "fbcsp_lda": FilterBankCSPLDA,
    "riemann_tsc": RiemannTSC,
}


def train_one(model_name: str, *, run_tag: str, max_folds: int = 0) -> Path:
    data_dir, prefix = resolve_data("challenge_mi_3s_59ch")
    X = np.load(data_dir / f"{prefix}_X.npy", mmap_mode="r")
    y = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    run_dir = (
        exp38_out()
        / f"{model_name}_challenge_mi_3s_59ch"
        / "challenge_mi_3s_59ch"
        / f"run_{run_tag}"
        / "three"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_name": model_name,
        "experiment": 38,
        "stage": "D1",
        "run_tag": run_tag,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    fold_results = []
    for info in iter_loso6(subjects):
        fold = int(info["fold"])
        if max_folds > 0 and fold >= max_folds:
            break
        tr = np.where(info["masks"]["train"])[0]
        va = np.where(info["masks"]["val"])[0]
        fold_dir = run_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"==== {model_name} fold{fold} n={len(tr)}/{len(va)} ====", flush=True)
        clf = MODELS[model_name]()
        Xtr = np.asarray(X[tr])
        Xva = np.asarray(X[va])
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xva)
        pred = prob.argmax(1)
        report = three_class_report(y[va], pred)
        np.save(fold_dir / "val_prob.npy", prob)
        np.save(fold_dir / "val_y.npy", y[va].astype(np.int64))
        np.save(fold_dir / "val_idx.npy", va.astype(np.int64))
        rec = {"fold": fold, "best_val_acc": float(report["acc"]), "val_metrics": report}
        (fold_dir / "summary.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fold_results.append(rec)
        print(f"  acc={report['acc']:.4f}", flush=True)

    accs = [float(r["best_val_acc"]) for r in fold_results]
    summary = {
        "model_name": model_name,
        "n_folds": len(accs),
        "val_acc_mean": float(np.mean(accs)) if accs else None,
        "val_acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "folds": fold_results,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{model_name} mean={summary['val_acc_mean']:.4f}±{summary['val_acc_std']:.4f}")
    return run_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="fbcsp_lda,riemann_tsc")
    ap.add_argument("--run-tag", default=RUN_TAG)
    ap.add_argument("--max-folds", type=int, default=0)
    args = ap.parse_args()
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        if name not in MODELS:
            raise SystemExit(f"unknown {name}")
        train_one(name, run_tag=args.run_tag, max_folds=args.max_folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
