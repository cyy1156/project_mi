"""BCI IV-2a P-track loader: 4-class, 8ch, T train / E eval, raw (no bandpass)."""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import scipy.io

EEG22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
TARGET_CHANNELS = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]
TARGET_IDX = [EEG22.index(c) for c in TARGET_CHANNELS]

REPO = Path(__file__).resolve().parents[5]  # D:/cyy/MI
DATA_DIR = REPO / "DATA" / "bci2a"
TRUE_LABELS = DATA_DIR / "true_labels"
DEFAULT_CACHE = REPO / "code" / "preprocess_lab" / "out" / "bci2a_p_8ch_cue05to5"


def _select8(x_nt_c: np.ndarray) -> np.ndarray:
    """x: (n_times, >=22) → (n_times, 8)."""
    return np.asarray(x_nt_c[:, TARGET_IDX], dtype=np.float64)


def _epoch_window(x: np.ndarray, cue: int, fs: float, t0: float, t1: float) -> np.ndarray | None:
    """Return (C, T) float64 or None if OOB. No baseline correction (paper: Z-score only)."""
    i0 = cue + int(round(t0 * fs))
    i1 = cue + int(round(t1 * fs))
    if i0 < 0 or i1 > x.shape[0] or i1 <= i0:
        return None
    win = x[i0:i1]  # (T, C)
    return win.T.copy()


def load_train_mat(
    subject: str,
    *,
    fs: float = 250.0,
    t0: float = 0.5,
    t1: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load A0xT.mat → X (N,C,T), y (N,) labels 0..3."""
    mat_path = DATA_DIR / f"{subject}T.mat"
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for run in mat["data"]:
        trial = np.atleast_1d(run.trial) if run.trial is not None else np.array([])
        if trial.size == 0:
            continue
        x = _select8(np.asarray(run.X, dtype=np.float64)[:, :22])
        y = np.atleast_1d(run.y).astype(int)
        for cue, lab in zip(trial.astype(int), y, strict=True):
            ep = _epoch_window(x, int(cue), fs, t0, t1)
            if ep is None:
                continue
            xs.append(ep)
            ys.append(int(lab) - 1)  # 1..4 → 0..3
    X = np.stack(xs, axis=0).astype(np.float32)
    y_arr = np.asarray(ys, dtype=np.int64)
    return X, y_arr


def load_eval_gdf(
    subject: str,
    *,
    fs: float = 250.0,
    t0: float = 0.5,
    t1: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load A0xE.gdf + true_labels → X (N,C,T), y (N,) 0..3."""
    gdf_path = DATA_DIR / f"{subject}E.gdf"
    lab_path = TRUE_LABELS / f"{subject}E.mat"
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    data = raw.get_data()  # (n_ch, n_times); MNE GDF is in Volts
    # Competition MAT files are in µV — scale E session to µV so train μ/σ apply.
    data = data * 1e6
    # mne renames poorly; first 22 channels keep official EEG22 order
    eeg = data[:22].T  # (n_times, 22)
    x8 = _select8(eeg)
    events, eid = mne.events_from_annotations(raw, verbose=False)
    code768 = eid["768"]
    cues = events[events[:, 2] == code768][:, 0].astype(int)
    y_raw = np.asarray(
        scipy.io.loadmat(lab_path, squeeze_me=True)["classlabel"], dtype=int
    )
    if len(cues) != len(y_raw):
        raise RuntimeError(f"{subject}E: cues={len(cues)} labels={len(y_raw)}")
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for cue, lab in zip(cues, y_raw, strict=True):
        ep = _epoch_window(x8, int(cue), fs, t0, t1)
        if ep is None:
            continue
        xs.append(ep)
        ys.append(int(lab) - 1)
    X = np.stack(xs, axis=0).astype(np.float32)
    y_arr = np.asarray(ys, dtype=np.int64)
    return X, y_arr


def zscore_from_train(X_tr: np.ndarray, X_ev: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Z-score with train-set μ,σ over all train samples/chans/times (global) — common MI practice.
    Paper Eq.1 is per-trial; we apply per-channel using train statistics across trials×time
    (more stable). Also expose per-trial option via note in meta.
    """
    # Per-channel μ,σ from training set: mean over (N,T)
    mu = X_tr.mean(axis=(0, 2), keepdims=True)  # (1,C,1)
    sigma = X_tr.std(axis=(0, 2), keepdims=True)
    sigma = np.maximum(sigma, 1e-6)
    X_tr_z = ((X_tr - mu) / sigma).astype(np.float32)
    X_ev_z = ((X_ev - mu) / sigma).astype(np.float32)
    stats = {"mu": mu.squeeze().tolist(), "sigma": sigma.squeeze().tolist()}
    return X_tr_z, X_ev_z, stats


def build_or_load_cache(
    cache_dir: Path | None = None,
    *,
    subjects: list[str] | None = None,
    force: bool = False,
    t0: float = 0.5,
    t1: float = 5.0,
) -> Path:
    cache_dir = Path(cache_dir or DEFAULT_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    subjects = subjects or [f"A0{i}" for i in range(1, 10)]
    meta = {
        "channels": TARGET_CHANNELS,
        "channel_idx_in_eeg22": TARGET_IDX,
        "win_sec": [t0, t1],
        "fs": 250.0,
        "n_classes": 4,
        "note": "P-track: full-band, no artifact reject, no bandpass; Z-score applied in runner from train",
        "subjects": [],
    }
    for sub in subjects:
        out = cache_dir / sub
        marker = out / "done.json"
        if marker.exists() and not force:
            meta["subjects"].append(sub)
            continue
        out.mkdir(parents=True, exist_ok=True)
        X_tr, y_tr = load_train_mat(sub, t0=t0, t1=t1)
        X_ev, y_ev = load_eval_gdf(sub, t0=t0, t1=t1)
        np.save(out / "X_train.npy", X_tr)
        np.save(out / "y_train.npy", y_tr)
        np.save(out / "X_eval.npy", X_ev)
        np.save(out / "y_eval.npy", y_ev)
        info = {
            "subject": sub,
            "X_train": list(X_tr.shape),
            "X_eval": list(X_ev.shape),
            "y_train_counts": {int(k): int(v) for k, v in zip(*np.unique(y_tr, return_counts=True))},
            "y_eval_counts": {int(k): int(v) for k, v in zip(*np.unique(y_ev, return_counts=True))},
        }
        marker.write_text(json.dumps(info, indent=2), encoding="utf-8")
        meta["subjects"].append(sub)
        print(f"[cache] {sub} train={X_tr.shape} eval={X_ev.shape}")
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return cache_dir


def load_subject(cache_dir: Path, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = Path(cache_dir) / subject
    return (
        np.load(d / "X_train.npy"),
        np.load(d / "y_train.npy"),
        np.load(d / "X_eval.npy"),
        np.load(d / "y_eval.npy"),
    )


if __name__ == "__main__":
    build_or_load_cache()
