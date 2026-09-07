"""数据加载 / 折划分 / Acc_paper 评测共用。"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
STEP_DIR = HERE.parent
CODE_ROOT = HERE.parents[3]
TRAIN_LAB = CODE_ROOT / "train_lab"
REPO_ROOT = CODE_ROOT.parent
PRE_ROOT = CODE_ROOT / "preprocess_lab"
ACC = STEP_DIR / "5060_baselines_openbmi_2s_hop100_accpaper"

# 本包必须优先，否则会命中其它 step/*/shared_hparams.py
_here = str(HERE)
if _here in sys.path:
    sys.path.remove(_here)
sys.path.insert(0, _here)
for p in (STEP_DIR, PRE_ROOT, ACC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)

from shared_hparams import SHARED, SharedHP, OUT_ROOT_TAG  # noqa: E402
from data_paths import resolve_data  # noqa: E402
from metrics import jsonify_metrics, three_class_metrics  # noqa: E402
from src.common.steps.split_subjects import iter_subject_kfold  # noqa: E402

# trial_metrics 在 accpaper 包内；临时把 ACC 置前再 import
_acc = str(ACC)
if _acc in sys.path:
    sys.path.remove(_acc)
sys.path.insert(0, _acc)
from trial_metrics import aggregate_windows_to_trials  # noqa: E402
sys.path.remove(_acc)
sys.path.append(_acc)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WindowDataset(Dataset):
    def __init__(self, X, indices, y=None, x_path: str | None = None):
        self.x_path = x_path
        self._X = None if x_path else X
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        self.y = None if y is None else np.asarray(y, dtype=np.int64)

    def _x(self):
        if self._X is None:
            self._X = np.load(self.x_path, mmap_mode="r")
        return self._X

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        gi = int(self.indices[i])
        x = np.asarray(self._x()[gi], dtype=np.float32)
        if x.ndim == 3:
            x = x.squeeze(0)  # (1,8,T) → (8,T)
        xt = torch.from_numpy(x.copy())
        if self.y is None:
            return xt
        return xt, int(self.y[gi])


def load_openbmi(hp: SharedHP = SHARED):
    data_dir, prefix = resolve_data(hp.data_tag)
    x_npy = data_dir / f"{prefix}_X.npy"
    X = np.load(x_npy, mmap_mode="r")
    y_three = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    trial_ids = np.load(data_dir / f"{prefix}_trial_id.npy")
    assert int(X.shape[-1]) == hp.n_times_expected
    return {
        "X": X,
        "y_three": y_three,
        "subjects": subjects,
        "trial_ids": trial_ids,
        "x_path": str(x_npy),
        "data_dir": data_dir,
    }


def subject_masks(subjects, train_subj, val_subj, test_subj):
    s = np.asarray([str(x) for x in subjects])
    tr = set(map(str, train_subj))
    va = set(map(str, val_subj))
    te = set(map(str, test_subj))
    return {
        "train": np.isin(s, list(tr)),
        "val": np.isin(s, list(va)),
        "test": np.isin(s, list(te)),
    }


@torch.no_grad()
def eval_acc_paper(model, X, y, subjects, trial_ids, mask, device, hp, x_path):
    idx = np.flatnonzero(mask)
    ds = WindowDataset(X, idx, y=y, x_path=x_path)
    loader = DataLoader(
        ds,
        batch_size=hp.batch_eval,
        shuffle=False,
        num_workers=0,
        pin_memory=hp.pin_memory and device.type == "cuda",
    )
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    yt = np.concatenate(ys)
    yp = np.concatenate(ps)
    subs = subjects[idx]
    tids = trial_ids[idx]
    trial = aggregate_windows_to_trials(yt, yp, subs, tids, n_classes=3)
    win = jsonify_metrics(three_class_metrics(yt, yp))
    return trial["metrics"], win


def make_loader(ds, batch, shuffle, hp, sampler=None):
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(hp.num_workers),
        pin_memory=bool(hp.pin_memory) and torch.cuda.is_available(),
        drop_last=False,
    )


def out_root(stamp: str | None = None) -> Path:
    from datetime import datetime

    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = TRAIN_LAB / "out" / OUT_ROOT_TAG / f"run_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def build_backbone(hp: SharedHP) -> nn.Module:
    from jepa_model import EEGJepa

    return EEGJepa(
        n_times=hp.n_times_expected,
        embed_dim=hp.embed_dim,
        n_heads=hp.n_heads,
        n_layers=hp.n_layers,
        patch_time=hp.patch_time,
        drop_prob=0.1,
        mask_ratio=hp.mask_ratio,
        mask_n_blocks=hp.mask_n_blocks,
        ema_momentum=hp.ema_momentum,
    )
