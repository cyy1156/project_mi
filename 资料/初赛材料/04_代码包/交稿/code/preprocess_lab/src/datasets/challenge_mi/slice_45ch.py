"""从已有 challenge_mi_3s_59ch 切片得到 challenge_mi_3s_45ch（零重滤波）。

用法（preprocess_lab 根）：
  python -m src.datasets.challenge_mi.slice_45ch
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.datasets.challenge_mi.channels_intersect import (
    INTERSECT_45,
    PROTOCOL_45,
    indices_59_to_45,
)

_PREPROCESS_ROOT = Path(__file__).resolve().parents[3]


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slice_arrays(src_dir: Path, dst_dir: Path) -> dict:
    idx = indices_59_to_45()
    dst_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "protocol": PROTOCOL_45,
        "source": str(src_dir),
        "n_chans": 45,
        "channels": list(INTERSECT_45),
        "indices_from_59": idx,
        "time": _utc(),
        "note": "slice of challenge_mi_3s_59ch; same filter/zscore; OpenBMI∩official=45",
    }
    copied_labels = (
        "challenge_y_task.npy",
        "challenge_y_three.npy",
        "challenge_subjects.npy",
        "challenge_trial_id.npy",
        "challenge_test_y_task.npy",
        "challenge_test_y_three.npy",
        "challenge_test_subjects.npy",
        "challenge_test_trial_id.npy",
    )
    for name in copied_labels:
        sp = src_dir / name
        if sp.is_file():
            shutil.copy2(sp, dst_dir / name)

    for split_prefix in ("challenge", "challenge_test"):
        xp = src_dir / f"{split_prefix}_X.npy"
        if not xp.is_file():
            continue
        X = np.load(xp, mmap_mode="r")
        if X.ndim != 4 or X.shape[2] != 59:
            raise ValueError(f"{xp} unexpected shape {X.shape}")
        X45 = np.asarray(X[:, :, idx, :], dtype=np.float32)
        out_p = dst_dir / f"{split_prefix}_X.npy"
        np.save(out_p, X45)
        meta[f"{split_prefix}_shape"] = list(X45.shape)
        print(f"wrote {out_p} {X45.shape}", flush=True)

    (dst_dir / "preprocess_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=Path,
        default=_PREPROCESS_ROOT / "out" / "challenge_mi_3s_59ch",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=_PREPROCESS_ROOT / "out" / PROTOCOL_45,
    )
    args = ap.parse_args()
    meta = slice_arrays(args.src, args.dst)
    print("ok", json.dumps(meta, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
