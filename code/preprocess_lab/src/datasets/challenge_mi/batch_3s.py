"""挑战杯官方集批处理：按 block 写 shard，再合并 train / test npy。

用法（在 preprocess_lab 根目录）：
  python -m src.datasets.challenge_mi.batch_3s --mode 59 --limit-blocks 1
  python -m src.datasets.challenge_mi.batch_3s --mode 59
  python -m src.datasets.challenge_mi.batch_3s --mode 8
  python -m src.datasets.challenge_mi.batch_3s --mode 59 --merge-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.datasets.challenge_mi import PROTOCOL_8, PROTOCOL_59
from src.datasets.challenge_mi.load_pkl import (
    list_test_blocks,
    list_train_blocks,
    resolve_data_root,
    subject_from_path,
)
from src.datasets.challenge_mi.pipeline import preprocess_block_path

_PREPROCESS_ROOT = Path(__file__).resolve().parents[3]
FULL_KEYS = ("X", "y_task", "y_three", "subjects", "trial_id")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mode_cfg(mode: str) -> tuple[str, str, str]:
    m = mode.strip().lower()
    if m in ("59", "a59"):
        return "59", PROTOCOL_59, "challenge"
    if m in ("8", "b8"):
        return "8", PROTOCOL_8, "challenge"
    raise ValueError(f"mode 应为 59 或 8，收到 {mode!r}")


def load_manifest(path: Path, protocol: str) -> dict:
    if not path.exists():
        return {"version": 1, "protocol": protocol, "files": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def shard_root(out_dir: Path) -> Path:
    d = out_dir / "shards"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fid_of(path: Path, split: str) -> str:
    # train_S01_block_1
    return f"{split}_{path.parent.name}_{path.stem}"


def save_shard(out_dir: Path, fid: str, arrays: dict[str, np.ndarray]) -> None:
    root = shard_root(out_dir) / fid
    root.mkdir(parents=True, exist_ok=True)
    for k, arr in arrays.items():
        np.save(root / f"{k}.npy", arr)


def process_one(
    path: Path,
    *,
    split: str,
    channel_mode: str,
    out_dir: Path,
    trial_counter: list[int],
) -> dict:
    fid = fid_of(path, split)
    subj = f"challenge:{subject_from_path(path)}"
    try:
        out = preprocess_block_path(path, channel_mode=channel_mode)
        n = int(out["X"].shape[0])
        subjects = np.asarray([subj] * n, dtype=object)
        tids = np.arange(trial_counter[0], trial_counter[0] + n, dtype=np.int64)
        trial_counter[0] += n
        arrays: dict[str, np.ndarray] = {
            "X": out["X"],
            "subjects": subjects,
            "trial_id": tids,
        }
        if out["y_three"] is not None:
            arrays["y_three"] = out["y_three"]
            arrays["y_task"] = out["y_task"]
        else:
            arrays["y_three"] = np.full(n, -1, dtype=np.int64)
            arrays["y_task"] = np.full(n, -1, dtype=np.int64)
        save_shard(out_dir, fid, arrays)
        return {
            "status": "ok",
            "n": n,
            "subject": subj,
            "split": split,
            "shape": list(out["X"].shape),
            "channels": out["channel_names"],
            "time": _utc_now(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
            "time": _utc_now(),
        }


def merge_split(
    out_dir: Path,
    *,
    split: str,
    prefix: str,
    protocol: str,
) -> dict:
    man = load_manifest(out_dir / "manifest.json", protocol)
    fids = sorted(
        fid
        for fid, info in man.get("files", {}).items()
        if info.get("status") == "ok" and str(fid).startswith(f"{split}_")
    )
    if not fids:
        raise RuntimeError(f"merge {split}: 无成功 shard")

    Xs, ytasks, ythrees, subs, tids = [], [], [], [], []
    for fid in fids:
        root = shard_root(out_dir) / fid
        Xs.append(np.load(root / "X.npy"))
        ytasks.append(np.load(root / "y_task.npy"))
        ythrees.append(np.load(root / "y_three.npy"))
        subs.append(np.load(root / "subjects.npy", allow_pickle=True))
        tids.append(np.load(root / "trial_id.npy"))

    X = np.concatenate(Xs, axis=0)
    y_task = np.concatenate(ytasks, axis=0)
    y_three = np.concatenate(ythrees, axis=0)
    subjects = np.concatenate(subs, axis=0)
    trial_id = np.concatenate(tids, axis=0)

    stem = f"{prefix}_{split}" if split != "train" else prefix
    # train: challenge_X.npy；test: challenge_test_X.npy
    if split == "train":
        np.save(out_dir / f"{prefix}_X.npy", X)
        np.save(out_dir / f"{prefix}_y_task.npy", y_task)
        np.save(out_dir / f"{prefix}_y_three.npy", y_three)
        np.save(out_dir / f"{prefix}_subjects.npy", subjects)
        np.save(out_dir / f"{prefix}_trial_id.npy", trial_id)
    else:
        np.save(out_dir / f"{prefix}_test_X.npy", X)
        np.save(out_dir / f"{prefix}_test_y_task.npy", y_task)
        np.save(out_dir / f"{prefix}_test_y_three.npy", y_three)
        np.save(out_dir / f"{prefix}_test_subjects.npy", subjects)
        np.save(out_dir / f"{prefix}_test_trial_id.npy", trial_id)

    return {
        "split": split,
        "n": int(X.shape[0]),
        "shape": list(X.shape),
        "n_subjects": int(len(set(str(s) for s in subjects.tolist()))),
        "files": fids,
    }


def write_meta(out_dir: Path, *, protocol: str, channel_mode: str, merges: list[dict]) -> None:
    ch = 59 if channel_mode == "59" else 8
    meta = {
        "protocol": protocol,
        "zscore": True,
        "bandpass_hz": [8.0, 30.0],
        "notch_hz": 50.0,
        "car": True,
        "cut_then_filter": True,
        "win_sec": 3.0,
        "hop_sec": None,
        "fs_out": 250.0,
        "n_chans": ch,
        "channel_mode": channel_mode,
        "label_map": {"201": 0, "202": 1, "204": 2, "note": "Left/Right/Rest"},
        "y_three": "0=Left,1=Right,2=Rest",
        "y_task": "0=Rest,1=MI(Left|Right)",
        "merge": merges,
        "time": _utc_now(),
        "experiment": 34,
        "device_note": "preprocess machine-agnostic; train on 5070",
    }
    with (out_dir / "preprocess_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def run(
    *,
    mode: str,
    data_root: Path | None,
    out_dir: Path | None,
    limit_blocks: int | None,
    merge_only: bool,
    skip_test: bool,
) -> Path:
    channel_mode, protocol, prefix = _mode_cfg(mode)
    root = resolve_data_root(data_root)
    out = out_dir or (_PREPROCESS_ROOT / "out" / protocol)
    out.mkdir(parents=True, exist_ok=True)

    man = load_manifest(out / "manifest.json", protocol)
    man["protocol"] = protocol
    man.setdefault("files", {})

    if not merge_only:
        train_blocks = list_train_blocks(root)
        test_blocks = [] if skip_test else list_test_blocks(root)
        if limit_blocks is not None:
            train_blocks = train_blocks[: max(0, int(limit_blocks))]
            test_blocks = test_blocks[: max(0, int(limit_blocks))]

        trial_counter = [0]
        jobs = [("train", p) for p in train_blocks] + [("test", p) for p in test_blocks]
        print(f"[challenge_mi] root={root}")
        print(f"[challenge_mi] out={out} mode={channel_mode} n_jobs={len(jobs)}")
        for i, (split, path) in enumerate(jobs, 1):
            fid = fid_of(path, split)
            print(f"  [{i}/{len(jobs)}] {fid} …", flush=True)
            info = process_one(
                path,
                split=split,
                channel_mode=channel_mode,
                out_dir=out,
                trial_counter=trial_counter,
            )
            man["files"][fid] = info
            save_manifest(out / "manifest.json", man)
            st = info.get("status")
            if st == "ok":
                print(f"    ok n={info['n']} shape={info.get('shape')}")
            else:
                print(f"    FAIL {info.get('error')}")

    merges = []
    merges.append(merge_split(out, split="train", prefix=prefix, protocol=protocol))
    if not skip_test:
        try:
            merges.append(merge_split(out, split="test", prefix=prefix, protocol=protocol))
        except RuntimeError as exc:
            print(f"[warn] skip test merge: {exc}")
    write_meta(out, protocol=protocol, channel_mode=channel_mode, merges=merges)
    print("[challenge_mi] done", merges)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Exp34 challenge_mi_3s preprocess")
    p.add_argument("--mode", default="59", choices=["59", "8", "a59", "b8"])
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--limit-blocks", type=int, default=None)
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    args = p.parse_args(argv)
    run(
        mode=args.mode,
        data_root=args.data_root,
        out_dir=args.out_dir,
        limit_blocks=args.limit_blocks,
        merge_only=args.merge_only,
        skip_test=args.skip_test,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
