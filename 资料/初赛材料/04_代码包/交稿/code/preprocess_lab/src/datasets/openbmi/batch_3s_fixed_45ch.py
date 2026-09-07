"""OpenBMI 45ch 固定 3s 批处理（Exp36 C1）。

用法（preprocess_lab 根）：
  python -m src.datasets.openbmi.batch_3s_fixed_45ch --limit 2
  python -m src.datasets.openbmi.batch_3s_fixed_45ch
  python -m src.datasets.openbmi.batch_3s_fixed_45ch --merge-only
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.datasets.challenge_mi.channels_intersect import INTERSECT_45, PROTOCOL_OPENBMI_45
from src.datasets.openbmi.load_mat import parse_sess_subj
from src.datasets.openbmi.pipeline_45ch import N_CH, N_TIMES, preprocess_file_3s_fixed_45ch

_PREPROCESS_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[5]
FULL_KEYS = ("X", "y_task", "y_three", "subjects", "trial_id")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_id(path: Path) -> str:
    return path.stem


def file_fingerprint(path: Path) -> dict:
    st = path.stat()
    h = hashlib.sha1()
    with path.open("rb") as f:
        head = f.read(1 << 20)
        if st.st_size > (1 << 20):
            f.seek(max(0, st.st_size - (1 << 20)))
            tail = f.read(1 << 20)
        else:
            tail = b""
    h.update(head)
    h.update(tail)
    return {
        "size": int(st.st_size),
        "sha1_sample": h.hexdigest(),
        "mtime": int(st.st_mtime),
    }


def load_manifest(path: Path, *, protocol: str = PROTOCOL_OPENBMI_45) -> dict:
    if not path.exists():
        return {"version": 1, "protocol": protocol, "files": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def shard_dir(out_dir: Path) -> Path:
    d = out_dir / "shards"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_shard(out_dir: Path, fid: str, arrays: dict[str, np.ndarray]) -> None:
    root = shard_dir(out_dir) / fid
    root.mkdir(parents=True, exist_ok=True)
    for k, arr in arrays.items():
        np.save(root / f"{k}.npy", arr)


def merge_shards(out_dir: Path, *, protocol: str = PROTOCOL_OPENBMI_45) -> None:
    out_dir = Path(out_dir)
    man = load_manifest(out_dir / "manifest.json", protocol=protocol)
    fids = sorted(
        fid
        for fid, rec in man.get("files", {}).items()
        if rec.get("status") == "ok"
    )
    if not fids:
        raise RuntimeError("无可用 shard")

    n_total = 0
    shapes = []
    for fid in fids:
        root = shard_dir(out_dir) / fid
        X = np.load(root / "X.npy", mmap_mode="r")
        n_total += int(X.shape[0])
        shapes.append(X.shape[1:])
    assert all(s == shapes[0] for s in shapes), shapes

    paths = {k: out_dir / f"openbmi_{k}.npy" for k in FULL_KEYS}
    X_mm = np.lib.format.open_memmap(
        paths["X"], mode="w+", dtype=np.float32, shape=(n_total, *shapes[0])
    )
    yt_mm = np.lib.format.open_memmap(
        paths["y_task"], mode="w+", dtype=np.int64, shape=(n_total,)
    )
    y3_mm = np.lib.format.open_memmap(
        paths["y_three"], mode="w+", dtype=np.int64, shape=(n_total,)
    )
    tid_mm = np.lib.format.open_memmap(
        paths["trial_id"], mode="w+", dtype=np.int64, shape=(n_total,)
    )
    subjects_chunks: list[np.ndarray] = []
    cursor = 0
    for fid in fids:
        root = shard_dir(out_dir) / fid
        X = np.load(root / "X.npy")
        n = int(X.shape[0])
        X_mm[cursor : cursor + n] = X
        yt_mm[cursor : cursor + n] = np.load(root / "y_task.npy")
        y3_mm[cursor : cursor + n] = np.load(root / "y_three.npy")
        tid_mm[cursor : cursor + n] = np.load(root / "trial_id.npy")
        subjects_chunks.append(np.load(root / "subjects.npy", allow_pickle=True))
        cursor += n
        del X
        gc.collect()
    assert cursor == n_total
    subjects = np.concatenate(subjects_chunks, axis=0)
    np.save(paths["subjects"], subjects)
    del X_mm, yt_mm, y3_mm, tid_mm, subjects_chunks
    gc.collect()

    X = np.load(paths["X"], mmap_mode="r")
    y_task = np.load(paths["y_task"])
    y_three = np.load(paths["y_three"])
    subjects = np.load(paths["subjects"], allow_pickle=True)
    assert X.shape == (n_total, 1, N_CH, N_TIMES), X.shape
    meta = {
        "protocol": protocol,
        "zscore": True,
        "bandpass_hz": [8.0, 30.0],
        "blocks": ["EEG_MI_train"],
        "n_windows": int(n_total),
        "n_subjects": int(len(set(subjects.tolist()))),
        "n_chans": N_CH,
        "channels": list(INTERSECT_45),
        "win_sec": 3.0,
        "hop_sec": None,
        "fs_out": 250,
        "task": "cue_0_to_3s",
        "rest": "cue_before_3s",
        "label_map": {"left": 1, "right": 2, "rest": 0},
        "shape": list(X.shape),
        "time": _utc_now(),
        "experiment": 36,
    }
    (out_dir / "preprocess_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("merged", tuple(X.shape), "subjects", meta["n_subjects"], "→", out_dir)


def run_batch(
    data_glob: str,
    out_dir: Path,
    *,
    limit: int | None = None,
    reset: bool = False,
    merge: bool = True,
    subjects: list[str] | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    protocol = PROTOCOL_OPENBMI_45

    if reset:
        sd = out_dir / "shards"
        if sd.exists():
            shutil.rmtree(sd)
        for p in out_dir.glob("openbmi_*.npy"):
            p.unlink(missing_ok=True)
        for p in (manifest_path, out_dir / "preprocess_meta.json"):
            p.unlink(missing_ok=True)
        print("已重置", out_dir)

    manifest = load_manifest(manifest_path, protocol=protocol)
    paths = sorted(Path(p) for p in glob.glob(data_glob))
    if subjects:
        want = {f"{int(s):02d}" for s in subjects}
        paths = [p for p in paths if parse_sess_subj(p)[1].replace("subj", "") in want
                 or parse_sess_subj(p)[1][-2:] in want]
    if limit is not None:
        paths = paths[: int(limit)]
    print(f"files={len(paths)} out={out_dir}", flush=True)

    for i, path in enumerate(paths, 1):
        fid = file_id(path)
        fp = file_fingerprint(path)
        prev = manifest.get("files", {}).get(fid)
        shard_ok = (shard_dir(out_dir) / fid / "X.npy").is_file()
        if (
            prev
            and prev.get("status") == "ok"
            and prev.get("fingerprint") == fp
            and shard_ok
        ):
            print(f"[{i}/{len(paths)}] skip {fid}", flush=True)
            continue
        try:
            X, yt, y3, subjects_arr, tid, stats = preprocess_file_3s_fixed_45ch(path)
            if len(yt) == 0:
                raise RuntimeError("empty windows")
            save_shard(
                out_dir,
                fid,
                {
                    "X": X,
                    "y_task": yt,
                    "y_three": y3,
                    "subjects": subjects_arr,
                    "trial_id": tid,
                },
            )
            manifest.setdefault("files", {})[fid] = {
                "status": "ok",
                "n": int(len(yt)),
                "shape": list(X.shape),
                "subject": stats.get("subject"),
                "fingerprint": fp,
                "time": _utc_now(),
            }
            print(
                f"[{i}/{len(paths)}] ok {fid} n={len(yt)} shape={tuple(X.shape)}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            manifest.setdefault("files", {})[fid] = {
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
                "fingerprint": fp,
                "time": _utc_now(),
            }
            print(f"[{i}/{len(paths)}] FAIL {fid}: {exc}", flush=True)
        save_manifest(manifest_path, manifest)

    if merge:
        merge_shards(out_dir, protocol=protocol)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-glob",
        default=str(_REPO_ROOT / "DATA" / "openbmi" / "*EEG_MI.mat"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_PREPROCESS_ROOT / "out" / PROTOCOL_OPENBMI_45,
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--subjects", type=str, default="")
    args = ap.parse_args()
    if args.merge_only:
        merge_shards(args.out)
        return 0
    subs = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    run_batch(
        args.data_glob,
        args.out,
        limit=args.limit,
        reset=args.reset,
        subjects=subs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
