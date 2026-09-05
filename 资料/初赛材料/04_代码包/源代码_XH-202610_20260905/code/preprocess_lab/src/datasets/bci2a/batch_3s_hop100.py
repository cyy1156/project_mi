"""BCI2a 3 s / hop100 批处理（实验 29）：每 run 写 shard，再合并。

- 输入：A01T.mat … A09T.mat（**仅 Training · 不用 E.gdf**）
- Task：Cue 后 0–4 s；Rest：Cue 前 4 s（与 OpenBMI/fnz 同口径）
- subjects 键：`A01|run3`（loader 内 session 字段，即 mat 中带标签 run）
"""
from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.datasets.bci2a.load_mat import load_bci2a_mat
from src.datasets.bci2a.pipeline import preprocess_run_3s_hop100, sanity_check_outputs

_PREPROCESS_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[5]
FULL_KEYS = ("X", "y_task", "y_three", "subjects", "trial_id")
NPY_PREFIX = "bci2a"

# 实验 29：每人 T 文件内 **最后 6 个带标签 run** → 3 伪 session（各 2 run，按顺序配对）
EXP29_RUNS_PER_SUBJECT = 6
EXP29_PSEUDO_SESSIONS = ("S1", "S2", "S3")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    return {"size": int(st.st_size), "sha1_sample": h.hexdigest(), "mtime": int(st.st_mtime)}


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "protocol": "bci2a_3s_hop100", "files": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    try:
        tmp.replace(path)
    except PermissionError:
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        tmp.unlink(missing_ok=True)


def shard_dir(out_dir: Path) -> Path:
    d = out_dir / "shards"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_shard(out_dir: Path, fid: str, arrays: dict[str, np.ndarray]) -> None:
    root = shard_dir(out_dir) / fid
    root.mkdir(parents=True, exist_ok=True)
    for k, arr in arrays.items():
        np.save(root / f"{k}.npy", arr)


def merge_shards(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    man = load_manifest(out_dir / "manifest.json")
    fids = sorted(
        fid
        for fid, meta in man.get("files", {}).items()
        if meta.get("status") == "ok" and (shard_dir(out_dir) / fid / "X.npy").exists()
    )
    if not fids:
        raise RuntimeError("没有可合并的 ok shard")

    print(f"合并 {len(fids)} 个 shard → {out_dir}")

    n_total = 0
    x_shape_tail: tuple[int, ...] | None = None
    for fid in fids:
        Xh = np.load(shard_dir(out_dir) / fid / "X.npy", mmap_mode="r")
        shape = tuple(int(s) for s in Xh.shape)
        n_total += int(shape[0])
        if x_shape_tail is None:
            x_shape_tail = shape[1:]
        del Xh
    assert x_shape_tail is not None

    paths = {k: out_dir / f"{NPY_PREFIX}_{k}.npy" for k in FULL_KEYS}
    for p in paths.values():
        p.unlink(missing_ok=True)

    X_mm = np.lib.format.open_memmap(
        paths["X"], mode="w+", dtype=np.float32, shape=(n_total, *x_shape_tail)
    )
    yt_mm = np.lib.format.open_memmap(paths["y_task"], mode="w+", dtype=np.int64, shape=(n_total,))
    y3_mm = np.lib.format.open_memmap(paths["y_three"], mode="w+", dtype=np.int64, shape=(n_total,))
    tid_mm = np.lib.format.open_memmap(paths["trial_id"], mode="w+", dtype=np.int64, shape=(n_total,))
    subjects_chunks: list[np.ndarray] = []

    cursor = 0
    tid_offset = 0
    for i, fid in enumerate(fids):
        root = shard_dir(out_dir) / fid
        X = np.load(root / "X.npy", mmap_mode="r")
        yt = np.load(root / "y_task.npy")
        y3 = np.load(root / "y_three.npy")
        sid = np.load(root / "subjects.npy", allow_pickle=True)
        tid = np.load(root / "trial_id.npy")
        n = int(X.shape[0])
        X_mm[cursor : cursor + n] = X
        yt_mm[cursor : cursor + n] = yt
        y3_mm[cursor : cursor + n] = y3
        tid_mm[cursor : cursor + n] = tid.astype(np.int64) + tid_offset
        subjects_chunks.append(np.asarray(sid, dtype=object))
        tid_offset = int(tid_mm[cursor + n - 1]) + 1 if n else tid_offset
        cursor += n
        if (i + 1) % 9 == 0 or (i + 1) == len(fids):
            print(f"  wrote {i + 1}/{len(fids)}  cursor={cursor}/{n_total}")
            X_mm.flush()
            yt_mm.flush()
            y3_mm.flush()
            tid_mm.flush()
        del X, yt, y3, sid, tid
        gc.collect()

    subjects = np.concatenate(subjects_chunks, axis=0)
    np.save(paths["subjects"], subjects)

    X = np.load(paths["X"], mmap_mode="r")
    y_task = np.load(paths["y_task"])
    y_three = np.load(paths["y_three"])
    print(
        "全量:",
        tuple(X.shape),
        "y_task",
        np.bincount(y_task, minlength=2),
        "y_three",
        np.bincount(y_three, minlength=3),
        "keys",
        len(set(subjects.tolist())),
    )
    sanity_check_outputs(np.asarray(X[: min(8, len(X))]), y_task[: min(8, len(y_task))], y_three[: min(8, len(y_three))], n_times=750)
    print(f"saved {NPY_PREFIX}_*.npy →", out_dir)


def _subject_from_mat(path: Path) -> str:
    return path.stem[:3].upper()


def _pseudo_session_for_run(run_index: int, n_runs: int) -> str | None:
    """run_index：该被试带标签 run 列表中的 0-based 下标。"""
    if n_runs < EXP29_RUNS_PER_SUBJECT:
        return None
    base = n_runs - EXP29_RUNS_PER_SUBJECT
    rel = run_index - base
    if rel < 0 or rel >= EXP29_RUNS_PER_SUBJECT:
        return None
    return EXP29_PSEUDO_SESSIONS[rel // 2]


def run_batch(
    data_glob: str,
    out_dir: Path,
    *,
    subjects: list[str] | None = None,
    limit: int | None = None,
    reset: bool = False,
    merge: bool = True,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    if reset:
        sd = out_dir / "shards"
        if sd.exists():
            shutil.rmtree(sd)
        for p in out_dir.glob(f"{NPY_PREFIX}_*.npy"):
            p.unlink(missing_ok=True)
        if manifest_path.exists():
            manifest_path.unlink()
        print(f"已重置 {out_dir}")

    manifest = load_manifest(manifest_path)
    mats = sorted(Path(p) for p in glob.glob(data_glob))
    if subjects is not None:
        allow = {s.strip().upper() for s in subjects}
        mats = [p for p in mats if _subject_from_mat(p) in allow]
    if limit is not None:
        mats = mats[: int(limit)]
    if not mats:
        raise FileNotFoundError(f"没有匹配: {data_glob}")

    print(f"[bci2a_3s_hop100] {len(mats)} 个 T.mat → {out_dir}")
    n_ok = n_skip = n_empty = n_fail = 0

    for mat_path in mats:
        subj = _subject_from_mat(mat_path)
        fp = file_fingerprint(mat_path)
        try:
            runs = load_bci2a_mat(mat_path)
        except Exception as e:
            print(f"  FAIL load {mat_path.name}: {e}")
            n_fail += 1
            continue

        n_runs = len(runs)
        for ri, eeg in enumerate(runs):
            fid = f"{subj}_{eeg.session}"
            prev = manifest["files"].get(fid)
            shard_ok = (shard_dir(out_dir) / fid / "X.npy").exists()
            if prev and prev.get("status") == "ok" and prev.get("fingerprint") == fp and shard_ok:
                print(f"  skip {fid}")
                n_skip += 1
                continue

            try:
                X, yt, y3, tid = preprocess_run_3s_hop100(eeg, add_rest=True)
            except Exception as e:
                print(f"  FAIL {fid}: {type(e).__name__}: {e}")
                manifest["files"][fid] = {"status": "fail", "error": str(e), "time": _utc_now()}
                n_fail += 1
                save_manifest(manifest_path, manifest)
                continue

            if len(yt) == 0:
                print(f"  empty {fid}")
                manifest["files"][fid] = {"status": "empty", "time": _utc_now()}
                n_empty += 1
                save_manifest(manifest_path, manifest)
                continue

            ps = _pseudo_session_for_run(ri, n_runs)
            sid_key = f"{subj}|{eeg.session}" + (f"|{ps}" if ps else "")
            sid = np.array([sid_key] * len(yt), dtype=object)

            save_shard(
                out_dir,
                fid,
                {"X": X, "y_task": yt, "y_three": y3, "subjects": sid, "trial_id": tid},
            )
            print(
                f"  ok {fid}: X={X.shape} task={np.bincount(yt, minlength=2).tolist()} ps={ps}"
            )
            manifest["files"][fid] = {
                "status": "ok",
                "n": int(len(yt)),
                "subject": subj,
                "run": str(eeg.session),
                "pseudo_session": ps,
                "fingerprint": fp,
                "time": _utc_now(),
            }
            n_ok += 1
            save_manifest(manifest_path, manifest)
            del X, yt, y3, sid, tid
            gc.collect()

    print(f"shard ok={n_ok} skip={n_skip} empty={n_empty} fail={n_fail}")
    if merge:
        merge_shards(out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="BCI2a 3s/hop100（实验 29 · 仅 T.mat）")
    p.add_argument("--glob", default=str(_REPO_ROOT / "DATA" / "bci2a" / "A0*T.mat"))
    p.add_argument("--out", default=str(_PREPROCESS_ROOT / "out" / "bci2a_3s_hop100"))
    p.add_argument("--subjects", default="all", help="all 或 A01,A02,…")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--no-merge", action="store_true")
    args = p.parse_args()

    if args.merge_only:
        merge_shards(Path(args.out))
        return

    subj_arg = str(args.subjects).strip().upper()
    subjects = None if subj_arg in {"ALL", "*"} else [x.strip().upper() for x in subj_arg.split(",") if x.strip()]

    run_batch(
        args.glob,
        Path(args.out),
        subjects=subjects,
        limit=args.limit,
        reset=bool(args.reset),
        merge=not args.no_merge,
    )


if __name__ == "__main__":
    main()
