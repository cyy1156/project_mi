"""Stieger Cue 3s/hop100 批处理（实验 29）：Task=Cue后0–4s，与 OpenBMI/fnz 对齐。"""
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

from src.datasets.stieger.pipeline_cue_3s_hop100 import (
    preprocess_session_cue_3s_hop100,
    sanity_check_outputs,
)

_PREPROCESS_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[5]
FULL_KEYS = ("X", "X_noz", "y_task", "y_three", "subjects", "trial_id")
NPY_PREFIX = "stieger_cue"

# 实验 29 预注册被试（Session_2–4 各 ≥1 文件）
EXP29_SUBJECTS = ("S1", "S2", "S4", "S8", "S12")
EXP29_SESSIONS = (2, 3, 4)


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
    return {"size": int(st.st_size), "sha1_sample": h.hexdigest(), "mtime": int(st.st_mtime)}


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "protocol": "slide_3s_hop100_cue", "files": {}}
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


def _subject_from_stem(stem: str) -> str | None:
    m = re.match(r"(S\d+)_Session_", stem, flags=re.IGNORECASE)
    if not m:
        return None
    return f"S{int(m.group(1)[1:])}"


def _session_num_from_stem(stem: str) -> int | None:
    m = re.match(r"S\d+_Session_(\d+)", stem, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


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

    print(f"合并 {len(fids)} 个 shard → {out_dir}（memmap）")

    n_total = 0
    x_shape_tail: tuple[int, ...] | None = None
    for fid in fids:
        xp = shard_dir(out_dir) / fid / "X.npy"
        Xh = np.load(xp, mmap_mode="r")
        shape = tuple(int(s) for s in Xh.shape)
        n = int(shape[0])
        n_total += n
        if x_shape_tail is None:
            x_shape_tail = shape[1:]
        elif shape[1:] != x_shape_tail:
            raise RuntimeError(f"{fid} X shape 不一致: {shape} vs {(n,)+x_shape_tail}")
        del Xh
    assert x_shape_tail is not None
    print(f"  预计全量 N={n_total} X=({n_total}, {', '.join(map(str, x_shape_tail))})")

    paths = {k: out_dir / f"{NPY_PREFIX}_{k}.npy" for k in FULL_KEYS}
    for p in paths.values():
        p.unlink(missing_ok=True)

    X_mm = np.lib.format.open_memmap(
        paths["X"], mode="w+", dtype=np.float32, shape=(n_total, *x_shape_tail)
    )
    X_noz_mm = np.lib.format.open_memmap(
        paths["X_noz"], mode="w+", dtype=np.float32, shape=(n_total, *x_shape_tail)
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
    tid_offset = 0
    for i, fid in enumerate(fids):
        root = shard_dir(out_dir) / fid
        X = np.load(root / "X.npy", mmap_mode="r")
        noz_path = root / "X_noz.npy"
        X_noz = np.load(noz_path, mmap_mode="r") if noz_path.exists() else None
        yt = np.load(root / "y_task.npy")
        y3 = np.load(root / "y_three.npy")
        sid = np.load(root / "subjects.npy", allow_pickle=True)
        tid = np.load(root / "trial_id.npy")
        n = int(X.shape[0])
        X_mm[cursor : cursor + n] = X
        if X_noz is not None:
            X_noz_mm[cursor : cursor + n] = X_noz
        else:
            X_noz_mm[cursor : cursor + n] = np.nan
        yt_mm[cursor : cursor + n] = yt
        y3_mm[cursor : cursor + n] = y3
        tid_mm[cursor : cursor + n] = tid.astype(np.int64) + tid_offset
        subjects_chunks.append(np.asarray(sid, dtype=object))
        tid_offset = int(tid_mm[cursor + n - 1]) + 1 if n else tid_offset
        cursor += n
        if (i + 1) % 5 == 0 or (i + 1) == len(fids):
            print(f"  wrote {i + 1}/{len(fids)} shards  cursor={cursor}/{n_total}")
            X_mm.flush()
            X_noz_mm.flush()
            yt_mm.flush()
            y3_mm.flush()
            tid_mm.flush()
        del X, X_noz, yt, y3, sid, tid
        gc.collect()

    assert cursor == n_total
    subjects = np.concatenate(subjects_chunks, axis=0)
    np.save(paths["subjects"], subjects)
    gc.collect()

    del X_mm, X_noz_mm, yt_mm, y3_mm, tid_mm
    gc.collect()

    X = np.load(paths["X"], mmap_mode="r")
    y_task = np.load(paths["y_task"])
    y_three = np.load(paths["y_three"])
    n_subj = len(set(subjects.tolist()))
    print(
        "全量:",
        tuple(X.shape),
        "y_task",
        np.bincount(y_task, minlength=2),
        "subjects(unique keys)",
        n_subj,
    )
    n_chk = min(8, len(X))
    sanity_check_outputs(np.asarray(X[:n_chk]), y_task[:n_chk], y_three[:n_chk])
    print(f"saved {NPY_PREFIX}_*.npy →", out_dir)


def run_batch(
    data_glob: str,
    out_dir: Path,
    *,
    baseline_sec: float = 0.5,
    task_sec: float = 4.0,
    rest_sec: float = 4.0,
    limit: int | None = None,
    reset: bool = False,
    merge: bool = True,
    subjects: list[str] | None = None,
    sessions: list[int] | None = None,
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
    files = [Path(p) for p in glob.glob(data_glob)]
    if subjects is not None:
        allow = {s.strip().upper() for s in subjects if s.strip()}
        files = [p for p in files if (_subject_from_stem(p.stem) or "").upper() in allow]
    if sessions is not None:
        allow_sess = {int(s) for s in sessions}
        files = [
            p
            for p in files
            if (_session_num_from_stem(p.stem) or -1) in allow_sess
        ]
    files = sorted(files, key=lambda p: (_session_num_from_stem(p.stem) or 0, p.name))
    if not files:
        raise FileNotFoundError(f"没有匹配到数据文件: {data_glob}")
    if limit is not None:
        files = files[: int(limit)]
    print(f"[stieger_cue_3s_hop100] 候选 {len(files)} 个文件 → {out_dir}")

    n_ok = n_skip = n_empty = n_fail = 0
    for fpath in files:
        fid = file_id(fpath)
        fp = file_fingerprint(fpath)
        prev = manifest["files"].get(fid)
        shard_ok = (shard_dir(out_dir) / fid / "X.npy").exists()
        if prev and prev.get("status") == "ok" and prev.get("fingerprint") == fp and shard_ok:
            print(f"  skip {fpath.name}")
            n_skip += 1
            continue

        try:
            X, X_noz, yt, y3, sid, tid, stats = preprocess_session_cue_3s_hop100(
                fpath,
                baseline_sec=baseline_sec,
                task_sec=task_sec,
                rest_sec=rest_sec,
            )
        except Exception as e:
            print(f"  FAIL {fpath.name}: {type(e).__name__}: {e}")
            manifest["files"][fid] = {
                "status": "fail",
                "error": f"{type(e).__name__}: {e}",
                "fingerprint": fp,
                "time": _utc_now(),
            }
            n_fail += 1
            save_manifest(manifest_path, manifest)
            gc.collect()
            continue

        if len(yt) == 0:
            print(f"  empty {fpath.name} stats={stats}")
            manifest["files"][fid] = {
                "status": "empty",
                "stats": stats,
                "fingerprint": fp,
                "time": _utc_now(),
            }
            n_empty += 1
            save_manifest(manifest_path, manifest)
            continue

        save_shard(
            out_dir,
            fid,
            {
                "X": X,
                "X_noz": X_noz,
                "y_task": yt,
                "y_three": y3,
                "subjects": sid,
                "trial_id": tid,
            },
        )
        print(
            f"  ok {fpath.name}: X={X.shape} "
            f"task={np.bincount(yt, minlength=2).tolist()} "
            f"rest_w={stats['n_rest_wins']} task_w={stats['n_task_wins']}"
        )
        manifest["files"][fid] = {
            "status": "ok",
            "n": int(len(yt)),
            "stats": stats,
            "fingerprint": fp,
            "time": _utc_now(),
        }
        n_ok += 1
        save_manifest(manifest_path, manifest)
        del X, X_noz, yt, y3, sid, tid, stats
        gc.collect()

    print(f"shard 阶段完成 ok={n_ok} skip={n_skip} empty={n_empty} fail={n_fail}")
    if merge:
        merge_shards(out_dir)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Stieger Cue 3s/hop100（实验 29）")
    p.add_argument(
        "--glob",
        default=str(_REPO_ROOT / "DATA" / "stieger" / "S*_Session_*.mat"),
    )
    p.add_argument(
        "--out",
        default=str(_PREPROCESS_ROOT / "out" / "stieger_cue_3s_hop100"),
    )
    p.add_argument("--baseline-sec", type=float, default=0.5)
    p.add_argument("--task-sec", type=float, default=4.0)
    p.add_argument("--rest-sec", type=float, default=4.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--no-merge", action="store_true")
    p.add_argument(
        "--subjects",
        default="exp29",
        help="exp29=S1,S2,S4,S7,S12；all=全部；或逗号分隔",
    )
    p.add_argument(
        "--sessions",
        default="2,3,4",
        help="逗号分隔 Session 编号；exp29 默认 2,3,4（跳过 baseline Session_1）",
    )
    args = p.parse_args()

    if args.merge_only:
        merge_shards(Path(args.out))
        return

    subj_arg = str(args.subjects).strip().lower()
    if subj_arg in {"all", "*"}:
        subjects = None
    elif subj_arg == "exp29":
        subjects = list(EXP29_SUBJECTS)
    else:
        subjects = [x.strip() for x in str(args.subjects).split(",") if x.strip()]

    sess_arg = str(args.sessions).strip().lower()
    if sess_arg in {"all", "*"}:
        sessions = None
    elif sess_arg == "exp29":
        sessions = list(EXP29_SESSIONS)
    else:
        sessions = _parse_int_list(sess_arg)

    run_batch(
        args.glob,
        Path(args.out),
        baseline_sec=args.baseline_sec,
        task_sec=args.task_sec,
        rest_sec=args.rest_sec,
        limit=args.limit,
        reset=bool(args.reset),
        merge=not args.no_merge,
        subjects=subjects,
        sessions=sessions,
    )


if __name__ == "__main__":
    main()
