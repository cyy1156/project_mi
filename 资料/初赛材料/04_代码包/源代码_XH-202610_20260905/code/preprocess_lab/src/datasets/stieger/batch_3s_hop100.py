"""Stieger 3 s / 100 ms 滑窗（旁路）批处理：每会话写 shard，结束（或 --merge-only）再合并为训练用 npy。"""
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

from src.datasets.stieger.pipeline_3s_hop100 import preprocess_session_3s_hop100, sanity_check_outputs

_PREPROCESS_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[5]
FULL_KEYS = ("X", "X_noz", "y_task", "y_three", "subjects", "trial_id")

# 与现有 stieger_2s 对齐的 15 名被试（默认只跑这些）
DEFAULT_SUBJECTS_3S = (
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S12",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
    "S18",
    "S19",
    "S20",
    "S21",
    "S22",
    "S23",
    "S24",
    "S25",

)


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
        return {"version": 1, "protocol": "slide_3s_hop100", "files": {}}
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


def shard_dir(out_dir: Path) -> Path:
    d = out_dir / "shards"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_shard(out_dir: Path, fid: str, arrays: dict[str, np.ndarray]) -> None:
    root = shard_dir(out_dir) / fid
    root.mkdir(parents=True, exist_ok=True)
    for k, arr in arrays.items():
        np.save(root / f"{k}.npy", arr)


def load_shard(out_dir: Path, fid: str) -> dict[str, np.ndarray] | None:
    root = shard_dir(out_dir) / fid
    if not (root / "X.npy").exists():
        return None
    return {
        "X": np.load(root / "X.npy"),
        "X_noz": np.load(root / "X_noz.npy")
        if (root / "X_noz.npy").exists()
        else None,
        "y_task": np.load(root / "y_task.npy"),
        "y_three": np.load(root / "y_three.npy"),
        "subjects": np.load(root / "subjects.npy", allow_pickle=True),
        "trial_id": np.load(root / "trial_id.npy"),
    }


def merge_shards(out_dir: Path, *, val_ratio: float = 0.2, seed: int = 42) -> None:
    """内存友好合并：先统计 N，再 memmap 逐 shard 写入，避免一次性 np.concatenate。"""
    del val_ratio, seed  # 伪在线 07 不用 train_/val_ 离线划分
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

    paths = {k: out_dir / f"stieger_{k}.npy" for k in FULL_KEYS}
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
    missing_noz = 0
    for i, fid in enumerate(fids):
        root = shard_dir(out_dir) / fid
        X = np.load(root / "X.npy", mmap_mode="r")
        noz_path = root / "X_noz.npy"
        if noz_path.exists():
            X_noz = np.load(noz_path, mmap_mode="r")
        else:
            missing_noz += 1
            X_noz = None
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
        if (i + 1) % 10 == 0 or (i + 1) == len(fids):
            print(f"  wrote {i + 1}/{len(fids)} shards  cursor={cursor}/{n_total}")
            X_mm.flush()
            X_noz_mm.flush()
            yt_mm.flush()
            y3_mm.flush()
            tid_mm.flush()
        del X, X_noz, yt, y3, sid, tid
        gc.collect()

    if missing_noz:
        print(f"WARN: {missing_noz} shard 缺 X_noz（门控不可用，需重跑预处理）")

    assert cursor == n_total
    subjects = np.concatenate(subjects_chunks, axis=0)
    np.save(paths["subjects"], subjects)
    del subjects_chunks
    gc.collect()

    del X_mm, X_noz_mm, yt_mm, y3_mm, tid_mm
    gc.collect()

    X = np.load(paths["X"], mmap_mode="r")
    y_task = np.load(paths["y_task"])
    y_three = np.load(paths["y_three"])
    subjects = np.load(paths["subjects"], allow_pickle=True)
    trial_id = np.load(paths["trial_id"])
    n_subj = len(set(subjects.tolist()))
    print(
        "全量:",
        tuple(X.shape),
        "y_task",
        np.bincount(y_task, minlength=2),
        "subjects",
        n_subj,
    )
    n_chk = min(8, len(X))
    sanity_check_outputs(
        np.asarray(X[:n_chk]),
        y_task[:n_chk],
        y_three[:n_chk],
    )
    assert X.ndim == 4 and len(X) == len(y_task) == len(y_three) == len(subjects) == len(trial_id)
    assert set(np.unique(y_task)).issubset({0, 1})
    assert set(np.unique(y_three)).issubset({0, 1, 2})
    assert np.all((y_three == 0) == (y_task == 0))

    for p in out_dir.glob("train_*.npy"):
        p.unlink(missing_ok=True)
    for p in out_dir.glob("val_*.npy"):
        p.unlink(missing_ok=True)

    print("saved stieger_*.npy →", out_dir)


def run_batch(
    data_glob: str,
    out_dir: Path,
    *,
    feedback_t_ms: float = 2000.0,
    baseline_sec: float = 0.5,
    rest_sec: float = 4.0,
    limit: int | None = None,
    reset: bool = False,
    merge: bool = True,
    subjects: list[str] | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    if reset:
        sd = out_dir / "shards"
        if sd.exists():
            shutil.rmtree(sd)
        for p in out_dir.glob("stieger_*.npy"):
            p.unlink(missing_ok=True)
        for p in out_dir.glob("train_*.npy"):
            p.unlink(missing_ok=True)
        for p in out_dir.glob("val_*.npy"):
            p.unlink(missing_ok=True)
        if manifest_path.exists():
            manifest_path.unlink()
        print("已重置 out/stieger_3s_hop100")

    manifest = load_manifest(manifest_path)
    files = [Path(p) for p in glob.glob(data_glob)]
    if subjects is not None:
        allow = {s.strip().upper() for s in subjects if s.strip()}
        files = [
            p
            for p in files
            if (_subject_from_stem(p.stem) or "").upper() in allow
        ]
    files = sorted(files, key=lambda p: (p.stat().st_size, p.name))
    if not files:
        raise FileNotFoundError(f"没有匹配到数据文件: {data_glob}")
    if limit is not None:
        files = files[: int(limit)]
    print(f"[stieger_3s_hop100] 候选 {len(files)} 个文件 → {out_dir}")

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
            X, X_noz, yt, y3, sid, tid, stats = preprocess_session_3s_hop100(
                fpath,
                feedback_t_ms=feedback_t_ms,
                baseline_sec=baseline_sec,
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


def main() -> None:
    p = argparse.ArgumentParser(description="Stieger 3s/hop100 shard 预处理（实验伪在线 07）")
    p.add_argument(
        "--glob",
        default=str(_REPO_ROOT / "DATA" / "stieger" / "S*_Session_*.mat"),
    )
    p.add_argument("--out", default=str(_PREPROCESS_ROOT / "out" / "stieger_3s_hop100"))
    p.add_argument("--feedback-t-ms", type=float, default=2000.0)
    p.add_argument("--baseline-sec", type=float, default=0.5)
    p.add_argument("--rest-sec", type=float, default=4.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--merge-only", action="store_true", help="只合并已有 shard")
    p.add_argument("--no-merge", action="store_true", help="只写 shard，不合并")
    p.add_argument(
        "--subjects",
        default="all",
        help="逗号分隔被试；all=DATA/stieger 全部会话；from_2s=仅旧 stieger_2s 同 15 人",
    )
    args = p.parse_args()

    if args.merge_only:
        merge_shards(Path(args.out))
        return

    subj_arg = str(args.subjects).strip().lower()
    if subj_arg in {"all", "*"}:
        subjects = None
    elif subj_arg in {"from_2s", "2s", "default"}:
        subjects = list(DEFAULT_SUBJECTS_3S)
    else:
        subjects = [x.strip() for x in str(args.subjects).split(",") if x.strip()]

    run_batch(
        args.glob,
        Path(args.out),
        feedback_t_ms=args.feedback_t_ms,
        baseline_sec=args.baseline_sec,
        rest_sec=args.rest_sec,
        limit=args.limit,
        reset=bool(args.reset),
        merge=not args.no_merge,
        subjects=subjects,
    )


if __name__ == "__main__":
    main()
