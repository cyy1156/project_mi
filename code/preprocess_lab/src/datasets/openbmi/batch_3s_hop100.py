"""OpenBMI 3s/hop100 批处理（实验 20）：按文件写 shard，再合并为训练 npy。

用法（在 preprocess_lab 根目录）：
  python -m src.datasets.openbmi.batch_3s_hop100 --limit 1
  python -m src.datasets.openbmi.batch_3s_hop100 --subjects 01,02
  python -m src.datasets.openbmi.batch_3s_hop100
  python -m src.datasets.openbmi.batch_3s_hop100 --merge-only
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

from src.datasets.openbmi.load_mat import parse_sess_subj
from src.datasets.openbmi.pipeline import (
    preprocess_file_3s_hop100,
    sanity_check_outputs,
)

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


def load_manifest(path: Path, *, protocol: str = "openbmi_3s_hop100") -> dict:
    if not path.exists():
        return {"version": 1, "protocol": protocol, "files": {}}
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


def load_shard(out_dir: Path, fid: str) -> dict[str, np.ndarray] | None:
    root = shard_dir(out_dir) / fid
    if not (root / "X.npy").exists():
        return None
    return {
        "X": np.load(root / "X.npy"),
        "y_task": np.load(root / "y_task.npy"),
        "y_three": np.load(root / "y_three.npy"),
        "subjects": np.load(root / "subjects.npy", allow_pickle=True),
        "trial_id": np.load(root / "trial_id.npy"),
    }


def merge_shards(
    out_dir: Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    protocol: str = "openbmi_3s_hop100",
    zscore: bool = True,
    bandpass_hz: tuple[float, float] = (8.0, 30.0),
) -> None:
    """内存友好合并：先统计 N，再写入 memmap，避免一次性装入全部 shard。"""
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

    # Pass 1: shapes
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

    # Pass 2: write memmaps
    paths = {
        "X": out_dir / "openbmi_X.npy",
        "y_task": out_dir / "openbmi_y_task.npy",
        "y_three": out_dir / "openbmi_y_three.npy",
        "subjects": out_dir / "openbmi_subjects.npy",
        "trial_id": out_dir / "openbmi_trial_id.npy",
    }
    for p in paths.values():
        p.unlink(missing_ok=True)

    X_mm = np.lib.format.open_memmap(
        paths["X"], mode="w+", dtype=np.float32, shape=(n_total, *x_shape_tail)
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
    # subjects 含 python str，不能直接 memmap float；先 list 再一次保存
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
        if (i + 1) % 10 == 0 or (i + 1) == len(fids):
            print(f"  wrote {i + 1}/{len(fids)} shards  cursor={cursor}/{n_total}")
            X_mm.flush()
            yt_mm.flush()
            y3_mm.flush()
            tid_mm.flush()
        del X, yt, y3, sid, tid
        gc.collect()

    assert cursor == n_total
    subjects = np.concatenate(subjects_chunks, axis=0)
    np.save(paths["subjects"], subjects)
    del subjects_chunks
    gc.collect()

    # flush & reopen read-only for checks / split
    del X_mm, yt_mm, y3_mm, tid_mm
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
    sanity_check_outputs(np.asarray(X[: min(8, len(X))]), y_task[: min(8, len(y_task))], y_three[: min(8, len(y_three))], n_times=750)
    # 全量 sanity（不整表拷贝 X）
    assert X.ndim == 4 and X.shape[1:] == (1, 8, 750), X.shape
    assert len(X) == len(y_task) == len(y_three) == len(subjects) == len(trial_id)
    assert set(np.unique(y_task)).issubset({0, 1})
    assert set(np.unique(y_three)).issubset({0, 1, 2})
    assert np.all((y_three == 0) == (y_task == 0))

    # 不写 train_*/val_*：五折训练只用 openbmi_*.npy；避免再占 ~10GiB
    for p in out_dir.glob("train_*.npy"):
        p.unlink(missing_ok=True)
    for p in out_dir.glob("val_*.npy"):
        p.unlink(missing_ok=True)

    meta = {
        "protocol": protocol,
        "zscore": bool(zscore),
        "bandpass_hz": [float(bandpass_hz[0]), float(bandpass_hz[1])],
        "blocks": ["EEG_MI_train"],
        "blocks_note": "train-only；不含官方 EEG_MI_test",
        "subject_key": "openbmi:subjNN (sess01+sess02 same person)",
        "n_windows": int(n_total),
        "n_subjects": int(n_subj),
        "n_shards": len(fids),
        "channels": ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"],
        "win_sec": 3.0,
        "hop_sec": 0.1,
        "fs_out": 250,
        "task": "cue_0_to_4s",
        "rest": "cue_before_4s",
        "source_blocks": ["EEG_MI_train"],
        "excluded_blocks": ["EEG_MI_test"],
        "label_map": {"left": 1, "right": 2, "rest": 0},
        "merge": "memmap",
        "wrote_train_val_split": False,
    }
    (out_dir / "preprocess_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("saved openbmi_*.npy + preprocess_meta.json →", out_dir)


def run_batch(
    data_glob: str,
    out_dir: Path,
    *,
    limit: int | None = None,
    reset: bool = False,
    merge: bool = True,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    zscore: bool = True,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    protocol: str | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if protocol is None:
        protocol = "openbmi_3s_hop100" if zscore else "openbmi_3s_hop100_noz"
    bandpass_hz = (float(l_freq), float(h_freq))

    if reset:
        sd = out_dir / "shards"
        if sd.exists():
            shutil.rmtree(sd)
        for p in out_dir.glob("openbmi_*.npy"):
            p.unlink(missing_ok=True)
        for p in out_dir.glob("train_*.npy"):
            p.unlink(missing_ok=True)
        for p in out_dir.glob("val_*.npy"):
            p.unlink(missing_ok=True)
        for p in (manifest_path, out_dir / "preprocess_meta.json"):
            p.unlink(missing_ok=True)
        print("已重置", out_dir)

    manifest = load_manifest(manifest_path, protocol=protocol)
    manifest["protocol"] = protocol
    manifest["bandpass_hz"] = [bandpass_hz[0], bandpass_hz[1]]
    files = [Path(p) for p in glob.glob(data_glob)]
    if subjects is not None:
        allow = {f"{int(s):02d}" for s in subjects}
        filtered = []
        for p in files:
            try:
                _, subj = parse_sess_subj(p)
            except ValueError:
                continue
            if subj.replace("subj", "") in allow:
                filtered.append(p)
        files = filtered
    if sessions is not None:
        allow_s = {s if s.startswith("sess") else f"sess{int(s):02d}" for s in sessions}
        filtered = []
        for p in files:
            try:
                sess, _ = parse_sess_subj(p)
            except ValueError:
                continue
            if sess in allow_s:
                filtered.append(p)
        files = filtered

    files = sorted(files, key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"没有匹配到数据文件: {data_glob}")
    if limit is not None:
        files = files[: int(limit)]
    print(
        f"[{protocol}] bp={bandpass_hz[0]}–{bandpass_hz[1]} Hz "
        f"候选 {len(files)} 个文件 → {out_dir} zscore={zscore}"
    )

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
            X, yt, y3, sid, tid, stats = preprocess_file_3s_hop100(
                fpath,
                zscore=zscore,
                l_freq=bandpass_hz[0],
                h_freq=bandpass_hz[1],
                protocol=protocol,
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
                "y_task": yt,
                "y_three": y3,
                "subjects": sid,
                "trial_id": tid,
            },
        )
        print(
            f"  ok {fpath.name}: X={X.shape} "
            f"task={np.bincount(yt, minlength=2).tolist()} "
            f"subj={sid[0] if len(sid) else '?'}"
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
        del X, yt, y3, sid, tid, stats
        gc.collect()

    print(f"shard 阶段完成 ok={n_ok} skip={n_skip} empty={n_empty} fail={n_fail}")
    if merge and (n_ok + n_skip) > 0:
        merge_shards(
            out_dir,
            protocol=protocol,
            zscore=zscore,
            bandpass_hz=bandpass_hz,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="OpenBMI 3s/hop100 shard 预处理")
    p.add_argument(
        "--glob",
        default=str(_REPO_ROOT / "DATA" / "openbmi" / "sess*_subj*_EEG_MI.mat"),
        help="默认 D:/MI/DATA/openbmi/sess*_subj*_EEG_MI.mat",
    )
    p.add_argument("--out", default=str(_PREPROCESS_ROOT / "out" / "openbmi_3s_hop100"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--no-merge", action="store_true")
    p.add_argument(
        "--no-zscore",
        action="store_true",
        help="关闭窗内 z-score（写出 openbmi_3s_hop100_noz）",
    )
    p.add_argument(
        "--subjects",
        default="all",
        help="逗号分隔被试号如 01,02；all=全部",
    )
    p.add_argument(
        "--sessions",
        default="all",
        help="逗号分隔 01,02 或 sess01；all=两场",
    )
    p.add_argument("--l-freq", type=float, default=8.0, help="带通下限 Hz（方案19：μ=8 / β=13）")
    p.add_argument("--h-freq", type=float, default=30.0, help="带通上限 Hz（方案19：μ=13 / β=30）")
    p.add_argument(
        "--band",
        choices=("full", "mu813", "beta1330"),
        default=None,
        help="快捷：full=8–30；mu813=8–13→out/..._mu813；beta1330=13–30→..._beta1330",
    )
    args = p.parse_args()
    zscore = not bool(args.no_zscore)
    l_freq, h_freq = float(args.l_freq), float(args.h_freq)
    out = Path(args.out)
    protocol = "openbmi_3s_hop100" if zscore else "openbmi_3s_hop100_noz"

    if args.band == "mu813":
        l_freq, h_freq = 8.0, 13.0
        protocol = "openbmi_3s_hop100_mu813" if zscore else "openbmi_3s_hop100_mu813_noz"
        if out.name in {"openbmi_3s_hop100", "openbmi_3s_hop100_noz"}:
            out = _PREPROCESS_ROOT / "out" / protocol
    elif args.band == "beta1330":
        l_freq, h_freq = 13.0, 30.0
        protocol = "openbmi_3s_hop100_beta1330" if zscore else "openbmi_3s_hop100_beta1330_noz"
        if out.name in {"openbmi_3s_hop100", "openbmi_3s_hop100_noz"}:
            out = _PREPROCESS_ROOT / "out" / protocol
    elif args.no_zscore and out.name == "openbmi_3s_hop100":
        out = _PREPROCESS_ROOT / "out" / "openbmi_3s_hop100_noz"
        protocol = "openbmi_3s_hop100_noz"
    elif (l_freq, h_freq) == (8.0, 13.0) and out.name == "openbmi_3s_hop100":
        protocol = "openbmi_3s_hop100_mu813"
        out = _PREPROCESS_ROOT / "out" / protocol
    elif (l_freq, h_freq) == (13.0, 30.0) and out.name == "openbmi_3s_hop100":
        protocol = "openbmi_3s_hop100_beta1330"
        out = _PREPROCESS_ROOT / "out" / protocol

    if args.merge_only:
        merge_shards(
            out,
            protocol=protocol,
            zscore=zscore,
            bandpass_hz=(l_freq, h_freq),
        )
        return

    subj_arg = str(args.subjects).strip().lower()
    subjects = None if subj_arg in {"all", "*"} else [
        x.strip() for x in str(args.subjects).split(",") if x.strip()
    ]
    sess_arg = str(args.sessions).strip().lower()
    sessions = None if sess_arg in {"all", "*"} else [
        x.strip() for x in str(args.sessions).split(",") if x.strip()
    ]

    run_batch(
        args.glob,
        out,
        limit=args.limit,
        reset=bool(args.reset),
        merge=not args.no_merge,
        subjects=subjects,
        sessions=sessions,
        zscore=zscore,
        l_freq=l_freq,
        h_freq=h_freq,
        protocol=protocol,
    )


if __name__ == "__main__":
    main()
