"""OpenBMI pf1000 批处理：shard → 合并 openbmi_*.npy。

用法（在 preprocess_lab 根目录）：
  python -m src.datasets.openbmi_pf1000.batch --limit 1
  python -m src.datasets.openbmi_pf1000.batch --subjects 01,02
  python -m src.datasets.openbmi_pf1000.batch
  python -m src.datasets.openbmi_pf1000.batch --merge-only
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
from src.datasets.openbmi_pf1000.pipeline import (
    PROTOCOL,
    preprocess_file_pf1000,
    sanity_check_pf1000,
)

_PREPROCESS_ROOT = Path(__file__).resolve().parents[3]
FULL_KEYS = (
    "X_full",
    "X_mask",
    "y_task",
    "y_three",
    "subjects",
    "trial_id",
    "t0_sec",
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
    return {
        "size": int(st.st_size),
        "sha1_sample": h.hexdigest(),
        "mtime": int(st.st_mtime),
    }


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "protocol": PROTOCOL, "files": {}}
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
        if meta.get("status") == "ok"
        and (shard_dir(out_dir) / fid / "X_full.npy").exists()
    )
    if not fids:
        raise RuntimeError(f"无可用 shard：{out_dir}")

    # 统计 N
    n_total = 0
    shapes = {}
    for fid in fids:
        root = shard_dir(out_dir) / fid
        xf = np.load(root / "X_full.npy", mmap_mode="r")
        n_total += int(xf.shape[0])
        shapes[fid] = xf.shape
    print(f"  合并 N={n_total} from {len(fids)} shards")

    paths = {k: out_dir / f"openbmi_{k}.npy" for k in FULL_KEYS}
    # 探测形状
    sample = np.load(shard_dir(out_dir) / fids[0] / "X_full.npy", mmap_mode="r")
    x_tail = sample.shape[1:]

    mm = {
        "X_full": np.lib.format.open_memmap(
            paths["X_full"], mode="w+", dtype=np.float32, shape=(n_total, *x_tail)
        ),
        "X_mask": np.lib.format.open_memmap(
            paths["X_mask"], mode="w+", dtype=np.float32, shape=(n_total, *x_tail)
        ),
        "y_task": np.lib.format.open_memmap(
            paths["y_task"], mode="w+", dtype=np.int64, shape=(n_total,)
        ),
        "y_three": np.lib.format.open_memmap(
            paths["y_three"], mode="w+", dtype=np.int64, shape=(n_total,)
        ),
        "trial_id": np.lib.format.open_memmap(
            paths["trial_id"], mode="w+", dtype=np.int64, shape=(n_total,)
        ),
        "t0_sec": np.lib.format.open_memmap(
            paths["t0_sec"], mode="w+", dtype=np.float32, shape=(n_total,)
        ),
    }
    subjects = np.empty((n_total,), dtype=object)

    # trial_id 全局连续
    cursor = 0
    tid_offset = 0
    for fid in fids:
        root = shard_dir(out_dir) / fid
        xf = np.load(root / "X_full.npy")
        xm = np.load(root / "X_mask.npy")
        yt = np.load(root / "y_task.npy")
        y3 = np.load(root / "y_three.npy")
        sid = np.load(root / "subjects.npy", allow_pickle=True)
        tid = np.load(root / "trial_id.npy")
        t0 = np.load(root / "t0_sec.npy")
        n = len(yt)
        tid_g = tid + tid_offset
        tid_offset = int(tid_g.max()) + 1 if n else tid_offset
        mm["X_full"][cursor : cursor + n] = xf
        mm["X_mask"][cursor : cursor + n] = xm
        mm["y_task"][cursor : cursor + n] = yt
        mm["y_three"][cursor : cursor + n] = y3
        mm["trial_id"][cursor : cursor + n] = tid_g
        mm["t0_sec"][cursor : cursor + n] = t0
        subjects[cursor : cursor + n] = sid
        cursor += n
        del xf, xm, yt, y3, sid, tid, t0
        gc.collect()

    assert cursor == n_total
    for a in mm.values():
        if hasattr(a, "flush"):
            a.flush()
    np.save(paths["subjects"], subjects)

    # 兼容训练侧：另存一份 openbmi_X.npy = X_full（硬链/复制，避免整表进内存）
    x_alias = out_dir / "openbmi_X.npy"
    if x_alias.exists():
        x_alias.unlink()
    try:
        x_alias.hardlink_to(paths["X_full"])
    except OSError:
        shutil.copyfile(paths["X_full"], x_alias)

    meta = {
        "protocol": PROTOCOL,
        "n_windows": int(n_total),
        "X_full_shape": [n_total, *list(x_tail)],
        "fs_out": 250,
        "geometry": "past100+cur500+future400",
        "post_mi_sec": 1.6,
        "no_rest": True,
        "task": "cue_0_to_4s_plus_post1.6s",
        "source_blocks": ["EEG_MI_train"],
        "excluded_blocks": ["EEG_MI_test"],
        "label_map": {"left": 1, "right": 2},
        "wrote_at": _utc_now(),
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
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    if reset:
        sd = out_dir / "shards"
        if sd.exists():
            shutil.rmtree(sd)
        for p in out_dir.glob("openbmi_*.npy"):
            p.unlink(missing_ok=True)
        for p in (manifest_path, out_dir / "preprocess_meta.json"):
            p.unlink(missing_ok=True)
        print("已重置", out_dir)

    manifest = load_manifest(manifest_path)
    manifest["protocol"] = PROTOCOL
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
        allow_s = {
            s if s.startswith("sess") else f"sess{int(s):02d}" for s in sessions
        }
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
    print(f"[{PROTOCOL}] 候选 {len(files)} 个文件 → {out_dir} zscore={zscore}")

    n_ok = n_skip = n_empty = n_fail = 0
    for fpath in files:
        fid = file_id(fpath)
        fp = file_fingerprint(fpath)
        prev = manifest["files"].get(fid)
        shard_ok = (shard_dir(out_dir) / fid / "X_full.npy").exists()
        if (
            prev
            and prev.get("status") == "ok"
            and prev.get("fingerprint") == fp
            and shard_ok
        ):
            print(f"  skip {fpath.name}")
            n_skip += 1
            continue
        try:
            Xf, Xm, yt, y3, sid, tid, t0, stats = preprocess_file_pf1000(
                fpath, zscore=zscore
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
            continue

        if len(yt) == 0:
            print(f"  empty {fpath.name}")
            manifest["files"][fid] = {
                "status": "empty",
                "fingerprint": fp,
                "stats": stats,
                "time": _utc_now(),
            }
            n_empty += 1
            save_manifest(manifest_path, manifest)
            continue

        try:
            sanity_check_pf1000(Xf, Xm, y3)
        except AssertionError as e:
            print(f"  FAIL sanity {fpath.name}: {e}")
            manifest["files"][fid] = {
                "status": "fail",
                "error": f"sanity: {e}",
                "fingerprint": fp,
                "time": _utc_now(),
            }
            n_fail += 1
            save_manifest(manifest_path, manifest)
            continue

        save_shard(
            out_dir,
            fid,
            {
                "X_full": Xf,
                "X_mask": Xm,
                "y_task": yt,
                "y_three": y3,
                "subjects": sid,
                "trial_id": tid,
                "t0_sec": t0,
            },
        )
        manifest["files"][fid] = {
            "status": "ok",
            "fingerprint": fp,
            "n_windows": int(len(yt)),
            "stats": stats,
            "time": _utc_now(),
        }
        n_ok += 1
        print(
            f"  ok {fpath.name} N={len(yt)} trials={stats.get('n_trials_kept')} "
            f"drop={stats.get('n_trials_dropped')}"
        )
        save_manifest(manifest_path, manifest)
        del Xf, Xm, yt, y3, sid, tid, t0
        gc.collect()

    print(f"done ok={n_ok} skip={n_skip} empty={n_empty} fail={n_fail}")
    if merge and n_ok + n_skip > 0:
        merge_shards(out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="OpenBMI pf1000 shard 预处理")
    p.add_argument(
        "--glob",
        default=str(
            Path("D:/cyy/MI/DATA/openbmi/openbmi/openbmi") / "sess*_subj*_EEG_MI.mat"
        ),
    )
    p.add_argument(
        "--out",
        default=str(_PREPROCESS_ROOT / "out" / "openbmi_2s_hop100_pf1000"),
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--subjects", default="", help="逗号分隔，如 01,02")
    p.add_argument("--sessions", default="", help="逗号分隔，如 01 或 sess01")
    p.add_argument("--reset", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--no-zscore", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    if args.merge_only:
        merge_shards(out)
        return

    subjects = (
        [s.strip() for s in args.subjects.split(",") if s.strip()]
        if args.subjects
        else None
    )
    sessions = (
        [s.strip() for s in args.sessions.split(",") if s.strip()]
        if args.sessions
        else None
    )
    run_batch(
        args.glob,
        out,
        limit=args.limit,
        reset=args.reset,
        merge=True,
        subjects=subjects,
        sessions=sessions,
        zscore=not args.no_zscore,
    )


if __name__ == "__main__":
    main()
