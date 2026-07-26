"""Stieger 增量批处理：跳过已处理 → 追加 npy → 写清单/日志 → 可选删原文件。"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.datasets.stieger.pipeline import preprocess_session, sanity_check_outputs
from src.common.steps.split_subjects import split_all_trials

FULL_KEYS = ("X", "y_task", "y_three", "subjects")

# batch.py → stieger → datasets → src → preprocess_lab
_PREPROCESS_ROOT = Path(__file__).resolve().parents[3]
# preprocess_lab → code → MI
_REPO_ROOT = Path(__file__).resolve().parents[5]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_id(path: Path) -> str:
    """清单主键：会话文件名（不含扩展名）。"""
    return path.stem  # S1_Session_1


def file_fingerprint(path: Path) -> dict:
    """辅助校验：大小 + 快速采样 hash（整文件 sha 太慢时可只用 size）。"""
    st = path.stat()
    h = hashlib.sha1()
    with path.open("rb") as f:
        # 只读头尾各 1MB，大文件也快；若需绝对严谨可改全文 sha1
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
        return {"version": 1, "files": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _npy_path(out_dir: Path, kind: str, split: str | None = None) -> Path:
    if split is None:
        return out_dir / f"stieger_{kind}.npy"
    return out_dir / f"{split}_{kind}.npy"


def load_existing_full(out_dir: Path) -> dict[str, np.ndarray] | None:
    paths = {k: _npy_path(out_dir, k) for k in FULL_KEYS}
    if not all(p.exists() for p in paths.values()):
        return None
    return {k: np.load(p, allow_pickle=(k == "subjects")) for k, p in paths.items()}


def save_full(out_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, arr in arrays.items():
        np.save(_npy_path(out_dir, k), arr)


def concat_or_new(
    old: dict[str, np.ndarray] | None,
    new: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if old is None:
        return new
    return {k: np.concatenate([old[k], new[k]], axis=0) for k in FULL_KEYS}


def append_split_arrays(
    out_dir: Path,
    split: str,
    X,
    y_task,
    y_three,
    subjects,
) -> None:
    """把本批划分结果追加到已有 train_/val_；没有则新建。"""
    keys = ("X", "y_task", "y_three", "subjects")
    new = {
        "X": X,
        "y_task": y_task,
        "y_three": y_three,
        "subjects": subjects,
    }
    for k in keys:
        p = _npy_path(out_dir, k, split=split)
        if p.exists():
            old = np.load(p, allow_pickle=(k == "subjects"))
            arr = np.concatenate([old, new[k]], axis=0)
        else:
            arr = new[k]
        np.save(p, arr)


def run_incremental_batch(
    data_glob: str,
    out_dir: Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    skip_if_processed: bool = True,
    delete_raw_after_ok: bool = False,
    rebuild_split: bool = False,
    manifest_name: str = "processed_manifest.json",
    batch_log_name: str = "batch_log.jsonl",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / manifest_name
    log_path = out_dir / batch_log_name
    manifest = load_manifest(manifest_path)

    files = sorted(Path(p) for p in glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"未匹配到文件: {data_glob}")

    batch_id = _utc_now()
    n_skip = n_ok = n_fail = n_empty = 0
    new_Xs, new_yts, new_y3s, new_sids = [], [], [], []

    print(f"[batch {batch_id}] 扫描到 {len(files)} 个文件；清单已有 {len(manifest['files'])} 条")

    for fp in files:
        fid = file_id(fp)
        fp_info = file_fingerprint(fp)

        # ---- 防重复：清单命中则跳过 ----
        if skip_if_processed and fid in manifest["files"]:
            prev = manifest["files"][fid]
            # 同名但大小变了：警告并仍跳过（避免静默污染）；要重跑请先从清单删除该键
            if prev.get("size") and prev["size"] != fp_info["size"]:
                print(
                    f"  WARN skip {fp.name}: 清单已有但 size 不同 "
                    f"prev={prev.get('size')} now={fp_info['size']}"
                )
            else:
                print(f"  skip {fp.name} (already in manifest)")
            n_skip += 1
            append_log(
                log_path,
                {
                    "time": _utc_now(),
                    "batch_id": batch_id,
                    "file": fp.name,
                    "file_id": fid,
                    "status": "skip_duplicate",
                },
            )
            continue

        try:
            X, yt, y3, sid, stats = preprocess_session(fp)
        except Exception as e:
            n_fail += 1
            print(f"  FAIL {fp.name}: {e}")
            append_log(
                log_path,
                {
                    "time": _utc_now(),
                    "batch_id": batch_id,
                    "file": fp.name,
                    "file_id": fid,
                    "status": "fail",
                    "error": str(e),
                },
            )
            continue

        if len(yt) == 0:
            n_empty += 1
            # 仍写入清单，避免空会话被反复重试
            manifest["files"][fid] = {
                **fp_info,
                "n_trials": 0,
                "processed_at": _utc_now(),
                "batch_id": batch_id,
                "stats": stats,
            }
            save_manifest(manifest_path, manifest)
            append_log(
                log_path,
                {
                    "time": _utc_now(),
                    "batch_id": batch_id,
                    "file": fp.name,
                    "file_id": fid,
                    "status": "ok_empty",
                    "stats": stats,
                },
            )
            print(f"  empty {fp.name}", stats)
            if delete_raw_after_ok:
                fp.unlink(missing_ok=True)
            continue

        # 先缓存本批，最后统一 concat 落盘（减少反复读写大 npy）
        new_Xs.append(X)
        new_yts.append(yt)
        new_y3s.append(y3)
        new_sids.append(sid)

        manifest["files"][fid] = {
            **fp_info,
            "n_trials": int(len(yt)),
            "n_task": int(np.sum(yt == 1)),
            "n_rest": int(np.sum(yt == 0)),
            "subject": str(sid[0]) if len(sid) else "",
            "processed_at": _utc_now(),
            "batch_id": batch_id,
            "stats": stats,
        }
        # 每成功一个文件就存清单，中断也不丢「已处理」记录
        save_manifest(manifest_path, manifest)
        append_log(
            log_path,
            {
                "time": _utc_now(),
                "batch_id": batch_id,
                "file": fp.name,
                "file_id": fid,
                "status": "ok",
                "n_trials": int(len(yt)),
                "stats": stats,
            },
        )
        n_ok += 1
        print(f"  ok {fp.name} n={len(yt)}", stats)

        if delete_raw_after_ok:
            fp.unlink(missing_ok=True)
            print(f"    deleted raw {fp.name}")

    if not new_Xs:
        print(f"本批无新样本可追加。skip={n_skip} fail={n_fail} empty={n_empty}")
        return

    new = {
        "X": np.concatenate(new_Xs, axis=0),
        "y_task": np.concatenate(new_yts, axis=0),
        "y_three": np.concatenate(new_y3s, axis=0),
        "subjects": np.concatenate(new_sids, axis=0),
    }
    sanity_check_outputs(new["X"], new["y_task"], new["y_three"])

    old = load_existing_full(out_dir)
    merged = concat_or_new(old, new)
    save_full(out_dir, merged)
    print(
        f"全量已保存: N={len(merged['X'])}"
        f"（本批 +{len(new['X'])}；此前 {0 if old is None else len(old['X'])}）"
    )

    # ---- train/val ----
    if rebuild_split:
        parts = split_all_trials(
            merged["X"],
            merged["y_task"],
            merged["y_three"],
            val_ratio=val_ratio,
            seed=seed,
            subjects=merged["subjects"],
        )
        for split in ("train", "val"):
            Xs_, yt_, y3_, sid_ = parts[split]
            np.save(_npy_path(out_dir, "X", split), Xs_)
            np.save(_npy_path(out_dir, "y_task", split), yt_)
            np.save(_npy_path(out_dir, "y_three", split), y3_)
            np.save(_npy_path(out_dir, "subjects", split), sid_)
        print("已按全量重建 train/val")
    else:
        parts = split_all_trials(
            new["X"],
            new["y_task"],
            new["y_three"],
            val_ratio=val_ratio,
            seed=seed,
            subjects=new["subjects"],
        )
        for split in ("train", "val"):
            Xs_, yt_, y3_, sid_ = parts[split]
            append_split_arrays(out_dir, split, Xs_, yt_, y3_, sid_)
        print("已将本批新样本追加到 train/val（未打乱旧样本）")

    append_log(
        log_path,
        {
            "time": _utc_now(),
            "batch_id": batch_id,
            "status": "batch_done",
            "n_ok": n_ok,
            "n_skip": n_skip,
            "n_fail": n_fail,
            "n_empty": n_empty,
            "n_full": int(len(merged["X"])),
            "n_batch_new": int(len(new["X"])),
        },
    )
    print(f"done ok={n_ok} skip={n_skip} fail={n_fail} empty={n_empty} → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stieger 增量预处理批处理")
    parser.add_argument(
        "--glob",
        default=str(_REPO_ROOT / "DATA" / "stieger" / "S*_Session_*.mat"),
        help="本批原始 mat 通配符",
    )
    parser.add_argument(
        "--out",
        default=str(_PREPROCESS_ROOT / "out" / "stieger_2s"),
        help="输出目录（现行 2s/500 点）",
    )
    parser.add_argument("--delete-raw", action="store_true", help="成功后删除原始 mat")
    parser.add_argument(
        "--rebuild-split", action="store_true", help="用全量重划 train/val"
    )
    args = parser.parse_args()

    run_incremental_batch(
        data_glob=args.glob,
        out_dir=Path(args.out),
        delete_raw_after_ok=args.delete_raw,
        rebuild_split=args.rebuild_split,
    )


if __name__ == "__main__":
    main()
