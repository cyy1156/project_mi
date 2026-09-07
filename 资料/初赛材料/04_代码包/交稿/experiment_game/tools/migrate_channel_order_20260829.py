# -*- coding: utf-8 -*-
"""一次性迁移脚本：把 syj0828 / xjh0828 的会话数据从旧设备序重排为新通道序。
新序: FC3, C3, CP3, CZ, CPZ, FC4, C4, CP4
处理对象: eeg.csv, continuous/eeg.csv, v3_segments/trial*.npy, eeg.meta.json(channel_labels)
先全量备份到 _backup_old_channel_order_20260829/，再原子写（tmp + os.replace）。
"""
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\MI\experiment_game\data\subjects")
BACKUP = ROOT / "_backup_old_channel_order_20260829"
NEW_ORDER = ["FC3", "C3", "CP3", "CZ", "CPZ", "FC4", "C4", "CP4"]
NORM = {"CZ": "CZ", "C3": "C3", "C4": "C4", "CP3": "CP3", "CP4": "CP4",
        "CPZ": "CPZ", "FC3": "FC3", "FC4": "FC4",
        "Cz": "CZ", "CPz": "CPZ"}  # 旧命名别名 -> 规范大写

subjects = ["syj0828", "xjh0828"]
report = []
differences = []

def norm(name: str) -> str:
    n = name.strip()
    if n in NORM:
        return NORM[n]
    raise KeyError(f"未知通道名: {name!r}")

def perm_from_header(header_chans):
    """返回 perm: new[i] = old[perm[i]]（针对 8 个数据列，不含 lsl_time）。"""
    old_norm = [norm(c) for c in header_chans]
    if sorted(old_norm) != sorted(NEW_ORDER):
        differences.append((header_chans, old_norm))
        return None
    return [old_norm.index(c) for c in NEW_ORDER]

def backup(src: Path) -> Path:
    rel = src.relative_to(ROOT)
    dst = BACKUP / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst

def atomic_replace_csv(src: Path, perm_cols) -> dict:
    """perm_cols 是相对整行（含 lsl_time 列 0）的完整列重排。流式重写。"""
    before_lines = 0
    with open(src, "r", encoding="utf-8", newline="") as f:
        for _ in f:
            before_lines += 1
    fd, tmp = tempfile.mkstemp(dir=str(src.parent), suffix=".tmp")
    after_lines = 0
    with open(src, "r", encoding="utf-8", newline="") as fin, \
         os.fdopen(fd, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout, lineterminator="\n")
        for row in reader:
            writer.writerow([row[i] for i in perm_cols])
            after_lines += 1
    os.replace(tmp, src)
    return {"lines_before": before_lines, "lines_after": after_lines}

def rewrite_csv(path: Path, stats: dict):
    with open(path, "r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    if header and header[0].strip() == "lsl_time":
        chans, offset = header[1:], 1
    else:
        chans, offset = header, 0
    perm8 = perm_from_header(chans)
    if perm8 is None:
        stats["skipped"] = True
        return
    perm_full = list(range(offset)) + [i + offset for i in perm8]
    backup(path)
    r = atomic_replace_csv(path, perm_full)
    if r["lines_before"] != r["lines_after"]:
        raise RuntimeError(f"行数不一致 {path}: {r}")
    stats["csv"] = r["lines_after"]

def rewrite_npys(segs_dir: Path, stats: dict):
    for npy in sorted(segs_dir.glob("trial*.npy")):
        a = np.load(npy)
        if a.ndim != 2 or a.shape[1] != 8:
            differences.append((str(npy), f"shape={a.shape} 非 (T,8)，跳过"))
            stats["npy_skipped"] += 1
            continue
        backup(npy)
        # 旧 npy 通道轴按旧设备序 [C3,C4,CZ,CP3,CP4,CPZ,FC3,FC4] 存
        old_dev = ["C3", "C4", "CZ", "CP3", "CP4", "CPZ", "FC3", "FC4"]
        perm = [old_dev.index(c) for c in NEW_ORDER]
        np.save(npy, a[:, perm].astype(a.dtype), allow_pickle=False)
        stats["npy"] += 1

def rewrite_meta(path: Path, stats: dict):
    d = json.loads(path.read_text(encoding="utf-8"))
    if "channel_labels" in d:
        old = d["channel_labels"]
        perm8 = perm_from_header(old)
        if perm8 is None:
            stats["meta_skipped"] = True
            return
        backup(path)
        d["channel_labels"] = [old[i] for i in perm8]
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        stats["meta"] = True

for subj in subjects:
    for sess in sorted((ROOT / subj / "sessions").glob("*")):
        if not sess.is_dir():
            continue
        stats = {"sess": f"{subj}/{sess.name}", "csv": 0, "npy": 0}
        try:
            eeg = sess / "eeg.csv"
            if eeg.exists():
                rewrite_csv(eeg, stats)
            cont = sess / "continuous" / "eeg.csv"
            if cont.exists():
                rewrite_csv(cont, stats)
            segs = sess / "v3_segments"
            if segs.is_dir():
                rewrite_npys(segs, stats)
            meta = sess / "eeg.meta.json"
            if meta.exists():
                rewrite_meta(meta, stats)
            stats["ok"] = True
        except Exception as e:  # noqa: BLE001
            stats["ok"] = False
            stats["err"] = repr(e)
        report.append(stats)

print("=" * 72)
for s in report:
    flag = "OK " if s.get("ok") else "ERR"
    extra = s.get("err", "")
    print(f"[{flag}] {s['sess']:52s} csv={s['csv']:>6}行 npy={s['npy']:>3}{extra}")
print("=" * 72)
if differences:
    print("!!! 差异清单（未处理）:")
    for d in differences:
        print("  ", d)
else:
    print("通道数量/命名差异: 无（全部 8 通道，命名集合一致，仅顺序不同）")
print(f"备份目录: {BACKUP}")
