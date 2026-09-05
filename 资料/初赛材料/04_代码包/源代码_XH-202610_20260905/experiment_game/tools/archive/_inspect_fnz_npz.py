"""检查 fnz 在线 BCI npz 数据。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FILES = [
    Path(r"d:\Wechat\Documents\xwechat_files\wxid_qtkpx1yy9qxr22_150b\msg\file\2026-08\online_bci_20260826_162523.raw.npz"),
    Path(r"d:\Wechat\Documents\xwechat_files\wxid_qtkpx1yy9qxr22_150b\msg\file\2026-08\online_bci_20260826_162653.raw.cue_segments_raw.npz"),
]


def summarize_array(name: str, v: np.ndarray) -> None:
    print(f"  {name}: dtype={v.dtype} shape={v.shape}")
    if v.dtype == object or v.shape == ():
        try:
            item = v.item() if v.shape == () else v
            s = repr(item)
            print(f"    -> {s[:500]}")
        except Exception as e:
            print(f"    -> (object) {e}")
        return
    if v.dtype.names:
        print(f"    structured fields: {v.dtype.names}")
        if len(v) <= 5:
            print(f"    rows: {v}")
        return
    if v.size <= 24:
        print(f"    data: {v}")
        return
    if np.issubdtype(v.dtype, np.number):
        flat = v.astype(np.float64).ravel()
        print(
            f"    min={flat.min():.4g} max={flat.max():.4g} "
            f"mean={flat.mean():.4g} std={flat.std():.4g}"
        )
    if name.lower() in ("ch_names", "channel_names", "labels", "events", "markers", "meta"):
        print(f"    head: {v[:min(20, len(v))]}")


def inspect_npz(path: Path) -> dict:
    print("=" * 72)
    print(path.name)
    if not path.is_file():
        print("  FILE NOT FOUND")
        return {}
    d = np.load(path, allow_pickle=True)
    info = {"file": str(path), "keys": list(d.keys())}
    for k in d.keys():
        v = d[k]
        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        summarize_array(k, v)
    d.close()
    return info


def compare_with_sessions():
    """若字段可识别，尝试与 fnz_ws01/ws02 对齐。"""
    from datetime import datetime

    # 162523 / 162653 -> 16:25:23 / 16:26:53 on 2026-08-26
    sessions = [
        Path(r"d:\MI\experiment_game\data\sessions\fnz_ws01_20260826_164149"),
        Path(r"d:\MI\experiment_game\data\sessions\fnz_ws02_20260826_171537"),
    ]
    print("\n" + "=" * 72)
    print("MI project fnz sessions (for time reference)")
    for s in sessions:
        if s.is_dir():
            meta = s / "session.meta.json"
            if meta.is_file():
                m = json.loads(meta.read_text(encoding="utf-8"))
                print(f"  {s.name}: started={m.get('started_at')} subject={m.get('subject_id')}")


def deep_dive_cue_segments(path: Path):
    print("\n" + "=" * 72)
    print(f"Deep dive: {path.name}")
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    print("keys:", keys)

    # common patterns
    for cand in ("segments", "X", "data", "eeg", "windows", "trials"):
        if cand in keys:
            x = d[cand]
            print(f"\n[{cand}] shape={x.shape} dtype={x.dtype}")
            if x.ndim >= 2:
                print(f"  sample shape[1:]: {x.shape[1:]}")

    for cand in ("labels", "y", "label", "cue_labels", "trial_labels"):
        if cand in keys:
            y = d[cand]
            uniq, cnt = np.unique(y, return_counts=True)
            print(f"\n[{cand}] unique={dict(zip(uniq.tolist(), cnt.tolist()))}")

    for cand in ("cue_times", "t_cue", "timestamps", "lsl_time", "times"):
        if cand in keys:
            t = d[cand]
            print(f"\n[{cand}] len={len(t)} first5={t[:5]} last5={t[-5:]}")

    for cand in ("ch_names", "channel_names", "channels"):
        if cand in keys:
            print(f"\n[{cand}] {d[cand]}")

    for cand in ("meta", "info", "attrs"):
        if cand in keys:
            m = d[cand]
            try:
                m = m.item() if m.shape == () else m
                print(f"\n[{cand}] {m}")
            except Exception:
                print(f"\n[{cand}] {m}")

    d.close()


def deep_dive_raw(path: Path):
    print("\n" + "=" * 72)
    print(f"Deep dive: {path.name}")
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    print("keys:", keys)

    for cand in ("data", "eeg", "X", "raw"):
        if cand in keys:
            x = d[cand]
            print(f"\n[{cand}] shape={x.shape} dtype={x.dtype}")
            if x.ndim == 2:
                print(f"  duration_s ~ {x.shape[0]/250:.1f} @250Hz" if x.shape[0] > 100 else "")

    for cand in ("timestamps", "lsl_time", "times", "ts"):
        if cand in keys:
            t = d[cand]
            print(f"\n[{cand}] len={len(t)} span={float(t[-1]-t[0]):.1f}s")

    for cand in ("markers", "events", "cue_events"):
        if cand in keys:
            ev = d[cand]
            print(f"\n[{cand}] type={type(ev)} shape={getattr(ev,'shape',None)}")
            if isinstance(ev, np.ndarray) and ev.dtype.names:
                print(f"  fields: {ev.dtype.names}")
                print(f"  first rows:\n{ev[:min(10,len(ev))]}")
            elif isinstance(ev, np.ndarray) and ev.ndim <= 2 and len(ev) <= 30:
                print(ev)
            else:
                try:
                    item = ev.item() if ev.shape == () else ev[:10]
                    print(f"  sample: {item}")
                except Exception:
                    print(f"  head: {ev[:10]}")

    for cand in ("ch_names", "channel_names", "sfreq", "fs"):
        if cand in keys:
            print(f"\n[{cand}] {d[cand]}")

    d.close()


if __name__ == "__main__":
    for f in FILES:
        inspect_npz(f)
    compare_with_sessions()
    deep_dive_raw(FILES[0])
    deep_dive_cue_segments(FILES[1])
