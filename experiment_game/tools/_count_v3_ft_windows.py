#!/usr/bin/env python3
"""统计 syj0828 / fnz0828 的 v3 会话按 openbmi_align 可微调窗数。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "code" / "preprocess_lab"))

from experiment_game.tools.ft_subject_from_v3 import (  # noqa: E402
    PROTOCOL_OPENBMI_ALIGN,
    _build_session_windows,
    detect_session_protocol,
)
from experiment_game.offline.openbmi_align_cut import n_windows_3s_hop100  # noqa: E402

SUBJECTS = ["syj0828", "fnz0828"]
ROOT = _REPO / "experiment_game" / "data" / "subjects"


def _phase_mode(session: Path) -> str:
    rc = session / "run_config.json"
    if not rc.is_file():
        return "?"
    blob = json.loads(rc.read_text(encoding="utf-8"))

    def dig(o):
        if isinstance(o, dict):
            if "phase_mode" in o:
                return str(o["phase_mode"])
            for v in o.values():
                r = dig(v)
                if r:
                    return r
        return None

    return dig(blob) or "?"


def _table_stats(session: Path) -> dict:
    table = session / "alignment" / "trial_table.csv"
    if not table.is_file():
        return {"error": "no trial_table"}
    df = pd.read_csv(table)
    labs = Counter(int(x) for x in df["label"].tolist() if int(x) in (0, 1, 2))
    n_rest_mark = int(
        df["t_rest_start"].notna().sum()
        if "t_rest_start" in df.columns
        else 0
    )
    # rest segments long enough for 3s
    n_rest_ok = 0
    if "t_rest_start" in df.columns and "t_rest_end" in df.columns:
        for _, r in df.iterrows():
            ts, te = r.get("t_rest_start"), r.get("t_rest_end")
            if pd.isna(ts) or pd.isna(te):
                continue
            if float(te) - float(ts) >= 3.0 - 1e-6:
                n_rest_ok += 1
    n_rej = int((df.get("rejected", 0) == 1).sum()) if "rejected" in df.columns else 0
    n_inv = int((df.get("invalid", 0) == 1).sum()) if "invalid" in df.columns else 0
    return {
        "n_rows": len(df),
        "label_counts": dict(labs),
        "n_rest_marked": n_rest_mark,
        "n_rest_ge3s": n_rest_ok,
        "n_rejected": n_rej,
        "n_invalid": n_inv,
        "theo_wins_per_4s": int(n_windows_3s_hop100(4.0)),
    }


def main() -> int:
    print(f"theo windows / 4s segment (3s hop100) = {n_windows_3s_hop100(4.0)}")
    print("FT path: openbmi_align · L/R from Cue+0..4s; Rest from t_rest_start/end")
    print("Rest cap in ft_subject_from_v3: min(n_left, n_right) when no label=0 trials\n")

    grand = Counter()
    for sid in SUBJECTS:
        sroot = ROOT / sid / "sessions"
        if not sroot.is_dir():
            print(f"{sid}: missing")
            continue
        print("=" * 72)
        print(sid)
        sub_tot = Counter()
        n_sess = 0
        for sp in sorted(sroot.iterdir()):
            if not sp.is_dir() or sp.name.startswith("_"):
                continue
            mode = _phase_mode(sp)
            if mode != "v3_session":
                print(f"  SKIP {sp.name} phase_mode={mode}")
                continue
            n_sess += 1
            ts = _table_stats(sp)
            try:
                proto = detect_session_protocol(sp)
            except Exception as e:
                proto = f"err:{e}"
            try:
                # include_invalid=True 与默认 FT 一致；再报 exclude 对照
                ds_all = _build_session_windows(
                    sp, include_invalid=True, protocol=PROTOCOL_OPENBMI_ALIGN
                )
                y = ds_all["y_three"]
                c_all = Counter(int(x) for x in y.tolist())
                n_trials = int(ds_all.get("n_trials") or 0)
                ds_ex = _build_session_windows(
                    sp, include_invalid=False, protocol=PROTOCOL_OPENBMI_ALIGN
                )
                c_ex = Counter(int(x) for x in ds_ex["y_three"].tolist())
            except Exception as e:
                print(f"  FAIL {sp.name}: {e}")
                continue
            sub_tot.update(c_all)
            grand.update(c_all)
            print(
                f"  {sp.name}\n"
                f"    protocol_detect={proto} table={ts}\n"
                f"    FT include_invalid: Rest={c_all.get(0,0)} L={c_all.get(1,0)} "
                f"R={c_all.get(2,0)} total={len(y)} "
                f"(task_trials_used≈{n_trials})\n"
                f"    FT exclude_invalid: Rest={c_ex.get(0,0)} L={c_ex.get(1,0)} "
                f"R={c_ex.get(2,0)} total={len(ds_ex['y_three'])}"
            )
        print(
            f"  SUBJECT TOTAL ({n_sess} v3): "
            f"Rest={sub_tot[0]} L={sub_tot[1]} R={sub_tot[2]} "
            f"all={sum(sub_tot.values())}"
        )
        if n_sess:
            print(
                f"  per-session mean: Rest={sub_tot[0]/n_sess:.0f} "
                f"L={sub_tot[1]/n_sess:.0f} R={sub_tot[2]/n_sess:.0f} "
                f"all={sum(sub_tot.values())/n_sess:.0f}"
            )
    print("=" * 72)
    print(
        f"GRAND: Rest={grand[0]} L={grand[1]} R={grand[2]} all={sum(grand.values())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
