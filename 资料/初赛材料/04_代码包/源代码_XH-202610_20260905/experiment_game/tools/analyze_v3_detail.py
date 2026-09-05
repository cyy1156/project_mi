"""详细拆解 v3 MI 试次：左右分开看模型与 ERD。"""
from __future__ import annotations

import json
from pathlib import Path

SESSIONS = [
    "opsmoke_ws01_20260824_223139",
    "opsmoke_ws01_20260824_200329",
    "opsmoke_ws01_20260824_230231",
]
ROOT = Path(__file__).resolve().parents[1] / "data" / "sessions"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def analyze(sess: str) -> None:
    path = ROOT / sess / "v3_trial_features.jsonl"
    if not path.is_file():
        print(f"\n=== {sess}: no features ===")
        return
    rows = load_jsonl(path)
    print(f"\n=== {sess} (n={len(rows)}) ===")

    for r in rows:
        lab = r.get("label")
        if lab not in (1, 2):
            continue
        side = "L" if lab == 1 else "R"
        pj = r.get("primary_judge") or {}
        f = r.get("features") or {}
        tg = (f.get("trial_grade") or {}).get("grade", "?")
        mu = f.get("mu_erd_contra")
        lat = f.get("laterality_pp")
        p3 = pj.get("p_three")
        gated = pj.get("gated")
        gp = pj.get("gated_pred")
        sb = r.get("signal_bad") or pj.get("signal_bad")
        p3s = ""
        if p3:
            p3s = f"Rest={p3[0]:.2f} L={p3[1]:.2f} R={p3[2]:.3f}"
        mu_s = f"{mu:+.0f}%" if mu is not None else "n/a"
        lat_s = f"{lat:+.1f}pp" if lat is not None else "n/a"
        lat_ok = lat is not None and lat >= 8
        mu_ok = mu is not None and mu <= -15
        print(
            f"  T{r['trial_id']:02d} {side} grade={tg} mu={mu_s}({'OK' if mu_ok else 'X'}) "
            f"lat={lat_s}({'OK' if lat_ok else 'X'}) gated={gated} pred={gp} sb={sb} {p3s}"
        )

    # block summary from events
    ev = ROOT / sess / "events.jsonl"
    if ev.is_file():
        for line in ev.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") in ("v3_baseline", "v3_hat_check", "v3_block_end"):
                print(f"  event: {e.get('type')} -> {json.dumps(e, ensure_ascii=False)[:200]}")


for s in SESSIONS:
    analyze(s)
