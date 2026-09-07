"""离线汇总最近 v3 会话：模型判定 + ERD 特征 + 信号门禁。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "sessions"
LABEL = {0: "Rest", 1: "Left", 2: "Right"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_session(sess_dir: Path) -> dict | None:
    feats_path = sess_dir / "v3_trial_features.jsonl"
    if not feats_path.is_file():
        return None
    rows = load_jsonl(feats_path)
    if not rows:
        return None

    mi = [r for r in rows if r.get("label") in (1, 2)]
    rest = [r for r in rows if r.get("label") == 0]

    def primary_pred(r: dict) -> int | None:
        pj = r.get("primary_judge") or {}
        if pj.get("signal_bad"):
            return None
        gp = pj.get("gated_pred")
        if gp is None:
            return pj.get("pred")
        return int(gp)

    mi_valid = [r for r in mi if r.get("valid")]
    correct = 0
    pred_counts = {0: 0, 1: 0, 2: 0, None: 0}
    for r in mi:
        p = primary_pred(r)
        pred_counts[p if p in (0, 1, 2) else None] = pred_counts.get(p if p in (0, 1, 2) else None, 0) + 1
        if p == r.get("label"):
            correct += 1

    mu_contras = []
    lateralities = []
    grades = []
    signal_bad_n = 0
    for r in mi:
        f = r.get("features") or {}
        if f.get("mu_erd_contra") is not None:
            mu_contras.append(float(f["mu_erd_contra"]))
        if f.get("laterality_pp") is not None:
            lateralities.append(float(f["laterality_pp"]))
        tg = (f.get("trial_grade") or f.get("grade") or {}).get("grade")
        if tg:
            grades.append(tg)
        if r.get("signal_bad"):
            signal_bad_n += 1

    lat_pass = sum(1 for x in lateralities if x >= 8.0)
    mu_pass = sum(1 for x in mu_contras if x <= -15.0)

    return {
        "session": sess_dir.name,
        "n_trials": len(rows),
        "n_mi": len(mi),
        "n_rest": len(rest),
        "mi_valid": len(mi_valid),
        "model_acc": f"{correct}/{len(mi)}" if mi else "—",
        "pred_left": pred_counts.get(1, 0),
        "pred_right": pred_counts.get(2, 0),
        "pred_rest": pred_counts.get(0, 0),
        "pred_none": pred_counts.get(None, 0),
        "mu_contra_mean": round(sum(mu_contras) / len(mu_contras), 1) if mu_contras else None,
        "mu_contra_pass": f"{mu_pass}/{len(mu_contras)}" if mu_contras else "—",
        "lat_mean": round(sum(lateralities) / len(lateralities), 1) if lateralities else None,
        "lat_pass": f"{lat_pass}/{len(lateralities)}" if lateralities else "—",
        "grades": dict((g, grades.count(g)) for g in sorted(set(grades))),
        "signal_bad_mi": signal_bad_n,
    }


def main() -> None:
    sessions = sorted(ROOT.glob("opsmoke_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    summaries = []
    for s in sessions[:20]:
        sm = summarize_session(s)
        if sm:
            summaries.append(sm)

    if not summaries:
        print("No v3_trial_features.jsonl found in recent opsmoke sessions")
        sys.exit(1)

    print(f"{'session':<35} {'MI':>3} {'acc':>7} {'pL':>3} {'pR':>3} {'pRest':>5} "
          f"{'muERD':>7} {'muOK':>5} {'lat':>6} {'latOK':>5} grades signal_bad")
    print("-" * 120)
    for sm in summaries:
        print(
            f"{sm['session']:<35} {sm['n_mi']:>3} {sm['model_acc']:>7} "
            f"{sm['pred_left']:>3} {sm['pred_right']:>3} {sm['pred_rest']:>5} "
            f"{str(sm['mu_contra_mean']):>7} {sm['mu_contra_pass']:>5} "
            f"{str(sm['lat_mean']):>6} {sm['lat_pass']:>5} "
            f"{sm['grades']} sb={sm['signal_bad_mi']}"
        )


if __name__ == "__main__":
    main()
