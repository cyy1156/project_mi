"""Audit real-subject v3 sessions: count runs + L/R balance + basic files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "subjects"
SUBJECTS = [
    "syj0828",
    "fnz0828",
    "cyy0830",
    "fnz0830",
    "wzr0830",
    "xj0830",
    "cjf0831",
    "npl0831",
    "ycx0831",
]


def session_key(name: str) -> str | None:
    for part in name.split("_"):
        p = part.lower()
        if p.startswith("ws") and p[2:].isdigit():
            return p
        if p.startswith("w") and not p.startswith("ws") and p[1:].isdigit():
            return p
    return None


def phase_of(d: Path) -> str:
    meta = d / "session.meta.json"
    if not meta.is_file():
        return ""
    try:
        return str(json.loads(meta.read_text(encoding="utf-8")).get("phase_mode") or "")
    except Exception:
        return "?"


def check_session(d: Path) -> dict:
    issues: list[str] = []
    eeg = d / "eeg.csv"
    table = d / "alignment" / "trial_table.csv"
    report = d / "v3_report.json"
    events = d / "events.jsonl"
    eeg_mb = 0.0
    if not eeg.is_file():
        issues.append("缺 eeg.csv")
    else:
        eeg_mb = round(eeg.stat().st_size / 1e6, 2)
        if eeg.stat().st_size < 1_000_000:
            issues.append(f"eeg过小({eeg_mb}MB)")
    if not events.is_file():
        issues.append("缺 events.jsonl")
    nL = nR = n_rest = n_rej = n_inv = None
    if table.is_file():
        try:
            df = pd.read_csv(table)
            if "label" in df.columns:
                mi = df[df["label"].isin([1, 2])]
                nL = int((mi["label"] == 1).sum())
                nR = int((mi["label"] == 2).sum())
            if "rejected" in df.columns:
                n_rej = int((df["rejected"] == 1).sum())
            if "invalid" in df.columns:
                n_inv = int((df["invalid"] == 1).sum())
            if "t_rest_start" in df.columns:
                n_rest = int(df["t_rest_start"].notna().sum())
        except Exception as exc:  # noqa: BLE001
            issues.append(f"trial_table读失败:{exc}")
    else:
        issues.append("缺 trial_table.csv")
    return {
        "eeg_mb": eeg_mb,
        "nL": nL,
        "nR": nR,
        "n_rest": n_rest,
        "n_rej": n_rej,
        "n_inv": n_inv,
        "has_report": report.is_file(),
        "phase": phase_of(d),
        "issues": issues,
    }


def expected_keys(sid: str) -> list[str]:
    if sid == "syj0828":
        return [f"ws0{i}" for i in range(1, 7)]
    if sid == "fnz0828":
        return [f"ws0{i}" for i in range(2, 8)]  # historical: ws02-ws07
    if sid == "ycx0831":
        return ["w01", "w02", "w03", "w04", "w05", "w07"]  # w06 半场排除
    return [f"w0{i}" for i in range(1, 7)]


def main() -> None:
    print("真人数据完整性检查（Leave-Next 所用 v3 run）\n")
    for sid in SUBJECTS:
        root = ROOT / sid / "sessions"
        print("=" * 72)
        print(f"SUBJECT {sid}")
        if not root.is_dir():
            print("  !! 无 sessions 目录")
            continue
        all_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name != "_archived"]
        v3_by: dict[str, Path] = {}
        v4: list[Path] = []
        other: list[tuple[Path, str]] = []
        for d in all_dirs:
            ph = phase_of(d)
            key = session_key(d.name)
            if ph == "v4_session" or d.name.startswith("v4_"):
                v4.append(d)
                continue
            if ph and ph != "v3_session":
                other.append((d, ph))
                continue
            if sid == "syj0828" and "124816" in d.name:
                other.append((d, "excluded_old"))
                continue
            if sid == "fnz0828" and d.name.endswith("_152231"):
                other.append((d, "excluded_v4"))
                continue
            if not key:
                other.append((d, "no_key"))
                continue
            prev = v3_by.get(key)
            if prev is None or d.name > prev.name:
                if prev is not None:
                    other.append((prev, "superseded"))
                v3_by[key] = d
            else:
                other.append((d, "older_dup"))

        keys = sorted(
            v3_by.keys(),
            key=lambda k: (
                0 if k.startswith("ws") else 1,
                int("".join(ch for ch in k if ch.isdigit()) or 0),
            ),
        )
        exp = expected_keys(sid)
        missing = [k for k in exp if k not in v3_by]
        extra = [k for k in keys if k not in exp]
        print(f"  Leave-Next v3 runs: {len(keys)} → {keys}")
        print(f"  期望 6 场: {exp}")
        print(f"  缺场: {missing or '无'} | 额外: {extra or '无'} | v4帽检: {len(v4)}")

        full_ok = 0
        for k in keys:
            d = v3_by[k]
            info = check_session(d)
            nL, nR = info["nL"], info["nR"]
            bal = "18:18" if nL == 18 and nR == 18 else f"{nL}:{nR}"
            good = (
                not info["issues"]
                and nL == 18
                and nR == 18
                and (info["n_rest"] or 0) >= 30
            )
            if nL == 18 and nR == 18:
                full_ok += 1
            mark = "OK" if good else ("WARN" if not info["issues"] else "BAD")
            print(
                f"  [{mark}] {k:4} {d.name}\n"
                f"         L:R={bal} rest打点={info['n_rest']} rej={info['n_rej']} "
                f"inv={info['n_inv']} eeg={info['eeg_mb']}MB "
                f"report={info['has_report']} phase={info['phase'] or '(空)'}"
            )
            if info["issues"]:
                print(f"         ISSUES: {', '.join(info['issues'])}")

        print(
            f"  汇总: 完整18:18 = {full_ok}/{len(keys)}; "
            f"{'场次齐全(6)' if not missing else '场次不齐'} "
            f"{'(另有w07)' if 'w07' in keys else ''}"
        )
        if v4:
            print(f"  v4: {[d.name for d in v4]}")


if __name__ == "__main__":
    main()
