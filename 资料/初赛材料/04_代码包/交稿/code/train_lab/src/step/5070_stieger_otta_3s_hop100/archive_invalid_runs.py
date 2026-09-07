"""将 results/S09-* 下非 v1.2 或未完成 run 移入 archive。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from config import PROTOCOL_VERSION, RESULTS_ROOT

ARCHIVE_ROOT = RESULTS_ROOT / "archive_v1.0_v1.1"
STOPPED_ROOT = RESULTS_ROOT / "archive_stopped"


def _is_v12(summary_path: Path) -> bool:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    spec = data.get("spec") or {}
    return spec.get("protocol_version") == PROTOCOL_VERSION


def _n_subjects(summary_path: Path) -> int:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    return len(data.get("subjects") or {})


def archive_run(run_dir: Path, dest_root: Path) -> None:
    arm_dir = run_dir.parent.name
    dest = dest_root / arm_dir / run_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(dest))
    arm_parent = run_dir.parent.parent
    stamp = run_dir.name.split("_", 1)[0] if "_" in run_dir.name else ""
    if stamp:
        for md in arm_parent.glob(f"{stamp}_*.md"):
            dest_md = dest_root / arm_dir / md.name
            if md.is_file() and not dest_md.exists():
                shutil.move(str(md), str(dest_md))


def _partial_run_dir(run_dir: Path) -> bool:
    return any(run_dir.glob("S*_summary.json"))


def main() -> None:
    moved_invalid = 0
    moved_stopped = 0
    for arm_dir in sorted(RESULTS_ROOT.glob("S09-*")):
        if not arm_dir.is_dir() or arm_dir.name.startswith("archive"):
            continue
        for run_dir in sorted(arm_dir.glob("*")):
            if not run_dir.is_dir():
                continue
            summary = run_dir / "summary.json"
            if not summary.is_file():
                if _partial_run_dir(run_dir):
                    archive_run(run_dir, STOPPED_ROOT)
                    moved_stopped += 1
                    print(f"[partial] {run_dir.name} -> archive_stopped")
                continue
            if not _is_v12(summary):
                archive_run(run_dir, ARCHIVE_ROOT)
                moved_invalid += 1
                print(f"[invalid] {run_dir.name} -> archive_v1.0_v1.1")
                continue
            # v1.2 but incomplete full run (smoke / interrupted)
            n = _n_subjects(summary)
            arm = arm_dir.name.replace("S09-", "")
            if arm in {"A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "B4"} and n < 24:
                archive_run(run_dir, ARCHIVE_ROOT / "v1.2_smoke")
                moved_invalid += 1
                print(f"[smoke] {run_dir.name} ({n} subj) -> archive_v1.0_v1.1/v1.2_smoke")
            elif arm == "C1" and n < 24:
                archive_run(run_dir, STOPPED_ROOT)
                moved_stopped += 1
                print(f"[stopped] {run_dir.name} ({n} subj) -> archive_stopped")

    # orphan md next to arm dirs (run folder already archived)
    for arm_dir in sorted(RESULTS_ROOT.glob("S09-*")):
        if not arm_dir.is_dir() or arm_dir.name.startswith("archive"):
            continue
        for md in arm_dir.glob("*.md"):
            if not (arm_dir / md.stem).is_dir():
                orphan = ARCHIVE_ROOT / "orphan_md" / arm_dir.name / md.name
                orphan.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(md), str(orphan))
                print(f"[orphan md] {md.name} -> archive_v1.0_v1.1/orphan_md")

    print(
        f"done: invalid/smoke={moved_invalid}, stopped_c1={moved_stopped}",
        flush=True,
    )


if __name__ == "__main__":
    import _bootstrap  # noqa: F401

    main()
