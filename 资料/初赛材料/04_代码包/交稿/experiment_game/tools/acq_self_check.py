#!/usr/bin/env python3
"""采集模块一键自检（合成板）：依赖 → 启停 → health_check → 质量报告。"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_game.acquisition import AcquisitionFacade
from experiment_game.experiment.alignment import write_alignment_bundle

CORE = ("numpy", "scipy", "websockets")
ACQ = ("brainflow", "pylsl", "yaml")


def _check_import(name: str) -> tuple[bool, str]:
    mod = "yaml" if name == "yaml" else name
    if importlib.util.find_spec(mod) is None:
        return False, "MISSING"
    try:
        m = importlib.import_module(mod)
        return True, str(getattr(m, "__version__", "?"))
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def main() -> int:
    failures: list[str] = []

    print("=== acquisition self-check (synthetic) ===")
    for pkg in CORE + ACQ:
        ok, info = _check_import(pkg)
        label = "PyYAML" if pkg == "yaml" else pkg
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: {info}")
        if not ok:
            failures.append(f"import:{label}")

    with tempfile.TemporaryDirectory(prefix="acq_self_") as tmp:
        csv_path = Path(tmp) / "eeg.csv"
        acq = AcquisitionFacade(use_synthetic=True)
        try:
            acq.create()
            acq.start(csv_path)
            hc = acq.health_check()
            print(
                f"  health_check: delta={hc['delta_samples']} "
                f"lsl={hc['lsl_detail']}"
            )
            time.sleep(5.0)
            report = acq.stop()
            quality = report.get("quality") or {}
            if not quality:
                failures.append("stop_quality_missing")
            else:
                drop = float(quality.get("drop_rate_pct", 0.0))
                lsl_ok = bool(quality.get("lsl_timeline_ok"))
                print(
                    f"  quality: drop_rate_pct={drop} "
                    f"lsl_timeline_ok={lsl_ok} severity={quality.get('severity')}"
                )
                if not lsl_ok and drop > 1.0:
                    failures.append(f"drop_rate_pct={drop}")
            acq.shutdown()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"facade:{exc}")
            try:
                acq.shutdown()
            except Exception:  # noqa: BLE001
                pass

        if not csv_path.is_file():
            failures.append("eeg_csv_missing")
        else:
            rows = sum(1 for _ in csv_path.open(encoding="utf-8")) - 1
            print(f"  csv_rows: {rows}")
            if rows < 500:
                failures.append(f"csv_rows={rows}")

        meta_path = csv_path.with_suffix(".meta.json")
        if not meta_path.is_file():
            failures.append("eeg_meta_missing")

        # alignment bundle with empty events（仅质量校验路径）
        session_root = Path(tmp) / "session"
        session_root.mkdir(parents=True, exist_ok=True)
        (session_root / "events.jsonl").write_text("", encoding="utf-8")
        import shutil

        shutil.copy2(csv_path, session_root / "eeg.csv")
        if meta_path.is_file():
            shutil.copy2(meta_path, session_root / "eeg.meta.json")
        verify = write_alignment_bundle(session_root, acq_enabled=True)
        names = {c["name"] for c in verify.get("checks", [])}
        for need in (
            "eeg_quality_present",
            "eeg_drop_rate_ok",
            "eeg_lsl_timeline_ok",
        ):
            if need not in names:
                failures.append(f"verify_check:{need}")
        if not verify.get("passed"):
            failures.append(f"verify_failed:{verify.get('errors')}")

    if failures:
        print("ACQ_SELF_CHECK_FAIL", failures)
        return 1
    print("ACQ_SELF_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
