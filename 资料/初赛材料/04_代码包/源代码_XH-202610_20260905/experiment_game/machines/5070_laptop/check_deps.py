"""Check experiment_game deps — thin wrapper over shared Preflight.

Machine: 5070_laptop. Implementation: ``python -m experiment_game.tools.preflight``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiment_game.tools.preflight import format_report, main, run_preflight  # noqa: E402


def run() -> int:
    report = run_preflight()
    print(format_report(report))
    print()
    print("(shared module: experiment_game.tools.preflight)")
    return 0 if report.ok else 1


if __name__ == "__main__":
    # allow --json etc.
    if len(sys.argv) > 1:
        raise SystemExit(main())
    raise SystemExit(run())
