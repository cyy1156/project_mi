"""方案 28 · prob dump 存在性校验（5090 · 不训练）。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from member_runs import member_run_dirs  # noqa: E402
from prob_io import load_run_three  # noqa: E402
from s28_config import ARM_MEMBERS  # noqa: E402


def _needed_members() -> tuple[str, ...]:
    names: set[str] = set()
    for pool in ARM_MEMBERS.values():
        names.update(pool)
    return tuple(sorted(names))


def main() -> None:
    names = list(_needed_members())
    run_dirs = member_run_dirs(names)
    for name, run_dir in zip(names, run_dirs):
        assert run_dir.is_dir(), f"missing run dir for {name}: {run_dir}"
        data = load_run_three(run_dir)
        n_val = int((data["split"] == "val").sum())
        n_test = int((data["split"] == "test").sum())
        assert n_val > 0 and n_test > 0, f"empty split under {run_dir}"
        print(f"OK {name}: {run_dir} val={n_val} test={n_test}")
    print(f"verify_r28_dumps: OK members={names}")


if __name__ == "__main__":
    main()
