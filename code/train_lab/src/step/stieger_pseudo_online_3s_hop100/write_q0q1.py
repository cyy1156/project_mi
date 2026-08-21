"""S07-04：把 S07-01(Q0) 与 S07-03(Q1) 链成读数表（不另训）。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from config import RESULTS_ROOT


def _latest(dir_path: Path, pattern: str) -> Path | None:
    if not dir_path.is_dir():
        return None
    cands = sorted(dir_path.glob(pattern), key=lambda p: p.name, reverse=True)
    return cands[0] if cands else None


def main() -> None:
    p = argparse.ArgumentParser(description="S07-04 Q0/Q1 汇总表")
    p.add_argument("--zeroshot-summary", default="", help="S07-01 summary.json")
    p.add_argument("--gate-summary", default="", help="S07-03 summary.json")
    args = p.parse_args()

    zs = (
        Path(args.zeroshot_summary)
        if args.zeroshot_summary
        else _latest(RESULTS_ROOT / "S07-01_zeroshot", "*/summary.json")
    )
    gt = (
        Path(args.gate_summary)
        if args.gate_summary
        else _latest(RESULTS_ROOT / "S07-03_gate", "*/summary.json")
    )
    out_dir = RESULTS_ROOT / "S07-04_q0q1_table"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    lines = [
        f"# S07-04 Q0/Q1 读数表 · {stamp}",
        "",
        "> 本臂不另训；Q0≡S07-01（H0）；Q1≡S07-03（H0–H3）。",
        "",
        f"- Q0 源：`{zs}`",
        f"- Q1 源：`{gt}`",
        "",
    ]
    pack = {"stamp": stamp, "q0": str(zs), "q1": str(gt)}
    if zs and zs.is_file():
        lines += ["## Q0（无门控）", "", "见 S07-01 `summary.json`。", ""]
        pack["q0_exists"] = True
    else:
        lines += ["## Q0", "", "- **尚未跑 S07-01**", ""]
        pack["q0_exists"] = False
    if gt and gt.is_file():
        lines += ["## Q1（H0–H3）", "", "见 S07-03 `summary.json` / `report.md`。", ""]
        pack["q1_exists"] = True
    else:
        lines += ["## Q1", "", "- **尚未跑 S07-03**", ""]
        pack["q1_exists"] = False

    md_path = out_dir / f"{stamp}_q0q1.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"{stamp}_q0q1.json").write_text(
        json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {md_path}")


if __name__ == "__main__":
    main()
