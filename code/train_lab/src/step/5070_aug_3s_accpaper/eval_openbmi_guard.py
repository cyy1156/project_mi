"""G1 三端评测之一：OpenBMI 域内 Test Acc_paper 护栏读数。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from s25_config import RESULTS_ROOT, anchor_s3_openbmi_three
from s25_weights import resolve_g1_run


def _read_head_summary(run_dir: Path, head: str) -> dict:
    path = run_dir / head / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="方案25 OpenBMI 域内护栏")
    p.add_argument("--run-stamp", default="")
    p.add_argument("--train-device", default="5070", choices=("5070", "5090"))
    args = p.parse_args()

    run_dir = resolve_g1_run(
        run_stamp=args.run_stamp or None, train_device=args.train_device
    )
    anchor_three = anchor_s3_openbmi_three(args.train_device)
    three = _read_head_summary(run_dir, "three")
    task = _read_head_summary(run_dir, "task")

    test_three = float(three["test_acc_paper_mean"])
    delta = test_three - anchor_three
    pass_guard = delta >= -0.01

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_ROOT / "S25-G1_openbmi_guard" / f"{stamp}_guard"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "weight_run": str(run_dir),
        "anchor_S3_three": anchor_three,
        "train_device": args.train_device,
        "task_test_acc_paper": float(task["test_acc_paper_mean"]),
        "three_test_acc_paper": test_three,
        "delta_three_vs_S3": delta,
        "guard_pass_ge_minus_1pp": pass_guard,
        "task_summary": task,
        "three_summary": three,
    }
    (out_dir / "guard.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = [
        "# S25-G1 OpenBMI 域内护栏",
        "",
        f"- 权重：`{run_dir}`",
        f"- Three Test Acc_paper：`{test_three:.4f}`",
        f"- S3 锚点（{args.train_device}）：`{anchor_three:.4f}`",
        f"- Δ：`{delta*100:+.2f} pp`",
        f"- 护栏（≥ −1 pp）：**{'PASS' if pass_guard else 'FAIL'}**",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[done] {out_dir} guard={'PASS' if pass_guard else 'FAIL'}")


if __name__ == "__main__":
    main()
