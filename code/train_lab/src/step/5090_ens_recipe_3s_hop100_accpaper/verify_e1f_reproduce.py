"""E1f（0.6173）复现链校验：方案 24 四成员 dump + 方案 26 E1f 零训练融合。

5070 在 git pull prob dump 后：
  python verify_e1f_reproduce.py              # 产物检查 + 读 replay_e1f.json
  python verify_e1f_reproduce.py --replay     # 重跑 E1f 融合（~90min，不重训）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from member_runs import member_run_dirs  # noqa: E402
from s26_config import ANCHOR_E_UNIFORM, DEFAULT_MEMBERS  # noqa: E402

PKG24 = HERE.parent / "5090_baselines_openbmi_3s_hop100_accpaper"

ANCHOR_E1F = 0.6173
SANITY_TOL = 0.0005
MEMBER_NAMES = ("shallow", "t_shallow", "eegnet", "conformer")
MEMBER_TEST_ANCHOR = {
    "shallow": 0.5839,
    "t_shallow": 0.5886,
    "eegnet": 0.5629,
    "conformer": 0.5767,
}
EXPECTED_E1F_CONFIG = {
    "smooth_radius": 1,
    "tau_conf": 0.4,
    "weights": [0.2, 0.2, 0.3, 0.3],
}


def _check_task_heads() -> None:
    """2026-08-29 补 Task 头；E1f 融合链仍只用 Three prob。"""
    task_anchor = {
        "shallow": 0.7424,
        "t_shallow": 0.7403,
        "eegnet": 0.7240,
        "conformer": 0.7597,
    }
    for name in MEMBER_NAMES:
        run_root = DEFAULT_MEMBERS.as_dict()[name].parent
        task_dir = run_root / "task"
        assert task_dir.is_dir(), f"{name}: missing task/ under {run_root}"
        summary_path = task_dir / "summary.json"
        assert summary_path.is_file(), f"{name}: missing {summary_path}"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        test_mean = float(summary["test_acc_paper_mean"])
        anchor = task_anchor[name]
        flag = "OK" if abs(test_mean - anchor) <= 0.003 else "WARN"
        dumps = sorted(task_dir.glob("fold*/prob_dump_task.csv"))
        assert len(dumps) >= 5, f"{name}: need 5 task prob dumps, got {len(dumps)}"
        fold0_pt = task_dir / "fold0" / "best_task.pt"
        pt_note = "OK" if fold0_pt.is_file() else "missing"
        print(
            f"{flag} task head {name}: test={test_mean:.4f} anchor={anchor:.4f} "
            f"dumps={len(dumps)} fold0_pt={pt_note}"
        )


def _check_dumps(*, full_merge: bool = False) -> None:
    run_dirs = member_run_dirs(list(MEMBER_NAMES))
    for name, run_dir in zip(MEMBER_NAMES, run_dirs):
        assert run_dir.is_dir(), f"missing run dir for {name}: {run_dir}"
        dumps = sorted(run_dir.glob("fold*/prob_dump_three.csv"))
        assert len(dumps) >= 5, f"{name}: need 5 prob dumps, got {len(dumps)} under {run_dir}"
        for d in dumps:
            assert d.stat().st_size > 1000, f"{name}: empty dump {d}"
        fold0_pt = run_dir / "fold0" / "best_three.pt"
        pt_note = "OK" if fold0_pt.is_file() else "missing (replay OK; deploy optional)"
        if full_merge:
            from prob_io import load_run_three

            data = load_run_three(run_dir)
            n_val = int((data["split"] == "val").sum())
            n_test = int((data["split"] == "test").sum())
        else:
            sp = str(PKG24)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            from prob_dump import load_prob_dump

            data = load_prob_dump(dumps[0])
            n_val = int((data["split"] == "val").sum())
            n_test = int((data["split"] == "test").sum())
        assert n_val > 0 and n_test > 0, f"{name}: empty split in dumps"
        tag = "merge5" if full_merge else "fold0"
        print(
            f"OK dump {name}: {len(dumps)} folds {tag} val={n_val} test={n_test} fold0_pt={pt_note}"
        )


def _check_member_summaries() -> None:
    for name in MEMBER_NAMES:
        run_dir = DEFAULT_MEMBERS.as_dict()[name]
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            print(f"SKIP summary {name}: no summary.json (5070 may omit)")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        test_mean = float(summary["test_acc_paper_mean"])
        anchor = MEMBER_TEST_ANCHOR[name]
        delta_pp = (test_mean - anchor) * 100.0
        flag = "OK" if abs(test_mean - anchor) <= 0.003 else "WARN"
        print(f"{flag} member {name}: summary test={test_mean:.4f} anchor={anchor:.4f} Δ={delta_pp:+.2f}pp")


def _validate_e1f_json(path: Path) -> dict:
    assert path.is_file(), f"missing {path}; run with --replay"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result.get("arm") == "E1f", result.get("arm")
    test_acc = float(result["test_acc_paper"])
    delta = abs(test_acc - ANCHOR_E1F)
    assert delta <= SANITY_TOL, f"E1f test {test_acc} vs anchor {ANCHOR_E1F} (tol {SANITY_TOL})"
    cfg = result["config"]
    assert int(cfg["smooth_radius"]) == EXPECTED_E1F_CONFIG["smooth_radius"]
    assert abs(float(cfg["tau_conf"]) - EXPECTED_E1F_CONFIG["tau_conf"]) < 1e-9
    w = [round(float(x), 1) for x in cfg["weights"]]
    assert w == EXPECTED_E1F_CONFIG["weights"], w
    print(
        f"OK E1f json: test={test_acc:.4f} val={result['val_acc_paper']:.4f} "
        f"Δ_vs_E_uniform={result['delta_test_pp_vs_E_uniform']:+.2f}pp verdict={result.get('verdict')}"
    )
    return result


def _replay_e1f(out: Path) -> dict:
    cmd = [
        sys.executable,
        str(HERE / "replay_e1.py"),
        "--arm",
        "E1f",
        "--four-member",
        "--out",
        str(out),
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(HERE))
    return _validate_e1f_json(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Verify E1f 0.6173 reproduction chain (no retrain)")
    p.add_argument(
        "--full-dumps",
        action="store_true",
        help="merge all 5-fold dumps per member (slow; default checks fold0 only)",
    )
    p.add_argument(
        "--replay",
        action="store_true",
        help="re-run replay_e1.py --arm E1f (~90min); default only checks artifacts + json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=HERE / "replay_e1f.json",
        help="E1f replay output json",
    )
    args = p.parse_args()

    print("=== Phase 1 · 方案 24 四成员 prob dump ===")
    _check_dumps(full_merge=args.full_dumps)
    print("=== Phase 1a · Task 头（5090 补训 · E1f 仍只用 Three） ===")
    _check_task_heads()
    print("=== Phase 1b · 单成员 summary（本地有则核对） ===")
    _check_member_summaries()
    print(f"=== Phase 2 · 方案 26 E1f 融合（锚点 E 均匀={ANCHOR_E_UNIFORM} · E1f={ANCHOR_E1F}） ===")
    if args.replay:
        _replay_e1f(args.out)
    else:
        _validate_e1f_json(args.out)
    print("verify_e1f_reproduce: OK")


if __name__ == "__main__":
    main()
