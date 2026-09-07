"""方案 28 · 集成成员经济性回放消融（R0–R6 · 零训练）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from e1_fusion_core import (  # noqa: E402
    E1Config,
    acc_paper_for_split,
    calibrate_members,
    fit_e1_config,
    fuse_pipeline,
    search_smooth_radius,
    search_tau_conf,
    search_weights,
    simulate_conf_early_stop,
)
from member_runs import member_run_dirs  # noqa: E402
from prob_io import load_members  # noqa: E402
from s26_config import ANCHOR_E_UNIFORM, DEFAULT_MEMBERS  # noqa: E402
from s28_config import (  # noqa: E402
    ANCHOR_E1F,
    ANCHOR_S3,
    ARM_MEMBERS,
    R1_ADOPT_THREE,
    R28_ARMS,
    R4_ADOPT_THREE,
    SANITY_TOL,
)


def _build_e1c(members: list[dict]) -> E1Config:
    temps = calibrate_members(members)
    w = search_weights(members, temperatures=temps, smooth_radius=0)
    r = search_smooth_radius(members, temperatures=temps, weights=w)
    return E1Config(temperatures=temps, weights=w, smooth_radius=r)


def _build_e1f(members: list[dict]) -> E1Config:
    base = fit_e1_config(members, use_temp=True, use_weights=True, use_smooth=True)
    fused = fuse_pipeline(
        members,
        temperatures=base.temperatures,
        weights=base.weights,
        smooth_radius=base.smooth_radius,
    )
    tau = search_tau_conf(fused, split="val")
    return E1Config(
        temperatures=base.temperatures,
        weights=base.weights,
        smooth_radius=base.smooth_radius,
        tau_conf=tau,
    )


def build_r28_config(arm: str, members: list[dict]) -> E1Config:
    n = len(members)
    uniform = tuple([1.0 / n] * n)
    if arm == "R0":
        return E1Config(temperatures=[1.0] * n, weights=uniform, smooth_radius=0)
    if arm == "R1":
        return _build_e1c(members)
    if arm == "R2":
        return _build_e1f(members)
    if arm == "R3":
        return fit_e1_config(members, use_temp=True, use_weights=True, use_smooth=True)
    if arm in ("R4", "R5", "R6"):
        return _build_e1f(members)
    raise ValueError(f"unknown arm {arm}")


def fuse_for_arm(arm: str, members: list[dict], cfg: E1Config) -> dict:
    if arm == "R0" and len(members) == 1:
        return members[0]
    fused = fuse_pipeline(
        members,
        temperatures=cfg.temperatures,
        weights=cfg.weights,
        smooth_radius=cfg.smooth_radius,
    )
    if cfg.tau_conf is not None:
        fused = simulate_conf_early_stop(fused, tau_conf=cfg.tau_conf)
    return fused


def _verdict_for_arm(arm: str, test_acc: float) -> str:
    if arm == "R0":
        if abs(test_acc - ANCHOR_S3) <= SANITY_TOL:
            return "sanity_ok"
        return "sanity_warn"
    if arm == "R1":
        if test_acc >= R1_ADOPT_THREE:
            return "adopt"
        return "negative"
    if arm == "R4":
        if test_acc >= R4_ADOPT_THREE:
            return "adopt"
        return "negative"
    if arm == "R6":
        if abs(test_acc - ANCHOR_E1F) <= SANITY_TOL:
            return "sanity_ok"
        return "sanity_warn"
    return "report"


def _threshold_for_arm(arm: str) -> str | None:
    if arm == "R0":
        return f"={ANCHOR_S3}±{SANITY_TOL}"
    if arm == "R1":
        return f">={R1_ADOPT_THREE}"
    if arm == "R4":
        return f">={R4_ADOPT_THREE}"
    if arm == "R6":
        return f"={ANCHOR_E1F}±{SANITY_TOL}"
    return None


def run_arm(
    arm: str,
    *,
    overrides: dict[str, Path | None] | None = None,
    out: Path | None = None,
) -> dict:
    names = list(ARM_MEMBERS[arm])
    run_dirs = member_run_dirs(names, overrides=overrides)
    members = load_members(run_dirs)
    cfg = build_r28_config(arm, members)
    fused = fuse_for_arm(arm, members, cfg)

    val_acc = acc_paper_for_split(fused, "val")
    test_acc = acc_paper_for_split(fused, "test")
    result = {
        "arm": arm,
        "member_names": names,
        "member_runs": [str(x) for x in run_dirs],
        "config": cfg.to_dict(),
        "val_acc_paper": val_acc,
        "test_acc_paper": test_acc,
        "anchor_S3": ANCHOR_S3,
        "anchor_E1f": ANCHOR_E1F,
        "anchor_E_uniform": ANCHOR_E_UNIFORM,
        "delta_test_pp_vs_S3": (test_acc - ANCHOR_S3) * 100.0,
        "delta_test_pp_vs_E1f": (test_acc - ANCHOR_E1F) * 100.0,
        "delta_test_pp_vs_E_uniform": (test_acc - ANCHOR_E_UNIFORM) * 100.0,
        "threshold": _threshold_for_arm(arm),
        "verdict": _verdict_for_arm(arm, test_acc),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    out_path = out or (HERE / f"replay_{arm.lower()}.json")
    out_path.write_text(text, encoding="utf-8")
    return result


def load_existing_results(base: Path = HERE) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for arm in R28_ARMS:
        path = base / f"replay_{arm.lower()}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run replay_r28 --arm {arm} first")
        results[arm] = json.loads(path.read_text(encoding="utf-8"))
    return results


def write_summary(
    results: dict[str, dict],
    summary_path: Path,
) -> dict:
    decision = evaluate_decision_tree(results)
    summary = {
        "scheme": 28,
        "arms": results,
        "decision_tree": decision,
        "attachments": {
            "R2_minus_R1_pp": (
                results["R2"]["test_acc_paper"] - results["R1"]["test_acc_paper"]
            )
            * 100.0,
            "R4_minus_R3_pp": (
                results["R4"]["test_acc_paper"] - results["R3"]["test_acc_paper"]
            )
            * 100.0,
            "R5_minus_R4_pp": (
                results["R5"]["test_acc_paper"] - results["R4"]["test_acc_paper"]
            )
            * 100.0,
            "R6_minus_anchor_E1f_pp": results["R6"]["delta_test_pp_vs_E1f"],
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def evaluate_decision_tree(results: dict[str, dict]) -> dict:
    r1 = results["R1"]["test_acc_paper"]
    r4 = results["R4"]["test_acc_paper"]
    if r1 >= R1_ADOPT_THREE:
        branch = "A"
        action = "单模 shallow + smooth + τ_conf（附报 R2）"
    elif r4 >= R4_ADOPT_THREE:
        branch = "B"
        action = "两成员 {shallow, eegnet} + E1 后处理；砍 conformer + T-shallow"
    else:
        branch = "C"
        action = "维持 E1f 四成员；仅砍 T-shallow"
    return {
        "branch": branch,
        "action": action,
        "R1_test": r1,
        "R1_line": R1_ADOPT_THREE,
        "R4_test": r4,
        "R4_line": R4_ADOPT_THREE,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme28 member economy replay (R0–R6)")
    p.add_argument(
        "--arm",
        required=True,
        choices=(*R28_ARMS, "all"),
        help="R0–R6 or all",
    )
    p.add_argument("--shallow-run", type=Path, default=None)
    p.add_argument("--t-shallow-run", type=Path, default=None)
    p.add_argument("--eegnet-run", type=Path, default=None)
    p.add_argument("--conformer-run", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="output json (single arm)")
    p.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="decision summary json when --arm all",
    )
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="read replay_r*.json and write summary without re-running arms",
    )
    args = p.parse_args()

    if args.summarize_only:
        results = load_existing_results()
        summary_path = args.summary_out or (HERE / "replay_r28_summary.json")
        summary = write_summary(results, summary_path)
        print(json.dumps({"decision_tree": summary["decision_tree"]}, indent=2, ensure_ascii=False))
        print(f"wrote summary -> {summary_path}")
        return

    overrides = {
        "shallow": args.shallow_run,
        "t_shallow": args.t_shallow_run,
        "eegnet": args.eegnet_run,
        "conformer": args.conformer_run,
    }

    if args.arm != "all":
        run_arm(args.arm, overrides=overrides, out=args.out)
        return

    results: dict[str, dict] = {}
    for arm in R28_ARMS:
        out = HERE / f"replay_{arm.lower()}.json"
        results[arm] = run_arm(arm, overrides=overrides, out=out)

    summary_path = args.summary_out or (HERE / "replay_r28_summary.json")
    summary = write_summary(results, summary_path)
    print(json.dumps({"decision_tree": summary["decision_tree"]}, indent=2, ensure_ascii=False))
    print(f"wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
