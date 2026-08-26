"""方案 26 · E1 推理满配回放（E1a–E1f）。"""

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
from prob_io import load_members  # noqa: E402
from s26_config import (  # noqa: E402
    ANCHOR_E_UNIFORM,
    DEFAULT_MEMBERS,
    E1_ADOPT_PP,
)


def _parse_runs(args: argparse.Namespace) -> list[Path]:
    m = DEFAULT_MEMBERS
    if args.four_member:
        conf = Path(args.conformer_run) if args.conformer_run else m.conformer
        return [m.shallow, m.t_shallow, m.eegnet, conf]
    return [
        Path(args.shallow_run or m.shallow),
        Path(args.eegnet_run or m.eegnet),
        Path(args.conformer_run or m.conformer),
    ]


def _build_config(arm: str, members: list[dict], args: argparse.Namespace) -> E1Config:
    n = len(members)
    uniform = tuple([1.0 / n] * n)
    if arm == "E1a":
        temps = calibrate_members(members)
        return E1Config(temperatures=temps, weights=uniform, smooth_radius=0)
    if arm == "E1b":
        temps = calibrate_members(members)
        w = search_weights(members, temperatures=temps, smooth_radius=0)
        return E1Config(temperatures=temps, weights=w, smooth_radius=0)
    if arm == "E1c":
        temps = calibrate_members(members)
        w = search_weights(members, temperatures=temps, smooth_radius=0)
        r = search_smooth_radius(members, temperatures=temps, weights=w)
        return E1Config(temperatures=temps, weights=w, smooth_radius=r)
    if arm in ("E1d", "E1e"):
        return fit_e1_config(members, use_temp=True, use_weights=True, use_smooth=True)
    if arm == "E1f":
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
    raise ValueError(f"unknown arm {arm}")


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme26 E1 fusion replay")
    p.add_argument(
        "--arm",
        required=True,
        choices=("E1a", "E1b", "E1c", "E1d", "E1e", "E1f"),
    )
    p.add_argument("--shallow-run", type=Path, default=None)
    p.add_argument("--eegnet-run", type=Path, default=None)
    p.add_argument("--conformer-run", type=Path, default=None)
    p.add_argument("--four-member", action="store_true", help="E1e/E1f: + T-shallow")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.arm == "E1e":
        args.four_member = True
    run_dirs = _parse_runs(args)
    members = load_members(run_dirs)
    cfg = _build_config(args.arm, members, args)

    fused = fuse_pipeline(
        members,
        temperatures=cfg.temperatures,
        weights=cfg.weights,
        smooth_radius=cfg.smooth_radius,
    )
    if cfg.tau_conf is not None:
        fused = simulate_conf_early_stop(fused, tau_conf=cfg.tau_conf)

    val_acc = acc_paper_for_split(fused, "val")
    test_acc = acc_paper_for_split(fused, "test")
    delta_pp = (test_acc - ANCHOR_E_UNIFORM) * 100.0

    out = {
        "arm": args.arm,
        "member_runs": [str(x) for x in run_dirs],
        "config": cfg.to_dict(),
        "val_acc_paper": val_acc,
        "test_acc_paper": test_acc,
        "anchor_E_uniform": ANCHOR_E_UNIFORM,
        "delta_test_pp_vs_E_uniform": delta_pp,
        "adopt_ge_pp": E1_ADOPT_PP,
        "verdict": "adopt" if delta_pp >= E1_ADOPT_PP else ("report" if delta_pp >= 0.2 else "negative"),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    out_path = args.out or (HERE / f"replay_{args.arm.lower()}.json")
    out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
