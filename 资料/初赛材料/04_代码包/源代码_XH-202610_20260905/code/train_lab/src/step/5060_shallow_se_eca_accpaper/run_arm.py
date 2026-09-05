"""方案 14 · Shallow 前置 SE / ECA 入口。

用法:
  python run_arm.py --arm S0
  python run_arm.py --arm A1                 # SE → Shallow · fold0
  python run_arm.py --arm B1                 # ECA → Shallow · fold0
  python run_arm.py --arm A2                 # SE 五折
  python run_arm.py --arm B2                 # ECA 五折
  python run_arm.py --arm A0 --max-epochs 2  # SE 冒烟
  python run_arm.py --arm B0 --max-epochs 2  # ECA 冒烟
"""

from __future__ import annotations

import argparse
import sys

from models import build_s0, build_shallow_eca, build_shallow_se
from task_runner import run_baseline_main

# max_folds: 0 = 全五折；1 = fold0 only
ARMS: dict[str, dict] = {
    "S0": dict(
        kind="s0",
        max_folds=0,
        note="S0 原版 ShallowFBCSPNet · Acc_paper",
    ),
    "A0": dict(
        kind="se",
        max_folds=1,
        note="A0 SE→Shallow 冒烟 · fold0",
        reduction=2,
    ),
    "A1": dict(
        kind="se",
        max_folds=1,
        note="A1 SE→Shallow · fold0 · Acc_paper",
        reduction=2,
    ),
    "A2": dict(
        kind="se",
        max_folds=0,
        note="A2 SE→Shallow · 五折 · Acc_paper",
        reduction=2,
    ),
    "B0": dict(
        kind="eca",
        max_folds=1,
        note="B0 ECA→Shallow 冒烟 · fold0",
        k_size=3,
    ),
    "B1": dict(
        kind="eca",
        max_folds=1,
        note="B1 ECA→Shallow · fold0 · Acc_paper",
        k_size=3,
    ),
    "B2": dict(
        kind="eca",
        max_folds=0,
        note="B2 ECA→Shallow · 五折 · Acc_paper",
        k_size=3,
    ),
}


def main() -> None:
    p = argparse.ArgumentParser(description="14 shallow front SE/ECA arm")
    p.add_argument("--arm", required=True, choices=sorted(ARMS.keys()))
    p.add_argument("--se-reduction", type=int, default=0, help=">0 覆盖 SE reduction")
    p.add_argument("--eca-k", type=int, default=0, help=">0 覆盖 ECA kernel size")
    args, rest = p.parse_known_args()
    cfg = dict(ARMS[args.arm])

    extra: list[str] = []
    if int(cfg["max_folds"]) > 0 and "--max-folds" not in rest:
        extra.extend(["--max-folds", str(cfg["max_folds"])])
    sys.argv = [sys.argv[0], *rest, *extra]

    kind = cfg["kind"]
    if kind == "s0":
        build = build_s0
        note = cfg["note"]
        meta = {"arm": args.arm, "attn": None}
        model_name = f"shallow_{args.arm.lower()}"
    elif kind == "se":
        r = int(args.se_reduction) if args.se_reduction > 0 else int(cfg["reduction"])

        def build(n_chans, n_times, n_outputs, drop_prob, _r=r):
            return build_shallow_se(
                n_chans, n_times, n_outputs, drop_prob, reduction=_r
            )

        note = f"{cfg['note']} | SE r={r}"
        meta = {"arm": args.arm, "attn": "SE", "se_reduction": r}
        model_name = f"shallow_{args.arm.lower()}_se"
    else:
        k = int(args.eca_k) if args.eca_k > 0 else int(cfg["k_size"])

        def build(n_chans, n_times, n_outputs, drop_prob, _k=k):
            return build_shallow_eca(
                n_chans, n_times, n_outputs, drop_prob, k_size=_k
            )

        note = f"{cfg['note']} | ECA k={k}"
        meta = {"arm": args.arm, "attn": "ECA", "eca_k": k}
        model_name = f"shallow_{args.arm.lower()}_eca"

    run_baseline_main(
        model_name=model_name,
        build_model=build,
        input_kind="time",
        structure_note=note,
        extra_meta={
            "shallow_se_eca": meta,
            "accpaper": True,
            "bypass": True,
            "scheme": "14",
        },
    )


if __name__ == "__main__":
    main()
