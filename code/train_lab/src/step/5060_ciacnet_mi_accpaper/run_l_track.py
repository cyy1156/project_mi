"""L 轨入口：本室 Acc_paper 协议下 EEGNet vs CIACNet。

L0 = fold0 only；L1 = 五折（门控后）。

用法:
  python run_l_track.py --arm L0e                 # EEGNet fold0
  python run_l_track.py --arm L0c                 # CIACNet fold0
  python run_l_track.py --arm L1e                 # EEGNet 五折
  python run_l_track.py --arm L1c                 # CIACNet 五折
  python chain_l0.py                              # L0e → L0c
"""
from __future__ import annotations

import argparse
import sys

import torch.nn as nn

from ciacnet_model import build_ciacnet
from eegnet_ref import build_eegnet
from task_runner import run_baseline_main


def build_eegnet_l(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return build_eegnet(n_chans, n_times, n_outputs, drop_prob=drop_prob)


def build_ciacnet_l(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    # L 轨：结构同 P；dropout 跟本室 SHARED.drop_prob（公平对照）
    return build_ciacnet(
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        drop_prob=drop_prob,
        ablation="full",
    )


ARMS = {
    "L0e": dict(model="eegnet", max_folds=1, note="L0 fold0 EEGNet · Acc_paper"),
    "L0c": dict(model="ciacnet", max_folds=1, note="L0 fold0 CIACNet · Acc_paper"),
    "L1e": dict(model="eegnet", max_folds=0, note="L1 五折 EEGNet · Acc_paper"),
    "L1c": dict(model="ciacnet", max_folds=0, note="L1 五折 CIACNet · Acc_paper"),
}


def main() -> None:
    p = argparse.ArgumentParser(description="CIACNet L-track (lab Acc_paper)")
    p.add_argument("--arm", required=True, choices=sorted(ARMS.keys()))
    args, rest = p.parse_known_args()
    cfg = ARMS[args.arm]

    # inject --max-folds for L0
    extra = []
    if cfg["max_folds"] > 0 and "--max-folds" not in rest:
        extra.extend(["--max-folds", str(cfg["max_folds"])])
    sys.argv = [sys.argv[0], *rest, *extra]

    if cfg["model"] == "eegnet":
        run_baseline_main(
            model_name=f"l_{args.arm.lower()}_eegnet",
            build_model=build_eegnet_l,
            input_kind="time",
            structure_note=cfg["note"] + " | EEGNet F1=8 D=2 F2=16",
            extra_meta={"track": "L", "arm": args.arm, "model": "eegnet", "accpaper": True},
        )
    else:
        run_baseline_main(
            model_name=f"l_{args.arm.lower()}_ciacnet",
            build_model=build_ciacnet_l,
            input_kind="time",
            structure_note=cfg["note"] + " | CIACNet full (CV1+CV2+IAT+TC)",
            extra_meta={"track": "L", "arm": args.arm, "model": "ciacnet", "accpaper": True},
        )


if __name__ == "__main__":
    main()
