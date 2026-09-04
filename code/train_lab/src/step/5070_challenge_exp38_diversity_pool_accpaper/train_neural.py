# -*- coding: utf-8 -*-
"""Exp38 D1：在 challenge 59ch 上训练家族候选（eegtcnet / deep4）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch.nn as nn
from braindecode.models import Deep4Net, EEGTCNet

_STEP = Path(__file__).resolve().parent
from exp38_config import OUT_ROOT_TAG, RUN_TAG, a59_step  # noqa: E402

_A59 = a59_step()
if str(_A59) not in sys.path:
    sys.path.insert(0, str(_A59))

from shared_hparams import SHARED, hp_for_conformer  # noqa: E402
from task_runner import run_baseline_main  # noqa: E402


def build_eegtcnet(n_chans, n_times, n_outputs, drop_prob) -> nn.Module:
    return EEGTCNet(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, drop_prob=drop_prob
    )


def build_deep4(n_chans, n_times, n_outputs, drop_prob) -> nn.Module:
    return Deep4Net(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, drop_prob=drop_prob
    )


BUILDERS = {
    "eegtcnet": (build_eegtcnet, "EEGTCNet · Exp38 D1 · 59ch", SHARED),
    "deep4": (build_deep4, "Deep4Net · Exp38 D1（替代 dgcnn_raw/8ch）", hp_for_conformer()),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="eegtcnet,deep4")
    ap.add_argument("--run-tag", default=RUN_TAG)
    ap.add_argument("--max-folds", type=int, default=0)
    args = ap.parse_args()

    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        if name not in BUILDERS:
            raise SystemExit(f"unknown model {name}")
        build, note, hp = BUILDERS[name]
        # inject out-root via argv for task_runner
        sys.argv = [
            sys.argv[0],
            "--run-tag",
            args.run_tag,
            "--out-root-tag",
            OUT_ROOT_TAG,
            "--max-folds",
            str(args.max_folds),
        ]
        print(f"==== train {name} → out/{OUT_ROOT_TAG} ====", flush=True)
        run_baseline_main(
            model_name=name,
            build_model=build,
            structure_note=note,
            extra_meta={"experiment": 38, "stage": "D1", "family": name},
            hp=hp,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
