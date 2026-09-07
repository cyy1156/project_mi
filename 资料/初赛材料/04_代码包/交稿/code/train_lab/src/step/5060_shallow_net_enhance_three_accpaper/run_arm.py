"""方案 09 · Shallow 结构臂入口（S0 / S1 / S3）。

用法:
  python run_arm.py --arm S0
  python run_arm.py --arm S3_mlp
  python run_arm.py --arm S3_stats
"""

from __future__ import annotations

import argparse

import torch.nn as nn
from braindecode.models import ShallowFBCSPNet

from models_s3 import build_s3_mlp, build_s3_stats
from task_runner import run_baseline_main


def _build(
    *,
    filter_time_length: int = 25,
    n_filters_time: int = 40,
    n_filters_spat: int = 40,
    pool_time_length: int = 75,
    pool_time_stride: int = 15,
):
    def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
        return ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            drop_prob=drop_prob,
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_filters_spat=n_filters_spat,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            final_conv_length="auto",
        )

    return build_model


ARM_PRESETS: dict[str, dict] = {
    "S0": dict(filter_time_length=25, n_filters_time=40, n_filters_spat=40, pool_time_stride=15),
    "S1a_t13": dict(filter_time_length=13, n_filters_time=40, n_filters_spat=40, pool_time_stride=15),
    "S1a_t25": dict(filter_time_length=25, n_filters_time=40, n_filters_spat=40, pool_time_stride=15),
    "S1a_t50": dict(filter_time_length=50, n_filters_time=40, n_filters_spat=40, pool_time_stride=15),
    "S1b_f20": dict(filter_time_length=25, n_filters_time=20, n_filters_spat=20, pool_time_stride=15),
    "S1b_f40": dict(filter_time_length=25, n_filters_time=40, n_filters_spat=40, pool_time_stride=15),
    "S1b_f64": dict(filter_time_length=25, n_filters_time=64, n_filters_spat=64, pool_time_stride=15),
    "S1c_ps10": dict(filter_time_length=25, n_filters_time=40, n_filters_spat=40, pool_time_stride=10),
    "S1c_ps15": dict(filter_time_length=25, n_filters_time=40, n_filters_spat=40, pool_time_stride=15),
    "S1c_ps25": dict(filter_time_length=25, n_filters_time=40, n_filters_spat=40, pool_time_stride=25),
    "S3_mlp": dict(head="mlp"),
    "S3_stats": dict(head="stats"),
}


def main():
    import sys

    p = argparse.ArgumentParser(description="09 shallow net-enhance arm")
    p.add_argument("--arm", type=str, required=True, choices=sorted(ARM_PRESETS.keys()))
    p.add_argument("--filter-time-length", type=int, default=0, help="覆盖预设核长（>0）")
    p.add_argument("--n-filters", type=int, default=0, help="同时覆盖 time/spat 滤波器数（>0）")
    p.add_argument("--pool-time-stride", type=int, default=0, help="覆盖池化步长（>0）")
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]

    preset = dict(ARM_PRESETS[args.arm])
    if preset.get("head") == "mlp":
        build_model = build_s3_mlp
        note = "ShallowFBCSPNet S3_mlp | AdaptiveAvgPool + MLP(40→64→C)"
        meta_cfg = {"arm": args.arm, "head": "mlp", "hidden": 64}
    elif preset.get("head") == "stats":
        build_model = build_s3_stats
        note = "ShallowFBCSPNet S3_stats | mean/std/max over T + Linear"
        meta_cfg = {"arm": args.arm, "head": "stats"}
    else:
        cfg = dict(preset)
        if args.filter_time_length > 0:
            cfg["filter_time_length"] = int(args.filter_time_length)
        if args.n_filters > 0:
            cfg["n_filters_time"] = int(args.n_filters)
            cfg["n_filters_spat"] = int(args.n_filters)
        if args.pool_time_stride > 0:
            cfg["pool_time_stride"] = int(args.pool_time_stride)
        build_model = _build(**cfg)
        note = (
            f"ShallowFBCSPNet {args.arm} | "
            f"tlen={cfg['filter_time_length']} nF={cfg['n_filters_time']}/"
            f"{cfg['n_filters_spat']} pool_stride={cfg['pool_time_stride']}"
        )
        meta_cfg = {"arm": args.arm, **cfg}

    model_name = f"shallow_{args.arm.lower()}"
    run_baseline_main(
        model_name=model_name,
        build_model=build_model,
        input_kind="time",
        structure_note=note,
        extra_meta={
            "shallow_net_enhance": meta_cfg,
            "accpaper": True,
            "bypass": True,
            "scheme": "09",
        },
    )


if __name__ == "__main__":
    main()
