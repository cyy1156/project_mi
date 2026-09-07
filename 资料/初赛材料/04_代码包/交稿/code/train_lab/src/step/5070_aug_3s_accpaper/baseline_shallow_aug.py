"""G1：OpenBMI 3s shallow 五折 + 域增广（--aug g1）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "5070_baselines_openbmi_3s_hop100_accpaper"
for p in (str(BASE), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from domain_aug import aug_config_from_spec
from patch_baseline import patch_baseline_modules, set_aug_config


def _parse_train_device() -> tuple[str, bool]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--train-device", default="5070", choices=("5070", "5090"))
    p.add_argument(
        "--fast-batch",
        action="store_true",
        help="5090 专用：batch 512/1024 提速（略损与 5070 严格可比）",
    )
    args, _ = p.parse_known_args()
    return args.train_device, bool(args.fast_batch)


def build_model(n_chans: int, n_times: int, n_outputs: int, drop_prob: float):
    from braindecode.models import ShallowFBCSPNet

    return ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
    )


def _parse_aug_from_argv() -> None:
    aug = "g1"
    if "--aug" in sys.argv:
        i = sys.argv.index("--aug")
        if i + 1 < len(sys.argv):
            aug = sys.argv[i + 1]
    cfg = aug_config_from_spec(aug)
    set_aug_config(cfg)


def _argv_for_task_runner(argv: list[str]) -> list[str]:
    """剥离本脚本专用 flag，保留 task_runner 可识别的 CLI（如 --num-workers）。"""
    skip = {"--aug", "--train-device"}
    out = [argv[0]]
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok == "--fast-batch":
            i += 1
            continue
        if tok in skip:
            i += 2 if i + 1 < len(argv) else 1
            continue
        out.append(tok)
        i += 1
    return out


if __name__ == "__main__":
    train_device, fast_batch = _parse_train_device()
    out_tag = patch_baseline_modules(
        train_device=train_device, fast_batch=fast_batch
    )
    _parse_aug_from_argv()
    sys.argv = _argv_for_task_runner(sys.argv)

    from task_runner import run_baseline_main  # noqa: E402

    run_baseline_main(
        model_name="shallow",
        build_model=build_model,
        input_kind="time",
        structure_note=(
            "ShallowFBCSPNet · Tw=3s hop=100ms · 方案25 G1 域增广 "
            f"(post-zscore · train_device={train_device})"
        ),
        extra_meta={
            "shallow": {"backbone": "ShallowFBCSPNet"},
            "accpaper": True,
            "experiment": 25,
            "arm": "G1",
            "train_device": train_device,
            "fast_batch": fast_batch,
            "win_sec": 3.0,
            "hop_sec": 0.1,
            "aug": (
                sys.argv[sys.argv.index("--aug") + 1]
                if "--aug" in sys.argv
                else "g1"
            ),
            "out_root_tag": out_tag,
        },
    )
