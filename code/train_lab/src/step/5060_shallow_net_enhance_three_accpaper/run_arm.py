"""统一入口：--arm 参数选择实验臂。

严格按方案 `资料/模型训练/09_旁路_shallow_网络结构增强_Three_openbmi_accpaper/方案.md`。

用法：
  # S0 复现锚点
  python run_arm.py --arm s0

  # S1 顺序搜索（需按阶段传入前序最优）
  python run_arm.py --arm s1a_t13                    # S1a: 扫核长
  python run_arm.py --arm s1b_f20 --s1a-best-t 13    # S1b: 在 t=13 上扫宽度
  python run_arm.py --arm s1c_ps10 --s1a-best-t 13 --s1b-best-f 64  # S1c
  python run_arm.py --arm s1d_d025 --s1a-best-t 13 --s1b-best-f 64 --s1c-best-ps 25  # S1d

  # S2 多尺度时间核
  python run_arm.py --arm s2_ms_concat
  python run_arm.py --arm s2_ms_sum

  # S3 读出头增强
  python run_arm.py --arm s3_mlp
  python run_arm.py --arm s3_stats
  python run_arm.py --arm s3_hier               # ⚠️ 需 task_runner 多任务支持
  python run_arm.py --arm s3_three_only_tune

  # S4 训练目标（模型同 S0，loss/评测在 task_runner 侧）
  python run_arm.py --arm s4_focal --focal-gamma 1.0
  python run_arm.py --arm s4_class_weight
  python run_arm.py --arm s4_conf_agg            # 仅评测，不改训练

  # S5 轻量混合骨干
  python run_arm.py --arm s5_res_pre
  python run_arm.py --arm s5_dual
  python run_arm.py --arm s5_se

  # 冒烟
  python run_arm.py --arm s0 --max-folds 1 --max-epochs 2 --patience 2 --num-workers 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from shallow_variants import get_variant, S1_BASE
from baseline_shallow_s0 import build_model as build_s0
from task_runner import run_baseline_main


def main() -> None:
    p = argparse.ArgumentParser(
        description="Shallow 结构增强 · 统一入口（--arm 选择实验臂）"
    )
    p.add_argument(
        "--arm",
        required=True,
        help=(
            "实验臂：s0, "
            "s1a_t13/s1a_t25/s1a_t50, s1b_f20/s1b_f40/s1b_f64, "
            "s1c_ps10/s1c_ps15/s1c_ps25, s1d_d025/s1d_d050, "
            "s2_ms_concat/s2_ms_sum, "
            "s3_mlp/s3_stats/s3_hier/s3_three_only_tune, "
            "s4_softvote_loss/s4_focal/s4_class_weight/s4_conf_agg, "
            "s5_res_pre/s5_dual/s5_se"
        ),
    )
    p.add_argument(
        "--data",
        default="openbmi_2s_hop100",
        choices=("openbmi_2s_hop100", "openbmi_2s_fixed_cue2to4_noz"),
        help="OpenBMI 数据；正式用 hop100，冒烟可用 fixed_cue2to4_noz",
    )
    # S1 顺序搜索：前序阶段最优参数
    p.add_argument("--s1a-best-t", type=int, default=None,
                   help="S1a 最优 filter_time_length（S1b/S1c/S1d 需传入）")
    p.add_argument("--s1b-best-f", type=int, default=None,
                   help="S1b 最优 n_filters（S1c/S1d 需传入）")
    p.add_argument("--s1c-best-ps", type=int, default=None,
                   help="S1c 最优 pool_time_stride（S1d 需传入）")
    # S4 参数
    p.add_argument("--focal-gamma", type=float, default=1.0,
                   help="S4_focal 的 gamma 值（默认 1.0，方案扫 1-2）")
    # 通用训练参数
    p.add_argument("--skip-three", action="store_true")
    p.add_argument("--max-folds", type=int, default=0, help=">0 时只跑前 N 折（冒烟）")
    p.add_argument("--max-epochs", type=int, default=0, help=">0 时覆盖默认 max_epochs")
    p.add_argument("--patience", type=int, default=0, help=">0 时覆盖默认 patience")
    p.add_argument("--num-workers", type=int, default=-1, help=">=0 覆盖 DataLoader workers")
    p.add_argument("--batch-train", type=int, default=0, help=">0 覆盖 batch_train")
    p.add_argument("--batch-eval", type=int, default=0, help=">0 覆盖 batch_eval")
    p.add_argument("--no-amp", action="store_true", help="关闭 CUDA AMP")
    p.add_argument("--deterministic", action="store_true", help="关 cudnn.benchmark")
    p.add_argument("--repro", action="store_true", help="严格复现抽检")
    args = p.parse_args()

    # 设置 S1 base 参数（全局）
    if args.s1a_best_t is not None:
        S1_BASE.filter_time_length = args.s1a_best_t
    if args.s1b_best_f is not None:
        S1_BASE.n_filters = args.s1b_best_f
    if args.s1c_best_ps is not None:
        S1_BASE.pool_time_stride = args.s1c_best_ps

    # S0 单独从 baseline_shallow_s0.py 导入
    if args.arm.lower().strip() == "s0":
        build_fn = build_s0
        structure_note = "ShallowFBCSPNet（braindecode 默认；S0 复现锚点）"
        extra_meta = {
            "shallow": {"backbone": "ShallowFBCSPNet", "variant": "S0_default"},
            "accpaper": True,
        }
    else:
        build_fn, structure_note, extra_meta = get_variant(args.arm)

    # S4: 传递 loss/评测参数
    if args.arm.startswith("s4_focal"):
        extra_meta = {**extra_meta, "s4_focal_gamma": args.focal_gamma}
    if args.arm.startswith("s4_"):
        extra_meta = {**extra_meta, "s4_mode": args.arm}

    # 构建传给 run_baseline_main 的参数
    run_args = []
    if args.data != "openbmi_2s_hop100":
        run_args.extend(["--data", args.data])
    if args.skip_three:
        run_args.append("--skip-three")
    if args.max_folds > 0:
        run_args.extend(["--max-folds", str(args.max_folds)])
    if args.max_epochs > 0:
        run_args.extend(["--max-epochs", str(args.max_epochs)])
    if args.patience > 0:
        run_args.extend(["--patience", str(args.patience)])
    if args.num_workers >= 0:
        run_args.extend(["--num-workers", str(args.num_workers)])
    if args.batch_train > 0:
        run_args.extend(["--batch-train", str(args.batch_train)])
    if args.batch_eval > 0:
        run_args.extend(["--batch-eval", str(args.batch_eval)])
    if args.no_amp:
        run_args.append("--no-amp")
    if args.deterministic:
        run_args.append("--deterministic")
    if args.repro:
        run_args.append("--repro")

    # 覆盖 sys.argv 让 run_baseline_main 的 argparse 接收
    sys.argv = [sys.argv[0]] + run_args

    model_name = f"shallow_{args.arm}"
    run_baseline_main(
        model_name=model_name,
        build_model=build_fn,
        input_kind="time",
        structure_note=structure_note,
        extra_meta=extra_meta,
        arm_name=args.arm,
    )


if __name__ == "__main__":
    main()
