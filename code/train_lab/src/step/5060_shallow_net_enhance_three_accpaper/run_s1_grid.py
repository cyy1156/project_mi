"""S1 超参网格搜索调度器（顺序搜索 S1a→S1b→S1c→S1d）。

严格按方案：
  S1a: filter_time_length ∈ {13, 25, 50}       — 其余=S0默认
  S1b: n_filters ∈ {20, 40, 64} (time=spat)    — 在 S1a 最优核长上
  S1c: pool_time_stride ∈ {10, 15, 25}          — 在 S1b 最优上
  S1d: drop_prob ∈ {0.25, 0.5}                  — 在 S1c 最优上

用法：
  # 第 1 步：S1a 扫核长（3 跑）
  python run_s1_grid.py --stage s1a

  # 第 2 步：S1b 在 S1a 最优上扫宽度（需手动填入 S1a 最优 t）
  python run_s1_grid.py --stage s1b --s1a-best-t 13

  # 第 3 步：S1c 在 S1b 最优上扫 pool stride
  python run_s1_grid.py --stage s1c --s1a-best-t 13 --s1b-best-f 64

  # 第 4 步：S1d 在 S1c 最优上扫 drop_prob
  python run_s1_grid.py --stage s1d --s1a-best-t 13 --s1b-best-f 64 --s1c-best-ps 25

  # 冒烟（1 折 2 轮）
  python run_s1_grid.py --stage s1a --smoke --num-workers 0

  # 只跑指定候选
  python run_s1_grid.py --stage s1a --only 13,50
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

from shallow_variants import S1A_CANDIDATES, S1B_CANDIDATES, S1C_CANDIDATES, S1D_CANDIDATES


def _stage_candidates(stage: str) -> list:
    """返回该阶段的候选值列表。"""
    table = {
        "s1a": S1A_CANDIDATES,
        "s1b": S1B_CANDIDATES,
        "s1c": S1C_CANDIDATES,
        "s1d": S1D_CANDIDATES,
    }
    if stage not in table:
        raise ValueError(f"未知阶段: {stage}; 可选: {list(table.keys())}")
    return table[stage]


def _make_arm_name(stage: str, value) -> str:
    """根据阶段和值生成臂名。"""
    if stage == "s1a":
        return f"s1a_t{value}"
    if stage == "s1b":
        return f"s1b_f{value}"
    if stage == "s1c":
        return f"s1c_ps{value}"
    if stage == "s1d":
        return f"s1d_d{int(round(value * 100)):03d}"  # 0.25→d025, 0.5→d050
    raise ValueError(stage)


def _stage_base_args(stage: str, args) -> list:
    """构建 --s1a-best-t / --s1b-best-f / --s1c-best-ps 命令行参数。"""
    base = []
    if stage in ("s1b", "s1c", "s1d"):
        if args.s1a_best_t is None:
            raise ValueError(f"{stage} 需要 --s1a-best-t")
        base.extend(["--s1a-best-t", str(args.s1a_best_t)])
    if stage in ("s1c", "s1d"):
        if args.s1b_best_f is None:
            raise ValueError(f"{stage} 需要 --s1b-best-f")
        base.extend(["--s1b-best-f", str(args.s1b_best_f)])
    if stage == "s1d":
        if args.s1c_best_ps is None:
            raise ValueError(f"{stage} 需要 --s1c-best-ps")
        base.extend(["--s1c-best-ps", str(args.s1c_best_ps)])
    return base


def main() -> None:
    p = argparse.ArgumentParser(
        description="S1 超参网格搜索调度器（顺序搜索 S1a→S1b→S1c→S1d）"
    )
    p.add_argument(
        "--stage",
        required=True,
        choices=("s1a", "s1b", "s1c", "s1d"),
        help="搜索阶段",
    )
    p.add_argument(
        "--data",
        default="openbmi_2s_hop100",
        choices=("openbmi_2s_hop100", "openbmi_2s_fixed_cue2to4_noz"),
        help="OpenBMI 数据",
    )
    # 前序阶段最优参数
    p.add_argument("--s1a-best-t", type=int, default=None,
                   help="S1a 最优 filter_time_length")
    p.add_argument("--s1b-best-f", type=int, default=None,
                   help="S1b 最优 n_filters")
    p.add_argument("--s1c-best-ps", type=int, default=None,
                   help="S1c 最优 pool_time_stride")
    # 筛选
    p.add_argument("--only", default="",
                   help="逗号分隔的候选值，只跑这些（如 13,50）")
    # 运行参数
    p.add_argument("--smoke", action="store_true",
                   help="冒烟模式：max-folds=1 max-epochs=2 patience=2")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--continue-on-error", action="store_true",
                   help="某候选失败后继续跑下一个")
    p.add_argument("--skip-three", action="store_true",
                   help="跳过 Three（只跑 Task）")
    args = p.parse_args()

    # 获取候选
    candidates = _stage_candidates(args.stage)
    if args.only:
        # 解析筛选
        only_vals = set()
        for s in args.only.split(","):
            s = s.strip()
            if not s:
                continue
            if args.stage == "s1d":
                only_vals.add(float(s))
            else:
                only_vals.add(int(s))
        candidates = [c for c in candidates if c in only_vals]
        if not candidates:
            print(f"ERROR: --only 筛选后无候选", flush=True)
            return

    # 构建 base 参数
    try:
        base_args = _stage_base_args(args.stage, args)
    except ValueError as e:
        print(f"ERROR: {e}", flush=True)
        return

    print(f"S1 网格搜索 · 阶段 {args.stage}", flush=True)
    print(f"  候选: {candidates}", flush=True)
    print(f"  base args: {base_args if base_args else '(S0 默认)'}", flush=True)
    print(f"  data={args.data} smoke={args.smoke} workers={args.num_workers}", flush=True)
    print()

    # 构建公共参数
    common_args = [
        sys.executable,
        str(HERE / "run_arm.py"),
        "--data", args.data,
        "--num-workers", str(args.num_workers),
    ] + base_args
    if args.smoke:
        common_args.extend(["--max-folds", "1", "--max-epochs", "2", "--patience", "2"])
    if args.skip_three:
        common_args.append("--skip-three")

    results: list[dict] = []
    failed: list[str] = []
    for i, val in enumerate(candidates, 1):
        arm = _make_arm_name(args.stage, val)
        cmd = common_args + ["--arm", arm]
        print(f"\n{'='*60}", flush=True)
        print(f"[{i}/{len(candidates)}] {args.stage} 候选值: {val} → arm={arm}", flush=True)
        print(f"  cmd: {' '.join(cmd)}", flush=True)
        print(f"{'='*60}", flush=True)

        r = subprocess.run(cmd, cwd=str(HERE))
        entry = {
            "stage": args.stage,
            "value": val,
            "arm": arm,
            "exit_code": r.returncode,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if r.returncode != 0:
            print(f"  FAILED: {arm} (exit={r.returncode})", flush=True)
            failed.append(arm)
            if not args.continue_on_error:
                print("  --continue-on-error 未启用，停止后续候选", flush=True)
                results.append(entry)
                break
        else:
            print(f"  OK: {arm}", flush=True)
        results.append(entry)

    # 汇总
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = HERE / f"_s1_{args.stage}_summary_{stamp}.json"
    summary = {
        "stage": args.stage,
        "candidates": candidates,
        "base_args": dict(zip(
            ["s1a_best_t", "s1b_best_f", "s1c_best_ps"],
            [args.s1a_best_t, args.s1b_best_f, args.s1c_best_ps]
        )),
        "timestamp": stamp,
        "data": args.data,
        "smoke": args.smoke,
        "total": len(candidates),
        "succeeded": len([r for r in results if r["exit_code"] == 0]),
        "failed": failed,
        "results": results,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{'='*60}", flush=True)
    print(f"{args.stage} 搜索完成：{summary['succeeded']}/{summary['total']} 成功", flush=True)
    if failed:
        print(f"失败: {failed}", flush=True)
    print(f"汇总: {summary_path}", flush=True)
    print(f"\n请查看五折结果，选出 {args.stage} 最优值，用于下一阶段。", flush=True)


if __name__ == "__main__":
    main()
