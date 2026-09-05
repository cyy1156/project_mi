"""单模型伪在线评测入口（对齐 accpaper 的 task_runner 形态）。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import (  # noqa: E402
    DEFAULT_SESSIONS,
    DOCS_OUT,
    N_FOLDS,
    PROTOCOL,
    SESSIONS_ROOT,
    TOP5_MODELS,
)
from eval_metrics import aggregate_windows_to_segments, mean_std  # noqa: E402
from infer import load_fold_model, predict_windows  # noqa: E402
from stream import build_eval_stream, save_stream_artifacts  # noqa: E402
from weights import resolve_accpaper_run  # noqa: E402

BuildFn = Callable[..., nn.Module]


def _summarize_folds(fold_rows: list[dict], *, head: str) -> dict:
    ap = [float(r["segment_metrics"]["acc_paper"]) for r in fold_rows]
    bm = [float(r["segment_metrics"]["balanced_accuracy"]) for r in fold_rows]
    wb = [float(r["window_metrics"]["balanced_accuracy"]) for r in fold_rows]
    out = {
        "head": head,
        "n_folds": len(fold_rows),
        "acc_paper_mean": mean_std(ap)[0],
        "acc_paper_std": mean_std(ap)[1],
        "balacc_maj_mean": mean_std(bm)[0],
        "balacc_maj_std": mean_std(bm)[1],
        "window_balacc_mean": mean_std(wb)[0],
        "window_balacc_std": mean_std(wb)[1],
        "folds": fold_rows,
    }
    if head == "task":
        for key in ("f1", "specificity", "recall", "accuracy"):
            xs = [float(r["window_metrics"][key]) for r in fold_rows]
            out[f"window_{key}_mean"] = mean_std(xs)[0]
            out[f"window_{key}_std"] = mean_std(xs)[1]
    else:
        for key in ("f1_macro", "accuracy"):
            if key in fold_rows[0]["window_metrics"]:
                xs = [float(r["window_metrics"][key]) for r in fold_rows]
                out[f"window_{key}_mean"] = mean_std(xs)[0]
                out[f"window_{key}_std"] = mean_std(xs)[1]
    return out


def _md_table(summ: dict, *, head: str) -> list[str]:
    return [
        f"### {head}",
        "",
        "| 模型 | 段级 Acc_paper | 段级 BalAcc_maj | 窗级 BalAcc | 合法窗数 | 权重 run |",
        "|------|----------------|-----------------|-------------|---------|----------|",
        f"| {summ['model']} | "
        f"{summ['acc_paper_mean']:.4f} ± {summ['acc_paper_std']:.4f} | "
        f"{summ['balacc_maj_mean']:.4f} ± {summ['balacc_maj_std']:.4f} | "
        f"{summ['window_balacc_mean']:.4f} ± {summ['window_balacc_std']:.4f} | "
        f"{summ['n_windows']} | `{summ['run']}` |",
        "",
    ]


def run_pseudo_online_main(
    *,
    model_name: str,
    build_model: BuildFn,
    structure_note: str,
    extra_meta: dict | None = None,
) -> None:
    if model_name not in TOP5_MODELS:
        raise SystemExit(f"模型 {model_name} 不在本臂 Top5: {TOP5_MODELS}")

    p = argparse.ArgumentParser(
        description=f"{model_name} 游戏伪在线（Acc_paper 权重）"
    )
    p.add_argument("--sessions", default=",".join(DEFAULT_SESSIONS))
    p.add_argument("--skip-three", action="store_true")
    p.add_argument("--smoke", action="store_true", help="仅 fold0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-filter", action="store_true")
    p.add_argument("--run-stamp", default="", help="指定 accpaper run_… 名")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    folds = (0,) if args.smoke else tuple(range(N_FOLDS))
    device = torch.device(args.device)

    run_dir = resolve_accpaper_run(
        model_name, run_stamp=args.run_stamp or None
    )
    print(f"[P-1] {model_name} weights: {run_dir}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{model_name}_pseudo_online"
    result_root = DOCS_OUT / "results" / f"{stamp}_{out_name}"
    result_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "stamp": stamp,
        "model_name": model_name,
        "out_name": out_name,
        "protocol": PROTOCOL,
        "structure_note": structure_note,
        "sessions": sessions,
        "folds": list(folds),
        "weight_run": str(run_dir),
        "device": str(device),
        "preprocess": "game_phase4_like",
        "extra_meta": extra_meta or {},
    }
    (result_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_lines = [
        f"# 伪在线实验记录（{stamp} / {out_name}）",
        "",
        f"- protocol：`{PROTOCOL}`",
        f"- model：`{model_name}` | {structure_note}",
        f"- weights：`{run_dir}`",
        f"- preprocess：`game_phase4_like`",
        f"- 主指标：**段级 Acc_paper**",
        "",
    ]

    for name in sessions:
        print(f"\n=== stream {name} ===", flush=True)
        stream = build_eval_stream(
            SESSIONS_ROOT / name, apply_filter=not args.no_filter
        )
        save_stream_artifacts(stream, DOCS_OUT / "out" / name)
        subj = stream.subject_id
        print(
            f"  subject={subj} segs={stream.meta['n_segments']} "
            f"wins={stream.meta['n_windows']}",
            flush=True,
        )
        md_lines.append(f"## {subj} / {name}")
        md_lines.append("")

        for head, n_classes, y in (
            ("task", 2, stream.y_task),
            ("three", 3, stream.y_three),
        ):
            if head == "three" and args.skip_three:
                md_lines.extend(["### three", "", "- （本次跳过）", ""])
                continue
            fold_rows = []
            for fold in folds:
                print(f"[{subj}] {model_name} {head} fold{fold}", flush=True)
                net = load_fold_model(
                    build_model, run_dir, head=head, fold=fold, device=device
                )
                pred = predict_windows(
                    net, stream.X, device, batch_size=args.batch_size
                )
                agg = aggregate_windows_to_segments(
                    y, pred, stream.seg_keys, n_classes=n_classes
                )
                fold_rows.append(
                    {
                        "fold": fold,
                        "segment_metrics": agg["segment_metrics"],
                        "window_metrics": agg["window_metrics"],
                    }
                )
                del net
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            summ = _summarize_folds(fold_rows, head=head)
            summ["model"] = model_name
            summ["run"] = run_dir.name
            summ["n_windows"] = int(stream.X.shape[0])
            summary = {
                "subject_id": subj,
                "session": name,
                "head": head,
                "n_windows": int(stream.X.shape[0]),
                "model": summ,
            }
            (result_root / f"{subj}_{head}_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            md_lines.extend(_md_table(summ, head=head))
            print(
                f"  → Acc_paper {summ['acc_paper_mean']:.4f}±{summ['acc_paper_std']:.4f}",
                flush=True,
            )

    md_path = result_root / f"{out_name}实验结果.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\n写入 {md_path}", flush=True)
    print("done", flush=True)
