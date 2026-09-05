"""06：05 OpenBMI 前半 FT 权重 · 后半 + 在线门控（H0–H3）。

方案：`资料/伪在线实验/06_旁路_OpenBMI_前半FT后半门控/方案.md`
只读 05 FT ckpt；不重训；产物写 06/results/。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from braindecode.models import ShallowFBCSPNet

HERE = Path(__file__).resolve().parent
FT_PKG = HERE.parent / "game_ft_openbmi_hop100_accpaper"
for p in (HERE, FT_PKG):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
sys.path.insert(0, str(FT_PKG))
sys.path.insert(0, str(HERE))

from channel_remap import OPENBMI_CHANS, remap_windows_to_openbmi
from config import DEFAULT_SESSIONS, N_FOLDS, N_TIMES, REPO_ROOT, SESSIONS_ROOT
from data_split import assert_no_leakage, build_trial_split, window_mask_for_trials
from gated_segment_metrics import aggregate_windows_to_segments_gated
from online_gate import build_gate_keeps, gate_stats
from stream import build_eval_stream

DOCS_06 = REPO_ROOT / "资料" / "伪在线实验" / "06_旁路_OpenBMI_前半FT后半门控"
RESULTS = DOCS_06 / "results"
FT_WEIGHT_ROOT = (
    REPO_ROOT
    / "code"
    / "train_lab"
    / "out"
    / "openbmi_game_ft_hop100_accpaper"
)
DEFAULT_FT_STAMP = "20260809_174914"
FT_NAME = "shallow_openbmi_game_ft_half_balbatch_accpaper"


def build_shallow(n_chans, n_times, n_outputs, drop_prob):
    return ShallowFBCSPNet(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, drop_prob=drop_prob
    )


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=0))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


@torch.no_grad()
def predict_windows(model, X, device, *, batch_size: int = 64) -> np.ndarray:
    model.eval()
    if X.ndim == 4 and X.shape[1] == 1:
        Xb = X[:, 0, :, :]
    elif X.ndim == 3:
        Xb = X
    else:
        raise ValueError(f"意外 X shape={X.shape}")
    preds: list[np.ndarray] = []
    for i in range(0, len(Xb), batch_size):
        batch = torch.from_numpy(
            np.asarray(Xb[i : i + batch_size], dtype=np.float32)
        ).to(device)
        logits = model(batch)
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    if not preds:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(preds, axis=0)


def load_ft_fold(
    *,
    ft_subject_root: Path,
    head: str,
    fold: int,
    device: torch.device,
):
    path = ft_subject_root / head / f"fold{fold}" / f"best_{head}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams") or {}
    drop = float(hp.get("drop_prob", 0.5))
    n_out = 2 if head == "task" else 3
    ckpt_n = ckpt.get("n_outputs")
    if ckpt_n is not None and int(ckpt_n) != n_out:
        raise RuntimeError(f"{path}: n_outputs={ckpt_n} 与 head={head} 不符")
    model = build_shallow(8, N_TIMES, n_out, drop).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="06：05 FT · 后半门控")
    p.add_argument("--ft-stamp", default=DEFAULT_FT_STAMP)
    p.add_argument("--sessions", default=",".join(DEFAULT_SESSIONS))
    p.add_argument("--gates", default="H0,H1,H2,H3")
    p.add_argument("--tasks", default="task,three")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--g3-top-p", type=float, default=0.5)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    gates = tuple(g.strip() for g in args.gates.split(",") if g.strip())
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    folds = (0,) if args.smoke else tuple(range(N_FOLDS))
    device = torch.device(args.device)

    ft_root = FT_WEIGHT_ROOT / FT_NAME / args.ft_stamp
    if not ft_root.is_dir():
        raise FileNotFoundError(f"缺少 05 FT 目录: {ft_root}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_run = RESULTS / f"{stamp}_shallow_ft_gated"
    out_run.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "phase": "06_ft_gated",
        "model": "shallow",
        "ft_stamp": args.ft_stamp,
        "ft_root": str(ft_root),
        "base_arm": "05",
        "gates": list(gates),
        "sessions": sessions,
        "device": str(device),
        "subjects": {},
    }

    md = [
        "# 06 OpenBMI 前半 FT · 后半门控 · shallow",
        "",
        f"- 时间：`{stamp}`",
        f"- FT 权重（05）：`{ft_root}`",
        f"- 门控：`{', '.join(gates)}`（仅后半段计分；H1≈G2）",
        f"- 通道重排：游戏序 → `{', '.join(OPENBMI_CHANS)}`",
        "",
    ]

    for ses_name in sessions:
        print(f"\n=== {ses_name} ===", flush=True)
        ses_dir = SESSIONS_ROOT / ses_name
        stream = build_eval_stream(ses_dir, zscore_windows=True)
        stream_q = build_eval_stream(ses_dir, zscore_windows=False)
        assert len(stream.X) == len(stream_q.X)

        ch_names = list(stream.meta.get("channels") or [])
        X_model = remap_windows_to_openbmi(stream.X, ch_names)
        print(
            f"  remap: {ch_names} -> {OPENBMI_CHANS} X={tuple(X_model.shape)}",
            flush=True,
        )

        split = build_trial_split(stream, val_ratio=0.2, seed=42)
        assert_no_leakage(split)
        eval_mask = window_mask_for_trials(stream.trial_ids, split.eval_trials)
        if int(eval_mask.sum()) == 0:
            raise RuntimeError(f"{ses_name}: eval half 无窗")

        keeps_full = build_gate_keeps(
            stream_q.X,
            stream.y_three,
            stream.trial_ids,
            stream.segs,
            ch_names=ch_names,
            top_p=float(args.g3_top_p),
            modes=gates,
        )
        # 切片到后半，避免前半段进入 abstain 分母
        segs_e = stream.segs[eval_mask]
        keys_e = stream.seg_keys[eval_mask]
        keeps_e = {g: np.asarray(keeps_full[g][eval_mask], dtype=bool) for g in gates}
        if "H0" in keeps_e:
            keeps_e["H0"] = np.ones(int(eval_mask.sum()), dtype=bool)

        subj = stream.subject_id
        ft_subj = ft_root / subj
        if not ft_subj.is_dir():
            raise FileNotFoundError(ft_subj)

        sub_pack: dict = {
            "session": ses_name,
            "n_eval_windows": int(eval_mask.sum()),
            "n_eval_trials": split.n_eval,
            "ft_subject_root": str(ft_subj),
            "gates": {},
        }
        for g, k in keeps_e.items():
            st = gate_stats(k, segs_e, keys_e)
            sub_pack["gates"][g] = st
            print(
                f"  {g}: kept={st['n_kept']}/{int(eval_mask.sum())} "
                f"mi_seg_abstain={st['n_mi_segments_abstain']}/{st['n_mi_segments']}",
                flush=True,
            )

        md += [
            f"## {subj} / {ses_name}",
            "",
            f"- eval_trials={split.n_eval} · eval_windows={int(eval_mask.sum())}",
            f"- FT：`{ft_subj}`",
            "",
        ]
        task_summ: dict = {}

        for head in tasks:
            n_classes = 2 if head == "task" else 3
            y_e = (stream.y_task if head == "task" else stream.y_three)[eval_mask]
            per_gate: dict[str, list[dict]] = {g: [] for g in gates}

            for fold in folds:
                print(f"[{subj}] shallow-ft {head} fold{fold}", flush=True)
                net = load_ft_fold(
                    ft_subject_root=ft_subj,
                    head=head,
                    fold=fold,
                    device=device,
                )
                pred_all = predict_windows(
                    net, X_model, device, batch_size=int(args.batch_size)
                )
                pred_e = pred_all[eval_mask]
                del net
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                for g in gates:
                    agg = aggregate_windows_to_segments_gated(
                        y_e, pred_e, keys_e, keeps_e[g], n_classes=n_classes
                    )
                    m = agg["segment_metrics"]
                    row = {
                        "fold": fold,
                        "acc_paper": float(m["acc_paper"])
                        if m.get("acc_paper") is not None
                        else float("nan"),
                        "abstain_rate": float(m["abstain_rate"]),
                        "abstain_as_wrong_acc": float(m["abstain_as_wrong_acc"])
                        if m.get("abstain_as_wrong_acc") is not None
                        else float("nan"),
                        "n_segments_scored": int(m["n_segments_scored"]),
                        "n_segments_all": int(m["n_segments_all"]),
                    }
                    per_gate[g].append(row)
                    print(
                        f"  {g}: acc_paper={row['acc_paper']:.4f} "
                        f"abstain={row['abstain_rate']:.3f} "
                        f"scored={row['n_segments_scored']}/{row['n_segments_all']}",
                        flush=True,
                    )

            h0_mean = _mean_std([r["acc_paper"] for r in per_gate.get("H0", [])])[0]
            summ_g = {}
            for g, rows in per_gate.items():
                mean, std = _mean_std([r["acc_paper"] for r in rows])
                ab_m, ab_s = _mean_std([r["abstain_rate"] for r in rows])
                summ_g[g] = {
                    "acc_paper_mean": mean,
                    "acc_paper_std": std,
                    "delta_vs_H0": float(mean - h0_mean)
                    if np.isfinite(mean) and np.isfinite(h0_mean)
                    else float("nan"),
                    "abstain_rate_mean": ab_m,
                    "abstain_rate_std": ab_s,
                    "folds": rows,
                }
            task_summ[head] = summ_g

            md += [
                f"### {head}",
                "",
                "| 代号 | Acc_paper | vs H0 Δ | abstain 率 | scored段/折 |",
                "|------|-----------|---------|------------|-------------|",
            ]
            for g in gates:
                s = summ_g[g]
                md.append(
                    f"| {g} | {s['acc_paper_mean']:.4f}±{s['acc_paper_std']:.4f} | "
                    f"{s['delta_vs_H0']:+.4f} | "
                    f"{s['abstain_rate_mean']:.3f}±{s['abstain_rate_std']:.3f} | "
                    f"{_mean_std([float(r['n_segments_scored']) for r in s['folds']])[0]:.1f} |"
                )
            md.append("")

            if head == "three" and "H1" in summ_g and "H0" in summ_g:
                d = summ_g["H1"]["delta_vs_H0"]
                tip = (
                    f"H1 ΔThree={d:+.4f} → 达 +0.02"
                    if d >= 0.02
                    else f"H1 ΔThree={d:+.4f} → 未达 +0.02"
                )
                md += [f"- **决策（{subj} Three）**：{tip}", ""]
                task_summ["three_decision"] = tip

        results["subjects"][subj] = {**sub_pack, "metrics": task_summ}
        (out_run / f"{subj}_summary.json").write_text(
            json.dumps(_jsonable(results["subjects"][subj]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    md += ["## 汇总决策", ""]
    for subj, pack in results["subjects"].items():
        tip = pack.get("metrics", {}).get("three_decision")
        if tip:
            md.append(f"- **{subj}**：{tip}")
    md.append("")
    md.append("成功线：H1 vs H0 Acc_paper ≥ +0.02（分人）；H0 应≈05 后半 FT。")
    md.append("")

    (out_run / "summary.json").write_text(
        json.dumps(_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path = RESULTS / f"{stamp}_shallow_ft_gated.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_run / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[done] {md_path}", flush=True)


if __name__ == "__main__":
    main()
