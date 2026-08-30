"""真被试 Leave-Next · E1f so vs all4（force）复验。

臂：
  e1f_so_force   — 只 FT shallow，另三员冻结，E1f 融合读出；FAIL 仍晋升
  e1f_all4_force — 四员各自 FT 再融合；FAIL 仍晋升

主指标：F5（因果平滑 lookback=2 + 多数票）试次 MI acc

用法：
  python experiment_game/tools/run_real_subject_all4_vs_so.py --all
  python experiment_game/tools/run_real_subject_all4_vs_so.py --subject syj0828
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.e1f import E1fRegistry, E1fStackConfig, build_fn_for_arch  # noqa: E402
from experiment_game.pipeline.finetune import run_subject_finetune  # noqa: E402

from experiment_game.tools.run_leave_next_e1f_task_ramp import (  # noqa: E402
    RAMP_CYY,
    RAMP_FNZ,
    RAMP_SYJ,
    _list_v3_sessions as list_v3_sessions,
    eval_f5_e1f,
)

E1F_CONFIG = _REPO / "experiment_game" / "config" / "e1f_four_member.json"
ANALYSIS_ROOT = _REPO / "experiment_game" / "data" / "subjects" / "_analysis"
DOC_REG = (
    _REPO
    / "资料"
    / "模型训练"
    / "33_旁路_真被试LeaveNext_all4复验_syj_fnz_openbmi_accpaper"
    / "总结"
    / "结果登记表.md"
)


def _ramp_for_subject(subject_id: str) -> list:
    if subject_id == "syj0828":
        return list(RAMP_SYJ)
    if subject_id == "fnz0828":
        return list(RAMP_FNZ)
    if subject_id == "cyy0830":
        return list(RAMP_CYY)
    raise ValueError(f"未知被试 ramp: {subject_id}")


def _base_stack() -> E1fStackConfig:
    return E1fStackConfig.load_json(E1F_CONFIG, repo_root=_REPO).resolve_paths(
        repo_root=_REPO
    )


def _member_base() -> Dict[str, Dict[str, Any]]:
    stack = _base_stack()
    out: Dict[str, Dict[str, Any]] = {}
    for m in stack.members:
        out[m.name] = {
            "arch": m.arch,
            "three": Path(m.three_ckpt),
            "task": Path(m.task_ckpt) if m.task_ckpt else None,
        }
    return out


def _registry(
    member_three: Dict[str, Path],
    member_task: Dict[str, Path],
    *,
    device: str,
) -> E1fRegistry:
    overrides: Dict[str, Dict[str, str]] = {}
    for name, three in member_three.items():
        ent: Dict[str, str] = {"three_ckpt": str(three)}
        tk = member_task.get(name)
        if tk is not None and Path(tk).is_file():
            ent["task_ckpt"] = str(tk)
        overrides[name] = ent
    stack = _base_stack().with_member_overrides(overrides).resolve_paths(repo_root=_REPO)
    return E1fRegistry(stack, device=device)


def _pack_f5(blob: Dict[str, Any]) -> Dict[str, Any]:
    f5 = blob["f5"]
    return {
        "window_acc": blob.get("window_acc"),
        "mi_acc": f5.get("mi_acc"),
        "mi_ok": f5.get("mi_ok"),
        "mi_n": f5.get("mi_n"),
        "rest_acc": f5.get("rest_acc"),
        "rest_ok": f5.get("rest_ok"),
        "rest_n": f5.get("rest_n"),
        "score": f5.get("score"),
        "score_max": f5.get("score_max"),
        "by_label": f5.get("by_label"),
    }


def _eval_e1f(
    hold_dirs: Sequence[Path],
    *,
    device: str,
    member_three: Dict[str, Path],
    member_task: Dict[str, Path],
) -> Dict[str, Any]:
    reg = _registry(member_three, member_task, device=device)
    return _pack_f5(eval_f5_e1f(hold_dirs, device=device, e1f_registry=reg))


def run_arm(
    *,
    subject_id: str,
    arm: str,
    by_ws: Dict[str, Path],
    ramp: list,
    work: Path,
    device: str,
) -> List[Dict[str, Any]]:
    force = arm.endswith("_force")
    scope = "all4" if "all4" in arm else "so"
    work.mkdir(parents=True, exist_ok=True)
    rows_p = work / "rows.json"

    mb = _member_base()
    cur_three = {n: v["three"] for n, v in mb.items()}
    cur_task: Dict[str, Path] = {
        n: v["task"] for n, v in mb.items() if v["task"] and Path(v["task"]).is_file()
    }

    rows: List[Dict[str, Any]] = []
    if rows_p.is_file():
        rows = json.loads(rows_p.read_text(encoding="utf-8"))
    done_r = max((int(r["R"]) for r in rows), default=-1)

    def _save() -> None:
        rows_p.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _restore_from_rows() -> None:
        nonlocal cur_three, cur_task
        for r in rows:
            if not (r.get("promoted") and r.get("ft_dir")):
                continue
            ft = Path(r["ft_dir"])
            if r.get("ft_scope") == "all4":
                for name in mb:
                    t3 = ft / "members" / name / "best_three.pt"
                    tk = ft / "members" / name / "best_task.pt"
                    if t3.is_file():
                        cur_three[name] = t3
                    if tk.is_file():
                        cur_task[name] = tk
            else:
                t3 = ft / "best_three.pt"
                tk = ft / "best_task.pt"
                if t3.is_file():
                    cur_three["shallow"] = t3
                if tk.is_file():
                    cur_task["shallow"] = tk

    _restore_from_rows()

    # R0
    if done_r < 0:
        hold0 = by_ws[ramp[0][1]]
        base_three = {n: v["three"] for n, v in mb.items()}
        base_task = {
            n: v["task"] for n, v in mb.items() if v["task"] and Path(v["task"]).is_file()
        }
        f5 = _eval_e1f(
            [hold0], device=device, member_three=base_three, member_task=base_task
        )
        rows.append(
            {
                "R": 0,
                "subject_id": subject_id,
                "arm": arm,
                "ft_scope": scope,
                "force_promote": force,
                "train": [],
                "heldout": ramp[0][1],
                "ft": False,
                "promoted": False,
                "gate_pass": None,
                "f5": f5,
                "note": "底座零样本 E1f",
            }
        )
        _save()
        print(
            f"  [{arm}] R0 hold={ramp[0][1]} win={f5['window_acc']:.3f} MI={f5['mi_acc']:.3f}",
            flush=True,
        )
        done_r = 0

    for i, (train_keys, hold_key, use_replay) in enumerate(ramp, start=1):
        if i <= done_r:
            continue
        train_dirs = [by_ws[k] for k in train_keys]
        hold_dir = by_ws[hold_key]
        out_dir = work / f"R{i}_train_{'+'.join(train_keys)}_eval_{hold_key}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"  [{arm}] R{i} train={train_keys} hold={hold_key} "
            f"replay={'0.10' if use_replay else 'off'}",
            flush=True,
        )

        gate_pass = False
        if scope == "all4":
            for name, info in mb.items():
                mdir = out_dir / "members" / name
                mdir.mkdir(parents=True, exist_ok=True)
                init_3 = cur_three[name]
                init_t = cur_task.get(name) or cur_task.get("shallow") or mb["shallow"]["task"]
                print(f"    FT member={name} arch={info['arch']}", flush=True)
                run_subject_finetune(
                    train_dirs,
                    mdir,
                    task_ckpt=Path(init_t),
                    three_ckpt=Path(init_3),
                    heldout_session_dirs=[hold_dir],
                    no_replay=not use_replay,
                    replay_ratio=0.10 if use_replay else 0.0,
                    early_stop=True,
                    max_epochs=20,
                    patience=5,
                    verbose=False,
                    device=device,
                    build_fn=build_fn_for_arch(info["arch"]),
                )
            cand_three = {n: out_dir / "members" / n / "best_three.pt" for n in mb}
            cand_task = {n: out_dir / "members" / n / "best_task.pt" for n in mb}
            # 融合门控：优先读 shallow 成员 gate；并落盘说明 force 仍晋升
            rg = out_dir / "members" / "shallow" / "release_gate.json"
            gate: Dict[str, Any] = {}
            if rg.is_file():
                gate = json.loads(rg.read_text(encoding="utf-8"))
                gate_pass = bool(gate.get("pass") or gate.get("ok"))
            (out_dir / "release_gate_note.json").write_text(
                json.dumps(
                    {
                        "scope": "all4",
                        "gate_source": "members/shallow/release_gate.json",
                        "gate_pass": gate_pass,
                        "force_promote": force,
                        "shallow_gate": gate,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            shutil.copy2(cand_three["shallow"], out_dir / "best_three.pt")
            if cand_task["shallow"].is_file():
                shutil.copy2(cand_task["shallow"], out_dir / "best_task.pt")
            promoted = bool(gate_pass or force)
            if promoted:
                cur_three = cand_three
                cur_task = {n: p for n, p in cand_task.items() if p.is_file()}
        else:
            init_3 = cur_three["shallow"]
            init_t = cur_task.get("shallow") or mb["shallow"]["task"]
            result = run_subject_finetune(
                train_dirs,
                out_dir,
                task_ckpt=Path(init_t),
                three_ckpt=Path(init_3),
                heldout_session_dirs=[hold_dir],
                no_replay=not use_replay,
                replay_ratio=0.10 if use_replay else 0.0,
                early_stop=True,
                max_epochs=20,
                patience=5,
                verbose=False,
                device=device,
                build_fn=build_fn_for_arch("shallow"),
            )
            gate = result.get("release_gate") or {}
            gate_pass = bool(result.get("release_pass") or gate.get("pass"))
            promoted = bool(gate_pass or force)
            if promoted:
                cur_three["shallow"] = out_dir / "best_three.pt"
                cur_task["shallow"] = out_dir / "best_task.pt"

        f5 = _eval_e1f(
            [hold_dir], device=device, member_three=cur_three, member_task=cur_task
        )
        base_f5 = _eval_e1f(
            [hold_dir],
            device=device,
            member_three={n: v["three"] for n, v in mb.items()},
            member_task={
                n: v["task"]
                for n, v in mb.items()
                if v["task"] and Path(v["task"]).is_file()
            },
        )
        dlt = None
        if f5.get("mi_acc") is not None and base_f5.get("mi_acc") is not None:
            dlt = float(f5["mi_acc"]) - float(base_f5["mi_acc"])
        rows.append(
            {
                "R": i,
                "subject_id": subject_id,
                "arm": arm,
                "ft_scope": scope,
                "force_promote": force,
                "train": list(train_keys),
                "heldout": hold_key,
                "use_replay": use_replay,
                "ft": True,
                "gate_pass": gate_pass,
                "promoted": promoted,
                "ft_dir": str(out_dir),
                "f5": f5,
                "f5_base_e1f": base_f5,
                "delta_mi_vs_base": dlt,
            }
        )
        _save()
        print(
            f"    → gate={'PASS' if gate_pass else 'FAIL'} promote={promoted} "
            f"win={f5['window_acc']:.3f} MI={f5['mi_acc']:.3f} "
            f"(base {base_f5['mi_acc']:.3f}) "
            f"score={f5['score']:.1f}/{f5.get('score_max')}",
            flush=True,
        )
        done_r = i

    return rows


def write_registry(stamp: str, all_results: Dict[str, Dict[str, List[Dict]]]) -> Path:
    lines = [
        "# 实验 33 · 真被试 Leave-Next · so vs all4（force）",
        "",
        f"> stamp=`{stamp}` · 主指标 **F5 MI acc**  ",
        f"> 原始：`experiment_game/data/subjects/_analysis/exp33_real_all4_{stamp}/`",
        "",
        "## 末档总表",
        "",
        "| 被试 | 臂 | R | heldout | win | MI | Rest | score | ΔMI vs 底座 |",
        "|------|----|---|---------|-----|-----|------|-------|-------------|",
    ]
    for sid, arms in sorted(all_results.items()):
        for arm, rows in sorted(arms.items()):
            last = next((r for r in reversed(rows) if r.get("ft")), rows[-1])
            f5 = last.get("f5") or {}
            dlt = last.get("delta_mi_vs_base")

            def _f(key: str, nd: int = 3) -> str:
                v = f5.get(key)
                if v is None:
                    return "nan"
                return f"{float(v):.{nd}f}"

            lines.append(
                f"| {sid} | {arm} | R{last.get('R')} | {last.get('heldout')} | "
                f"{_f('window_acc')} | {_f('mi_acc')} | {_f('rest_acc')} | "
                f"{_f('score', 1)}/{f5.get('score_max', 54)} | "
                f"{'' if dlt is None else f'{float(dlt):+.3f}'} |"
            )
    lines += ["", "## 分档", ""]
    for sid, arms in sorted(all_results.items()):
        lines.append(f"### {sid}")
        lines.append("")
        for arm, rows in sorted(arms.items()):
            lines.append(f"**{arm}**")
            lines.append("")
            lines.append("| R | hold | win | MI | Rest | promote | gate |")
            lines.append("|---|------|-----|-----|------|---------|------|")
            for r in rows:
                f5 = r.get("f5") or {}

                def _f2(key: str) -> str:
                    v = f5.get(key)
                    return "nan" if v is None else f"{float(v):.3f}"

                lines.append(
                    f"| R{r.get('R')} | {r.get('heldout')} | "
                    f"{_f2('window_acc')} | {_f2('mi_acc')} | {_f2('rest_acc')} | "
                    f"{r.get('promoted')} | {r.get('gate_pass')} |"
                )
            lines.append("")
    lines += ["## 结论", ""]
    for sid, arms in sorted(all_results.items()):
        so = next((r for r in reversed(arms.get("e1f_so_force") or []) if r.get("ft")), None)
        a4 = next(
            (r for r in reversed(arms.get("e1f_all4_force") or []) if r.get("ft")), None
        )
        if so and a4:
            d = float(a4["f5"]["mi_acc"]) - float(so["f5"]["mi_acc"])
            lines.append(
                f"- **{sid}** 末档 MI：all4={a4['f5']['mi_acc']:.3f} · "
                f"so={so['f5']['mi_acc']:.3f} · Δ={d:+.3f}"
            )
    lines.append("")
    lines.append(
        "门槛：两人均 all4≥so，或平均 ΔMI≥+0.02 → 支持线上默认 all4。"
    )
    lines.append("")
    text = "\n".join(lines) + "\n"
    # 仅 syj/fnz 复验写正式登记表；其它被试写分析目录，避免覆盖
    only_legacy = set(all_results) <= {"syj0828", "fnz0828"}
    if only_legacy:
        DOC_REG.parent.mkdir(parents=True, exist_ok=True)
        DOC_REG.write_text(text, encoding="utf-8")
        return DOC_REG
    local = ANALYSIS_ROOT / f"exp33_real_all4_{stamp}" / "结果登记表.md"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(text, encoding="utf-8")
    return local


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subject",
        choices=("syj0828", "fnz0828", "cyy0830"),
        action="append",
    )
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--arms", type=str, default="e1f_so_force,e1f_all4_force")
    ap.add_argument("--stamp", type=str, default="")
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    subjects = list(args.subject or [])
    if args.all or not subjects:
        subjects = ["syj0828", "fnz0828"]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ANALYSIS_ROOT / f"exp33_real_all4_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_meta.json").write_text(
        json.dumps(
            {
                "stamp": stamp,
                "subjects": subjects,
                "arms": arms,
                "device": device,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    all_results: Dict[str, Dict[str, List[Dict]]] = {}
    for sid in subjects:
        print(f"\n==== {sid} ====", flush=True)
        by_ws = list_v3_sessions(sid)
        ramp = _ramp_for_subject(sid)
        print(f"  sessions={sorted(by_ws)}", flush=True)
        missing = sorted(
            {
                k
                for train, hold, _ in ramp
                for k in list(train) + [hold]
                if k not in by_ws
            }
        )
        if missing:
            raise SystemExit(f"{sid}: ramp 缺少 session {missing}；已有 {sorted(by_ws)}")
        all_results[sid] = {}
        for arm in arms:
            print(f"\n-- {sid} / {arm} --", flush=True)
            rows = run_arm(
                subject_id=sid,
                arm=arm,
                by_ws=by_ws,
                ramp=ramp,
                work=out_root / sid / arm,
                device=device,
            )
            all_results[sid][arm] = rows

    (out_root / "all_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    reg = write_registry(stamp, all_results)
    print(f"\nDONE stamp={stamp}\n  raw={out_root}\n  registry={reg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
