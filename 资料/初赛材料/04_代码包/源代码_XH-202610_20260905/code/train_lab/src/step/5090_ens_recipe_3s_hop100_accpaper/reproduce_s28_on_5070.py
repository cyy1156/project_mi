"""5070 复现：方案 26 E1f（最优四成员）+ 方案 28 建议三成员（砍 T-shallow）。

前置（本机常缺，因 .gitignore 排除 *.pt 与 prob_dump_three.csv）：
  从 5090 同步下列四个 run 的 three/ 下 fold*/prob_dump_three.csv
  （或同步 fold*/best_three.pt 后在本机 dump）：

  code/train_lab/out/5090_alg_incr_3s_hop100_accpaper/
    shallow_.../run_20260823_095327/three/fold*/prob_dump_three.csv
    shallow_.../run_20260823_123900/three/...   # T-shallow，仅 E1f 四成员需要
    eegnet_.../run_20260823_131435/three/...
    conformer_.../run_20260823_135213/three/...

用法（在仓库根，conda cyy）：
  python code/train_lab/src/step/5090_ens_recipe_3s_hop100_accpaper/reproduce_s28_on_5070.py
  python .../reproduce_s28_on_5070.py --verify-only   # 只校验已有 JSON 锚点
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from member_runs import member_run_dirs  # noqa: E402
from prob_io import load_run_three  # noqa: E402
from s26_config import ANCHOR_E_UNIFORM  # noqa: E402
from s28_config import ANCHOR_E1F, ANCHOR_S3, SANITY_TOL  # noqa: E402

PY = sys.executable
OUT_DIR = HERE / "_reproduce_5070"
ANCHOR_E1F_REG = 0.6173


def _dumps_ok(names: list[str]) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    ok = True
    try:
        dirs = member_run_dirs(names)
    except Exception as exc:  # noqa: BLE001
        return False, [str(exc)]
    for name, run_dir in zip(names, dirs):
        if not run_dir.is_dir():
            ok = False
            msgs.append(f"MISSING dir {name}: {run_dir}")
            continue
        try:
            data = load_run_three(run_dir)
            n_val = int((data["split"] == "val").sum())
            n_test = int((data["split"] == "test").sum())
            msgs.append(f"OK {name}: val={n_val} test={n_test}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            msgs.append(f"NO DUMP {name}: {run_dir} ({exc})")
    return ok, msgs


def _run_e1f(members: str, out: Path) -> dict:
    cmd = [
        PY,
        str(HERE / "replay_e1.py"),
        "--arm",
        "E1f",
        "--members",
        members,
        "--out",
        str(out),
    ]
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(HERE))
    return json.loads(out.read_text(encoding="utf-8"))


def verify_published_json() -> dict:
    """不依赖 dump：核对仓库内已提交的 replay JSON 与登记锚点。"""
    e1f = json.loads((HERE / "replay_e1f.json").read_text(encoding="utf-8"))
    r6 = json.loads((HERE / "replay_r6.json").read_text(encoding="utf-8"))
    rows = {
        "published_E1f_test": e1f["test_acc_paper"],
        "published_R6_test": r6["test_acc_paper"],
        "anchor_E1f": ANCHOR_E1F_REG,
        "E1f_matches_anchor": abs(e1f["test_acc_paper"] - ANCHOR_E1F_REG) <= SANITY_TOL,
        "R6_matches_E1f": abs(r6["test_acc_paper"] - e1f["test_acc_paper"]) <= SANITY_TOL,
        "E1f_config": e1f.get("config"),
        "R6_members": r6.get("member_names"),
    }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report: dict = {
        "stamp": stamp,
        "machine": "5070_laptop",
        "published_json_check": verify_published_json(),
    }
    print("=== published JSON anchors ===", flush=True)
    print(json.dumps(report["published_json_check"], ensure_ascii=False, indent=2), flush=True)

    if args.verify_only:
        out_p = OUT_DIR / f"verify_only_{stamp}.json"
        out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_p}", flush=True)
        return

    four = ["shallow", "t_shallow", "eegnet", "conformer"]
    three = ["shallow", "eegnet", "conformer"]

    ok4, msg4 = _dumps_ok(four)
    ok3, msg3 = _dumps_ok(three)
    report["dump_check_four"] = {"ok": ok4, "messages": msg4}
    report["dump_check_three"] = {"ok": ok3, "messages": msg3}
    print("=== dump check (4-member) ===", flush=True)
    print("\n".join(msg4), flush=True)
    print("=== dump check (3-member) ===", flush=True)
    print("\n".join(msg3), flush=True)

    if not ok4 and not ok3:
        report["status"] = "blocked_no_dumps"
        report["how_to_sync"] = [
            "在 5090 上打包 fold*/prob_dump_three.csv（或 best_three.pt）到本机对应相对路径",
            "robocopy 示例（在 5090 仓库根）:",
            r"  robocopy code\train_lab\out\5090_alg_incr_3s_hop100_accpaper \\目标\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper *.csv /S",
            "或拷贝各 run 的 three/fold0..fold4/best_three.pt 后对本机执行 dump_member_probs.py",
        ]
        out_p = OUT_DIR / f"blocked_{stamp}.json"
        out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nBLOCKED: no prob dumps on this machine. See {out_p}", flush=True)
        sys.exit(2)

    if ok4:
        out4 = OUT_DIR / f"e1f_four_{stamp}.json"
        report["e1f_four"] = _run_e1f("shallow,t_shallow,eegnet,conformer", out4)
        report["e1f_four_path"] = str(out4)
        t = report["e1f_four"]["test_acc_paper"]
        print(
            f"E1f four-member test={t:.4f} Δvs anchor={(t - ANCHOR_E1F_REG) * 100:.3f} pp",
            flush=True,
        )

    if ok3:
        out3 = OUT_DIR / f"e1f_three_no_tshallow_{stamp}.json"
        report["e1f_three_sec28"] = _run_e1f("shallow,eegnet,conformer", out3)
        report["e1f_three_path"] = str(out3)
        t3 = report["e1f_three_sec28"]["test_acc_paper"]
        print(
            f"E1f three-member (no T-shallow) test={t3:.4f} "
            f"Δvs E1f_anchor={(t3 - ANCHOR_E1F_REG) * 100:.3f} pp "
            f"Δvs S3={(t3 - ANCHOR_S3) * 100:.3f} pp "
            f"Δvs E_uniform={(t3 - ANCHOR_E_UNIFORM) * 100:.3f} pp",
            flush=True,
        )
        if ok4:
            t4 = report["e1f_four"]["test_acc_paper"]
            print(f"Δ (three − four) = {(t3 - t4) * 100:.3f} pp", flush=True)

    report["status"] = "ok"
    out_p = OUT_DIR / f"report_{stamp}.json"
    out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = OUT_DIR / f"report_{stamp}.md"
    lines = [
        f"# 5070 复现 E1f / 方案28 三成员 · {stamp}",
        "",
        f"- 最优四成员 E1f test: {report.get('e1f_four', {}).get('test_acc_paper', '—')}",
        f"- 方案28 建议三成员（无 T-shallow）test: {report.get('e1f_three_sec28', {}).get('test_acc_paper', '—')}",
        f"- 登记锚点 E1f: {ANCHOR_E1F_REG}",
        "",
        f"JSON: `{out_p.name}`",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_p}\nwrote {md}", flush=True)


if __name__ == "__main__":
    main()
