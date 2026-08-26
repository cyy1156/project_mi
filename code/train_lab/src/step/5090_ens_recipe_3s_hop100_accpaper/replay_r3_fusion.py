"""方案 26 · R3 替换 conformer 后的四成员融合复评。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG24 = HERE.parent / "5090_baselines_openbmi_3s_hop100_accpaper"
PY = sys.executable

R3_RUN = (
    Path(__file__).resolve().parents[3]
    / "out"
    / "5090_ens_recipe_3s_hop100_accpaper"
    / "conformer_recipe_r3_openbmi_3s_hop100_balbatch_accpaper"
    / "openbmi_3s_hop100"
    / "run_20260825_163632"
)
R3_THREE = R3_RUN / "three"
E1F_BEST = 0.6173456790123457
FUSION_ADOPT_PP = 0.3


def _ensure_r3_dump() -> None:
    dumps = list(R3_THREE.glob("fold*/prob_dump_three.csv"))
    if len(dumps) >= 5:
        print(f"R3 prob dump OK ({len(dumps)} folds)")
        return
    script = """
import sys
from pathlib import Path
HERE = Path(r"%s")
PKG24 = Path(r"%s")
sys.path.insert(0, str(PKG24))
sys.path.insert(0, str(HERE))
from patch_recipe import install_recipe
install_recipe("R3")
from baseline_conformer import build_model
from dump_probs import dump_run
from shared_hparams import SHARED
dump_run(
    run_dir=Path(r"%s"),
    stage="three",
    build_model=build_model,
    hp=SHARED,
)
""" % (HERE, PKG24, R3_RUN)
    print("Dumping R3 prob...")
    subprocess.check_call([PY, "-c", script], cwd=str(PKG24))


def _replay(arm: str, out: Path) -> dict:
    cmd = [
        PY,
        str(HERE / "replay_e1.py"),
        "--arm",
        arm,
        "--four-member",
        "--conformer-run",
        str(R3_THREE),
        "--out",
        str(out),
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(HERE))
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> None:
    _ensure_r3_dump()
    e1d = _replay("E1d", HERE / "replay_r3_e1d.json")
    e1f = _replay("E1f", HERE / "replay_r3_e1f.json")
    best = e1f if e1f["test_acc_paper"] >= e1d["test_acc_paper"] else e1d
    delta_pp = (best["test_acc_paper"] - E1F_BEST) * 100.0
    verdict = "adopt" if delta_pp >= FUSION_ADOPT_PP else ("report" if delta_pp >= 0.1 else "negative")
    summary = {
        "arm": "R3_fusion_replay",
        "r3_run": str(R3_THREE),
        "e1f_anchor_best": E1F_BEST,
        "e1d_r3": {"test_acc_paper": e1d["test_acc_paper"], "delta_pp_vs_E_uniform": e1d["delta_test_pp_vs_E_uniform"]},
        "e1f_r3": {"test_acc_paper": e1f["test_acc_paper"], "delta_pp_vs_E_uniform": e1f["delta_test_pp_vs_E_uniform"]},
        "best_readout": best["arm"],
        "test_acc_paper": best["test_acc_paper"],
        "delta_pp_vs_E1f": delta_pp,
        "adopt_ge_pp_vs_E1f": FUSION_ADOPT_PP,
        "verdict": verdict,
    }
    out_path = HERE / "replay_r3_fusion.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
