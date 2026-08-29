"""E1f 四成员切换验收：配置 · 离线 prob 融合 · 在线权重加载（若有 .pt）。

用法（仓库根，conda cyy）：
  python experiment_game/tools/test_e1f_switch.py
  python experiment_game/tools/test_e1f_switch.py --offline-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "code" / "train_lab" / "src" / "step" / "5090_ens_recipe_3s_hop100_accpaper"))

from adapt_engine.e1f import E1fStackConfig  # noqa: E402
from experiment_game.experiment.registry_factory import (  # noqa: E402
    build_registry,
    is_e1f_mode,
    verify_registry_paths,
)
from experiment_game.experiment.v3_config import V3Config  # noqa: E402


def _check_yaml() -> dict:
    cfg = V3Config.load_yaml()
    out = {
        "readout_mode": cfg.readout_mode,
        "primary_judge_mode": cfg.primary_judge_mode,
        "e1f_config_path": cfg.e1f_config_path,
        "is_e1f": is_e1f_mode(cfg),
        "verify_errors": cfg.verify_errors(),
    }
    return out


def _offline_e1f_replay() -> dict:
    from e1_fusion_core import (  # type: ignore
        acc_paper_for_split,
        fuse_pipeline,
        simulate_conf_early_stop,
    )
    from member_runs import member_run_dirs  # type: ignore
    from prob_io import load_members  # type: ignore

    stack = E1fStackConfig.load_json(
        _REPO / "experiment_game/config/e1f_four_member.json",
        repo_root=_REPO,
    )
    names = [m.name for m in stack.members]
    run_dirs = member_run_dirs(names)
    members = load_members(run_dirs)
    fusion = stack.fusion
    fused = fuse_pipeline(
        members,
        temperatures=list(fusion.temperatures),
        weights=fusion.weights,
        smooth_radius=fusion.smooth_radius,
    )
    if fusion.tau_conf is not None:
        fused = simulate_conf_early_stop(fused, tau_conf=fusion.tau_conf)
    test_acc = acc_paper_for_split(fused, "test")
    anchor = float(stack.test_acc_paper or 0.6173)
    delta_pp = (test_acc - anchor) * 100.0
    return {
        "test_acc_paper": test_acc,
        "anchor": anchor,
        "delta_pp": delta_pp,
        "pass": abs(delta_pp) <= 0.05,
        "members": names,
        "fusion": fusion.__dict__,
    }


def _online_weight_smoke() -> dict:
    cfg = V3Config.load_yaml()
    missing = verify_registry_paths(cfg, repo_root=_REPO)
    if missing:
        return {"ok": False, "missing": missing}
    try:
        reg = build_registry(cfg, repo_root=_REPO)
        import numpy as np

        win = np.random.default_rng(0).normal(0, 1, (8, 750)).astype(np.float32)
        heads = reg.forward_heads(win)
        p3 = heads["p_three"]
        return {
            "ok": True,
            "registry": type(reg).__name__,
            "p_three_shape": list(np.asarray(p3).shape),
            "p_three_sum": float(np.sum(p3)),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-only", action="store_true")
    args = ap.parse_args()

    report: dict = {"yaml": _check_yaml()}
    print("=== v3_session.yaml ===")
    print(json.dumps(report["yaml"], ensure_ascii=False, indent=2))

    print("\n=== offline E1f replay (published fusion) ===", flush=True)
    report["offline"] = _offline_e1f_replay()
    print(json.dumps(report["offline"], ensure_ascii=False, indent=2))

    if not args.offline_only:
        print("\n=== online weight smoke ===", flush=True)
        report["online"] = _online_weight_smoke()
        print(json.dumps(report["online"], ensure_ascii=False, indent=2))

    out = _REPO / "experiment_game/data/models/e1f_switch_test_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")

    ok_offline = report["offline"].get("pass", False)
    ok_mode = report["yaml"].get("is_e1f") and report["yaml"].get("readout_mode") == "e1f"
    ok_online = report.get("online", {}).get("ok", False)
    if ok_offline and ok_mode:
        if ok_online:
            print("\nPASS: E1f config + offline replay + online weights.")
            return 0
        print("\nPASS offline/config; online 待同步 5090 成员 fold0/best_*.pt 后启动 v3。")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
