"""按「下一场 session 在线试次准确率」重排实验 27 配方。"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from experiment_game.tools.exp27_cross_session_eval import eval_ckpt_on_session
from experiment_game.tools.exp27_fnz_replay_grid import ARMS, save_winner_ckpt
from experiment_game.tools.ft_subject_from_v3 import DEFAULT_THREE

WS02 = _REPO / "experiment_game/data/sessions/fnz_ws02_20260826_171537"
WS03 = _REPO / "experiment_game/data/sessions/fnz_ws03_20260826_174526"
OUT = _REPO / "experiment_game/data/models/fnz/exp27/online_leaderboard.json"

S_ARMS = ["A0", "A1", "B2", "B3", "D1", "E1"]
M_ARMS = ["A0", "B1", "B3", "E1", "E2"]


def _score_online(tm: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    acc = float(tm["acc_trial_majority"])
    mx = float(tm["max_class_frac"])
    pc = tm.get("pred_counts") or {}
    n_right = int(pc.get("Right", 0))
    n_trials = int(tm["n_trials"])
    penalty = 0.0
    if mx >= 0.65:
        penalty += 0.15
    if n_right == 0 and n_trials >= 10:
        penalty += 0.10
    return acc - penalty, {
        "online_score": acc - penalty,
        "penalties": penalty,
        "n_right_pred_trials": n_right,
    }


def _eval_arm(track: str, arm_id: str, test_session: Path, device: str) -> Dict[str, Any]:
    if arm_id == "BASE":
        ckpt = DEFAULT_THREE
    else:
        ckpt = save_winner_ckpt(track, arm_id, device=device)
    online = eval_ckpt_on_session(ckpt, test_session, device)
    tm = online["trial_metrics"]
    sc, extra = _score_online(tm)
    row: Dict[str, Any] = {
        "arm": arm_id,
        "online_trial_acc": tm["acc_trial_majority"],
        "online_window_acc": online["acc_window"],
        "pred_trials": tm["pred_counts"],
        "max_class_frac": tm["max_class_frac"],
        **extra,
    }
    if arm_id != "BASE":
        spec = next(a for a in ARMS if a.arm_id == arm_id)
        row.update(
            pool=spec.pool,
            replay_ratio=spec.replay_ratio,
            head_only=spec.head_only,
        )
    return row


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    s_rows: List[Dict[str, Any]] = []
    m_rows: List[Dict[str, Any]] = []

    print("=== S: ws01 FT → ws02 ONLINE ===", flush=True)
    for arm_id in ["BASE"] + S_ARMS:
        print(f"  {arm_id}", flush=True)
        row = _eval_arm("S", arm_id, WS02, device)
        s_rows.append(row)
        print(
            f"    trial={row['online_trial_acc']:.3f} score={row['online_score']:.3f} "
            f"pred={row['pred_trials']}",
            flush=True,
        )

    print("\n=== M: ws01+ws02 FT → ws03 ONLINE ===", flush=True)
    for arm_id in ["BASE"] + M_ARMS:
        print(f"  {arm_id}", flush=True)
        row = _eval_arm("M", arm_id, WS03, device)
        m_rows.append(row)
        print(
            f"    trial={row['online_trial_acc']:.3f} score={row['online_score']:.3f} "
            f"pred={row['pred_trials']}",
            flush=True,
        )

    s_rows.sort(key=lambda r: -r["online_score"])
    m_rows.sort(key=lambda r: -r["online_score"])

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "criterion": "next-session trial majority acc; score = acc - collapse penalties",
        "S_ws01_to_ws02": s_rows,
        "M_merge_to_ws03": m_rows,
        "winner_S_online": s_rows[0]["arm"],
        "winner_M_online": m_rows[0]["arm"],
        "heldout_winners": {"S": "B3", "M": "E1"},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nS online winner: {payload['winner_S_online']} (heldout was B3)")
    print(f"M online winner: {payload['winner_M_online']} (heldout was E1)")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
